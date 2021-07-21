# -*- coding: utf-8 -*-
# Copyright 1999-2021 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import subprocess  # nosec
import sys
import time
from collections import namedtuple
from typing import List, Optional

import psutil

from .lib import nvutils

logger = logging.getLogger(__name__)

CGROUP_CPU_STAT_FILE = '/sys/fs/cgroup/cpuacct/cpuacct.stat'
CGROUP_MEM_STAT_FILE = '/sys/fs/cgroup/memory/memory.stat'

_proc = psutil.Process()
_timer = getattr(time, 'monotonic', time.time)

_cpu_use_process_stat = bool(int(os.environ.get('MARS_CPU_USE_PROCESS_STAT', '0').strip('"')))
_cpu_use_cgroup_stat = bool(int(os.environ.get('MARS_CPU_USE_CGROUP_STAT', '0').strip('"')))
_mem_use_process_stat = bool(int(os.environ.get('MARS_MEM_USE_PROCESS_STAT', '0').strip('"')))
_mem_use_cgroup_stat = bool(int(os.environ.get('MARS_MEM_USE_CGROUP_STAT', '0').strip('"')))

if 'MARS_USE_PROCESS_STAT' in os.environ:
    _cpu_use_process_stat = _mem_use_process_stat = \
        bool(int(os.environ['MARS_USE_PROCESS_STAT'].strip('"')))
if 'MARS_USE_CGROUP_STAT' in os.environ:
    _cpu_use_cgroup_stat = _mem_use_cgroup_stat = \
        bool(int(os.environ['MARS_USE_CGROUP_STAT'].strip('"')))

if 'MARS_CPU_TOTAL' in os.environ:
    _cpu_total = int(os.environ['MARS_CPU_TOTAL'].strip('"'))
else:
    _cpu_total = psutil.cpu_count(logical=True)

if 'MARS_MEMORY_TOTAL' in os.environ:
    _mem_total = int(os.environ['MARS_MEMORY_TOTAL'].strip('"'))
else:
    _mem_total = None

_virt_memory_stat = namedtuple('virtual_memory', 'total available percent used free')

_shm_path = [pt.mountpoint for pt in psutil.disk_partitions(all=True)
             if pt.mountpoint in ('/tmp', '/dev/shm') and pt.fstype == 'tmpfs']
if not _shm_path:
    _shm_path = None
else:
    _shm_path = _shm_path[0]


def _read_cgroup_stat_file():
    with open(CGROUP_MEM_STAT_FILE, 'r') as cg_file:
        contents = cg_file.read()
    kvs = dict()
    for line in contents.splitlines():
        parts = line.split(' ')
        if len(parts) == 2:
            kvs[parts[0]] = int(parts[1])
    return kvs


_root_pid = None


def virtual_memory() -> _virt_memory_stat:
    global _root_pid

    sys_mem = psutil.virtual_memory()
    if _mem_use_cgroup_stat:
        # see section 5.5 in https://www.kernel.org/doc/Documentation/cgroup-v1/memory.txt
        cgroup_mem_info = _read_cgroup_stat_file()
        total = cgroup_mem_info['hierarchical_memory_limit']
        total = min(_mem_total or total, total)
        used = cgroup_mem_info['rss'] + cgroup_mem_info.get('swap', 0)
        if _shm_path:
            shm_stats = psutil.disk_usage(_shm_path)
            used += shm_stats.used
        available = free = total - used
        percent = 100.0 * (total - available) / total
        return _virt_memory_stat(total, available, percent, used, free)
    elif not _mem_use_process_stat:
        total = min(_mem_total or sys_mem.total, sys_mem.total)
        used = sys_mem.used + getattr(sys_mem, 'shared', 0)
        available = sys_mem.available
        free = sys_mem.free
        percent = 100.0 * (total - available) / total
        return _virt_memory_stat(total, available, percent, used, free)
    else:
        used = 0
        if _root_pid is None:
            cur_proc = psutil.Process()
            while True:
                par_proc = cur_proc.parent()
                if par_proc is None:
                    break
                try:
                    cmd = par_proc.cmdline()
                    if 'python' not in ' '.join(cmd).lower():
                        break
                    cur_proc = par_proc
                except:  # noqa: E722  # nosec  # pylint: disable=bare-except  # pragma: no cover
                    break
            _root_pid = cur_proc.pid

        root_proc = psutil.Process(_root_pid)
        for p in root_proc.children(True):
            try:
                used += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if _shm_path:
            shm_stats = psutil.disk_usage(_shm_path)
            used += shm_stats.used

        total = min(_mem_total or sys_mem.total, sys_mem.total)
        # TODO sys_mem.available does not work in container
        # available = min(sys_mem.available, total - used)
        available = total - used
        free = min(sys_mem.free, total - used)
        percent = 100.0 * (total - available) / total
        return _virt_memory_stat(total, available, percent, used, free)


def cpu_count():
    return _cpu_total


_last_cgroup_cpu_measure = None
_last_proc_cpu_measure = None
_last_cpu_percent = None


def _take_process_cpu_snapshot():
    pts = dict()
    sts = dict()
    for p in psutil.process_iter():
        try:
            pts[p.pid] = p.cpu_times()
            sts[p.pid] = _timer()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return pts, sts


def cpu_percent():
    global _last_cgroup_cpu_measure, _last_proc_cpu_measure, _last_cpu_percent
    if _cpu_use_cgroup_stat:
        # see https://www.kernel.org/doc/Documentation/cgroup-v1/cpuacct.txt
        with open(CGROUP_CPU_STAT_FILE, 'r') as cgroup_file:
            cpu_acct = int(cgroup_file.read())
            sample_time = _timer()
        if _last_cgroup_cpu_measure is None:
            _last_cgroup_cpu_measure = (cpu_acct, sample_time)
            return None

        last_cpu_acct, last_sample_time = _last_cgroup_cpu_measure
        time_delta = sample_time - last_sample_time
        if time_delta < 1e-2:
            return _last_cpu_percent

        _last_cgroup_cpu_measure = (cpu_acct, sample_time)
        # nanoseconds / seconds * 100, we shall divide 1e7.
        _last_cpu_percent = round((cpu_acct - last_cpu_acct) / (sample_time - last_sample_time) / 1e7, 1)
        return _last_cpu_percent
    elif _cpu_use_process_stat:
        pts, sts = _take_process_cpu_snapshot()

        if _last_proc_cpu_measure is None:
            _last_proc_cpu_measure = (pts, sts)
            return None

        old_pts, old_sts = _last_proc_cpu_measure

        percents = []
        for pid in pts:
            if pid not in old_pts:
                continue
            pt1 = old_pts[pid]
            pt2 = pts[pid]
            delta_proc = (pt2.user - pt1.user) + (pt2.system - pt1.system)
            time_delta = sts[pid] - old_sts[pid]

            if time_delta < 1e-2:
                return _last_cpu_percent or 0
            percents.append((delta_proc / time_delta) * 100)
        _last_proc_cpu_measure = (pts, sts)
        _last_cpu_percent = round(sum(percents), 1)
        return _last_cpu_percent
    else:
        return sum(psutil.cpu_percent(percpu=True))


def disk_usage(d):
    return psutil.disk_usage(d)


def iowait():
    cpu_percent = psutil.cpu_times_percent()
    try:
        return cpu_percent.iowait
    except AttributeError:
        return None


_last_disk_io_metas = dict()
_path_to_device = dict()
_win_diskperf_called = False


def get_path_device(path: str):
    for part in psutil.disk_partitions(all=True):
        if path.startswith(part.mountpoint):
            return part.device
    return None


def _get_path_device(path: str):
    if path in _path_to_device:
        return _path_to_device[path]

    for part in psutil.disk_partitions(all=True):
        if path.startswith(part.mountpoint):
            dev_name = _path_to_device[path] = part.device.replace('/dev/', '')
            return dev_name
    _path_to_device[path] = None
    return None


_disk_io_usage_type = namedtuple('_disk_io_usage_type', 'reads writes')


def disk_io_usage(path=None) -> Optional[_disk_io_usage_type]:
    global _win_diskperf_called

    # Needed by psutil.disk_io_counters() under newer version of Windows.
    # diskperf -y need to be called or no disk information can be found.
    if sys.platform == 'win32' and not _win_diskperf_called:
        CREATE_NO_WINDOW = 0x08000000
        try:
            proc = subprocess.Popen(['diskperf', '-y'], shell=False,
                                    creationflags=CREATE_NO_WINDOW)  # nosec
            proc.wait()
        except (subprocess.CalledProcessError, OSError):  # pragma: no cover
            pass
        _win_diskperf_called = True

    if path is None:
        disk_counters = psutil.disk_io_counters()
    else:
        dev_to_counters = psutil.disk_io_counters(perdisk=True)
        disk_counters = dev_to_counters.get(_get_path_device(path))
        if disk_counters is None:
            return None
    tst = time.time()

    read_bytes = disk_counters.read_bytes
    write_bytes = disk_counters.write_bytes
    if path not in _last_disk_io_metas:
        _last_disk_io_metas[path] = (read_bytes, write_bytes, tst)
        return None

    last_read_bytes, last_write_bytes, last_time = _last_disk_io_metas[path]
    delta_time = tst - last_time
    if delta_time == 0:
        return None

    read_speed = (read_bytes - last_read_bytes) / delta_time
    write_speed = (write_bytes - last_write_bytes) / delta_time

    _last_disk_io_metas[path] = (read_bytes, write_bytes, tst)
    return _disk_io_usage_type(read_speed, write_speed)


_last_net_io_meta = None


def net_io_usage():
    global _last_net_io_meta

    net_counters = psutil.net_io_counters()
    tst = time.time()

    send_bytes = net_counters.bytes_sent
    recv_bytes = net_counters.bytes_recv
    if _last_net_io_meta is None:
        _last_net_io_meta = (send_bytes, recv_bytes, tst)
        return None

    last_send_bytes, last_recv_bytes, last_time = _last_net_io_meta
    delta_time = tst - last_time
    if delta_time == 0:
        return None

    recv_speed = (recv_bytes - last_recv_bytes) / delta_time
    send_speed = (send_bytes - last_send_bytes) / delta_time

    _last_net_io_meta = (send_bytes, recv_bytes, tst)
    return recv_speed, send_speed


_cuda_info = namedtuple('cuda_info', 'driver_version cuda_version products gpu_count')
_cuda_card_stat = namedtuple('cuda_card_stat', 'index product_name gpu_usage temperature fb_mem_info')


def cuda_info():  # pragma: no cover
    driver_info = nvutils.get_driver_info()
    if not driver_info:
        return
    gpu_count = nvutils.get_device_count()
    return _cuda_info(
        driver_version=driver_info.driver_version,
        cuda_version=driver_info.cuda_version,
        products=[nvutils.get_device_info(idx).name for idx in range(gpu_count)],
        gpu_count=gpu_count,
    )


def cuda_count():
    return nvutils.get_device_count() or 0


def cuda_card_stats() -> List[_cuda_card_stat]:  # pragma: no cover
    infos = []
    device_count = nvutils.get_device_count()
    if not device_count:
        return infos
    for device_idx in range(device_count):
        device_info = nvutils.get_device_info(device_idx)
        device_status = nvutils.get_device_status(device_idx)

        infos.append(_cuda_card_stat(
            index=device_info.index,
            product_name=device_info.name,
            gpu_usage=device_status.gpu_util,
            temperature=device_status.temperature,
            fb_mem_info=_virt_memory_stat(
                total=device_status.fb_total_mem, used=device_status.fb_used_mem,
                free=device_status.fb_free_mem, available=device_status.fb_free_mem,
                percent=device_status.mem_util,
            )
        ))
    return infos
