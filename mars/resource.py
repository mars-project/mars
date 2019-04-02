# -*- coding: utf-8 -*-
# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

import os
import subprocess  # nosec
import sys
import time
from collections import namedtuple

import psutil

_proc = psutil.Process()
_timer = getattr(time, 'monotonic', time.time)

if 'MARS_USE_PROCESS_STAT' in os.environ:
    _use_process_stat = bool(int(os.environ['MARS_USE_PROCESS_STAT'].strip('"')))
else:
    _use_process_stat = False

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
             if pt.mountpoint in ('/tmp', '/dev/shm') and pt.device == 'tmpfs']
if not _shm_path:
    _shm_path = None
else:
    _shm_path = _shm_path[0]


def virtual_memory():
    sys_mem = psutil.virtual_memory()
    if not _use_process_stat:
        total = sys_mem.total
        used = sys_mem.used + getattr(sys_mem, 'shared', 0)
        available = sys_mem.available
        free = sys_mem.free
        percent = 100.0 * (total - available) / total
        return _virt_memory_stat(total, available, percent, used, free)
    else:
        used = 0
        for p in psutil.process_iter():
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


_last_cpu_measure = None


def _take_process_cpu_snapshot():
    num_cpus = cpu_count() or 1

    def timer():
        return _timer() * num_cpus

    processes = [p for p in psutil.process_iter() if p.pid != _proc.pid]

    pts = dict()
    sts = dict()
    for p in processes:
        try:
            pts[p.pid] = p.cpu_times()
            sts[p.pid] = timer()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    pts[_proc.pid] = _proc.cpu_times()
    sts[_proc.pid] = timer()
    return pts, sts


def cpu_percent():
    global _last_cpu_measure
    if not _use_process_stat:
        return sum(psutil.cpu_percent(percpu=True))

    num_cpus = cpu_count() or 1
    pts, sts = _take_process_cpu_snapshot()

    if _last_cpu_measure is None:
        _last_cpu_measure = (pts, sts)
        return None

    old_pts, old_sts = _last_cpu_measure

    percents = []
    for pid in pts:
        if pid not in old_pts:
            continue
        pt1 = old_pts[pid]
        pt2 = pts[pid]
        delta_proc = (pt2.user - pt1.user) + (pt2.system - pt1.system)
        delta_time = sts[pid] - old_sts[pid]

        try:
            overall_cpus_percent = (delta_proc / delta_time) * 100
        except ZeroDivisionError:
            percents.append(0.0)
        else:
            single_cpu_percent = overall_cpus_percent * num_cpus
            percents.append(single_cpu_percent)
    _last_cpu_measure = (pts, sts)
    return round(sum(percents), 1)


def disk_usage(d):
    return psutil.disk_usage(d)


def iowait():
    cpu_percent = psutil.cpu_times_percent()
    try:
        return cpu_percent.iowait
    except AttributeError:
        return None


_last_disk_io_meta = None
_win_diskperf_called = False


def disk_io_usage():
    global _last_disk_io_meta, _win_diskperf_called

    # Needed by psutil.disk_io_counters() under newer version of Windows.
    # diskperf -y need to be called or no disk information can be found.
    if sys.platform == 'win32' and not _win_diskperf_called:  # pragma: no cover
        CREATE_NO_WINDOW = 0x08000000
        try:
            proc = subprocess.Popen(['diskperf', '-y'], shell=False,
                                    creationflags=CREATE_NO_WINDOW)  # nosec
            proc.wait()
        except (subprocess.CalledProcessError, OSError):
            pass
        _win_diskperf_called = True

    disk_counters = psutil.disk_io_counters()
    tst = time.time()

    read_bytes = disk_counters.read_bytes
    write_bytes = disk_counters.write_bytes
    if _last_disk_io_meta is None:
        _last_disk_io_meta = (read_bytes, write_bytes, tst)
        return None

    last_read_bytes, last_write_bytes, last_time = _last_disk_io_meta
    delta_time = tst - last_time
    read_speed = (read_bytes - last_read_bytes) / delta_time
    write_speed = (write_bytes - last_write_bytes) / delta_time

    _last_disk_io_meta = (read_bytes, write_bytes, tst)
    return read_speed, write_speed


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
    recv_speed = (recv_bytes - last_recv_bytes) / delta_time
    send_speed = (send_bytes - last_send_bytes) / delta_time

    _last_net_io_meta = (send_bytes, recv_bytes, tst)
    return recv_speed, send_speed


_last_nvml_output = None
_last_nvml_output_time = None


def _get_nvml_output(async_ctx=None):
    from xml.etree import ElementTree
    global _last_nvml_output, _last_nvml_output_time

    popen = getattr(async_ctx, 'popen', subprocess.Popen)
    if _last_nvml_output_time is None or _last_nvml_output_time < time.time() - 1:
        try:
            proc = popen(['nvidia-smi', '-q', '-x'], shell=False, stdout=subprocess.PIPE)
            proc.wait()

            _last_nvml_output = ElementTree.fromstring(proc.stdout.read())
            _last_nvml_output_time = time.time()
            proc.stdout.close()
        except (subprocess.CalledProcessError, OSError):
            pass
    return _last_nvml_output


_cuda_info = namedtuple('cuda_info', 'driver_version cuda_version products gpu_count')
_cuda_card_stat = namedtuple('cuda_card_stat', 'product_name gpu_usage temperature fb_mem_info')


def cuda_info(async_ctx=None):
    nvml_output = _get_nvml_output(async_ctx)
    if nvml_output is None:
        return None
    return _cuda_info(
        driver_version=nvml_output.find('driver_version').text,
        cuda_version=nvml_output.find('cuda_version').text,
        products=[e.find('product_name').text for e in nvml_output.findall('gpu')],
        gpu_count=int(nvml_output.find('attached_gpus').text),
    )


def cuda_card_stats(async_ctx=None):
    from .utils import parse_readable_size

    nvml_output = _get_nvml_output(async_ctx)
    if nvml_output is None:
        return None
    infos = []
    for gpu_element in nvml_output.findall('gpu'):
        fb_total_mem = parse_readable_size(gpu_element.find('fb_memory_usage/total').text)[0]
        fb_used_mem = parse_readable_size(gpu_element.find('fb_memory_usage/used').text)[0]
        fb_free_mem = parse_readable_size(gpu_element.find('fb_memory_usage/free').text)[0]

        infos.append(_cuda_card_stat(
            product_name=gpu_element.find('product_name').text,
            gpu_usage=parse_readable_size(gpu_element.find('utilization/gpu_util').text)[0],
            temperature=float(gpu_element.find('temperature/gpu_temp').text.split()[0]),
            fb_mem_info=_virt_memory_stat(
                total=fb_total_mem, used=fb_used_mem, free=fb_free_mem, available=fb_free_mem,
                percent=fb_used_mem / fb_total_mem,
            )
        ))
    return infos
