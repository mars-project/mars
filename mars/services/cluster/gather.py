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
import platform
import socket
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import scipy
except ImportError:  # pragma: no cover
    scipy = None

from ... import resource as mars_resource
from ...config import options
from ...utils import git_info, lazy_import
from ...storage import StorageLevel
from .core import WorkerSlotInfo, QuotaInfo, DiskInfo, StorageInfo

cp = lazy_import('cupy', globals=globals(), rename='cp')
cudf = lazy_import('cudf', globals=globals())

logger = logging.getLogger(__name__)

_is_initial = True


def gather_node_env():
    from ...lib.mkl_interface import mkl_get_version
    from ...lib.nvutils import NVError
    from ..._version import __version__ as mars_version

    global _is_initial
    if _is_initial:
        _is_initial = False
        mars_resource.cpu_percent()

    mem_stats = mars_resource.virtual_memory()

    node_info = {
        'command_line': sys.argv,
        'platform': platform.platform(),
        'host_name': socket.gethostname(),
        'python_version': sys.version,
        'mars_version': mars_version,
        'cpu_total': mars_resource.cpu_count(),
        'memory_total': mem_stats.total,
        'options': options.to_dict(),
    }

    if 'MARS_K8S_POD_NAME' in os.environ:
        node_info['k8s_pod_name'] = os.environ['MARS_K8S_POD_NAME']
    if 'CONTAINER_ID' in os.environ:
        node_info['yarn_container_id'] = os.environ['CONTAINER_ID']

    try:
        cuda_info = mars_resource.cuda_info()
    except NVError:  # pragma: no cover
        logger.exception('NVError encountered, cannot gather CUDA devices.')
        cuda_info = None

    if cuda_info:
        node_info['cuda_info'] = {
            'driver': cuda_info.driver_version,
            'cuda': cuda_info.cuda_version,
            'products': list(cuda_info.products),
        }

    package_vers = {
        'numpy': np.__version__,
        'pandas': pd.__version__,
    }
    if hasattr(np, '__mkl_version__') and mkl_get_version:
        mkl_version = mkl_get_version()
        package_vers['mkl'] = f'{mkl_version.major}.{mkl_version.minor}.{mkl_version.update}'

    if scipy is not None:
        package_vers['scipy'] = scipy.__version__
    if cp is not None:
        package_vers['cupy'] = cp.__version__
    if cudf is not None:
        package_vers['cudf'] = cudf.__version__

    node_info['package_versions'] = package_vers

    git = git_info()
    if git:
        node_info['git_info'] = {
            'hash': git.commit_hash,
            'ref': git.commit_ref,
        }

    bands = node_info['bands'] = dict()

    cpu_band = {
        'resources': {
            'cpu': mars_resource.cpu_count(),
            'memory': mars_resource.virtual_memory().total,
        }
    }
    # todo numa can be supported by adding more bands
    bands['numa-0'] = cpu_band

    for idx, gpu_card_stat in enumerate(mars_resource.cuda_card_stats()):  # pragma: no cover
        bands[f'gpu-{idx}'] = {
            'resources': {
                'gpu': 1,
                'memory': gpu_card_stat.fb_mem_info.total,
            }
        }
    return node_info


def gather_node_resource(band_to_slots=None, use_gpu=True):
    # todo numa can be supported by adding more bands
    res = dict()
    mem_info = mars_resource.virtual_memory()
    num_cpu = mars_resource.cpu_count() if band_to_slots is None \
        else band_to_slots.get('numa-0', None)
    if num_cpu:  # pragma: no branch
        res['numa-0'] = {
            'cpu_avail': mars_resource.cpu_count() - mars_resource.cpu_percent() / 100.0,
            'cpu_total': num_cpu,
            'memory_avail': mem_info.available,
            'memory_total': mem_info.total,
        }

    if use_gpu:
        for idx, gpu_card_stat in enumerate(mars_resource.cuda_card_stats()):  # pragma: no cover
            num_gpu = 1 if band_to_slots is None else band_to_slots.get(f'gpu-{idx}')
            if not num_gpu:
                continue
            res[f'gpu-{idx}'] = {
                'gpu_avail': 1 - gpu_card_stat.gpu_usage,
                'gpu_total': num_gpu,
                'memory_avail': gpu_card_stat.fb_mem_info.available,
                'memory_total': gpu_card_stat.fb_mem_info.total,
            }
    return res


def gather_node_details(
    band_slot_infos: Dict[str, List[WorkerSlotInfo]] = None,
    band_quota_infos: Dict[str, QuotaInfo] = None,
    disk_infos: List[DiskInfo] = None,
    band_storage_infos: Dict[str, Dict[StorageLevel, StorageInfo]] = None
):
    disk_io_usage = mars_resource.disk_io_usage()
    net_io_usage = mars_resource.net_io_usage()
    res = {
        'disk': dict(zip(('reads', 'writes'), disk_io_usage)) if disk_io_usage else dict(),
        'network': dict(zip(('receives', 'sends'), net_io_usage)) if net_io_usage else dict(),
        'iowait': mars_resource.iowait(),
    }

    if disk_infos:
        part_dict = dict()
        for info in disk_infos:
            part_dev = mars_resource.get_path_device(info.path)
            if part_dev in part_dict:
                continue

            disk_usage_result = mars_resource.disk_usage(info.path)
            io_usage_result = mars_resource.disk_io_usage(info.path)
            part_dict[part_dev] = disk_info = {
                'size_limit': info.limit_size,
                'size_used': disk_usage_result.used,
                'size_total': disk_usage_result.total,
            }
            if io_usage_result is not None:
                disk_info.update({
                    'reads': io_usage_result.reads if io_usage_result else None,
                    'writes': io_usage_result.writes if io_usage_result else None,
                })
            if not sys.platform.startswith('win'):
                in_usage_result = os.statvfs(info.path)
                disk_info.update({
                    'inode_used': in_usage_result.f_files - in_usage_result.f_favail,
                    'inode_total': in_usage_result.f_files,
                })
        res['disk']['partitions'] = part_dict

    band_slot_infos = band_slot_infos or dict()
    res['slot'] = {
        band: [{
            'slot_id': slot_info.slot_id,
            'session_id': slot_info.session_id,
            'subtask_id': slot_info.subtask_id,
            'processor_usage': slot_info.processor_usage,
        } for slot_info in slot_infos]
        for band, slot_infos in band_slot_infos.items()
    }

    band_quota_infos = band_quota_infos or dict()
    res['quota'] = {
        band: {
            'quota_size': quota_info.quota_size,
            'allocated_size': quota_info.allocated_size,
            'hold_size': quota_info.hold_size,
        } for band, quota_info in band_quota_infos.items()
    }

    band_storage_infos = band_storage_infos or dict()
    res['storage'] = {
        band: {
            level.name.lower(): {
                'size_used': storage_info.used_size,
                'size_total': storage_info.total_size,
                'size_pinned': storage_info.pinned_size,
            }
            for level, storage_info in storage_infos.items()
        }
        for band, storage_infos in band_storage_infos.items()
    }
    return res
