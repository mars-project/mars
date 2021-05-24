# Copyright 1999-2020 Alibaba Group Holding Ltd.
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
import sys

from ... import resource as mars_resource
# todo move `node_info` module here when actors are done
from ...node_info import gather_node_env as _ext_gather_node_env


def gather_node_env():
    info = _ext_gather_node_env()
    bands = info['bands'] = dict()

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
    return info


def gather_node_resource(band_to_slots=None, use_gpu=True):
    # todo numa can be supported by adding more bands
    res = dict()
    band_to_slots = band_to_slots or dict()
    mem_info = mars_resource.virtual_memory()
    res['numa-0'] = {
        'cpu_avail': mars_resource.cpu_count() - mars_resource.cpu_percent() / 100.0,
        'cpu_total': band_to_slots.get('numa-0', mars_resource.cpu_count()),
        'memory_avail': mem_info.available,
        'memory_total': mem_info.total,
    }
    if use_gpu:
        for idx, gpu_card_stat in enumerate(mars_resource.cuda_card_stats()):  # pragma: no cover
            res[f'gpu-{idx}'] = {
                'gpu_avail': 1 - gpu_card_stat.gpu_usage,
                'gpu_total': band_to_slots.get(f'gpu-{idx}', 1),
                'memory_avail': gpu_card_stat.fb_mem_info.available,
                'memory_total': gpu_card_stat.fb_mem_info.total,
            }
    return res


def gather_node_states(dirs=None):
    disk_io_usage = mars_resource.disk_io_usage()
    net_io_usage = mars_resource.net_io_usage()
    res = {
        'disk': dict(zip(('reads', 'writes'), disk_io_usage)) if disk_io_usage else dict(),
        'network': dict(zip(('receives', 'sends'), net_io_usage)) if net_io_usage else dict(),
        'iowait': mars_resource.iowait(),
    }
    if dirs:
        part_dict = dict()
        for d in dirs:
            part_dev = mars_resource.get_path_device(d)
            if part_dev in part_dict:
                continue

            disk_usage_result = mars_resource.disk_usage(d)
            io_usage_result = mars_resource.disk_io_usage(d)
            part_dict[part_dev] = disk_info = {
                'size_used': disk_usage_result.used,
                'size_total': disk_usage_result.total,
            }
            if io_usage_result is not None:
                disk_info.update({
                    'reads': io_usage_result.reads if io_usage_result else None,
                    'writes': io_usage_result.writes if io_usage_result else None,
                })
            if not sys.platform.startswith('win'):
                in_usage_result = os.statvfs(d)
                disk_info.update({
                    'inode_used': in_usage_result.f_files - in_usage_result.f_favail,
                    'inode_total': in_usage_result.f_files,
                })
        res['disk']['partitions'] = part_dict
    return res
