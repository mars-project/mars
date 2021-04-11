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

from itertools import count
from typing import Dict

from ... import oscar as mo


async def create_supervisor_actor_pool(
        address: str,
        n_process: int,
        subprocess_start_method: str = None):
    return await mo.create_actor_pool(
        address, n_process=n_process,
        subprocess_start_method=subprocess_start_method)


async def create_worker_actor_pool(
        address: str,
        band_to_slots: Dict[str, int],
        subprocess_start_method: str = None):
    # TODO: support NUMA when ready
    n_process = sum(slot for slot in band_to_slots.values())
    envs = []
    labels = ['main']
    i_gpu = count()
    for band, slot in band_to_slots.items():
        if band.startswith('gpu'):  # pragma: no cover
            envs.append({'CUDA_VISIBLE_DEVICES': str(next(i_gpu))})
            labels.append(band)
        else:
            assert band.startswith('numa')
            labels.extend([band] * slot)

    return await mo.create_actor_pool(
        address, n_process=n_process,
        labels=labels, envs=envs,
        subprocess_start_method=subprocess_start_method)
