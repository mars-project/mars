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

from itertools import count
from typing import Dict, List

from ... import oscar as mo

try:
    from IPython import get_ipython
except ImportError:
    get_ipython = None


async def create_supervisor_actor_pool(
        address: str,
        n_process: int,
        modules: List[str] = None,
        ports: List[int] = None,
        subprocess_start_method: str = None,
        **kwargs):
    suspend_sigint = get_ipython is not None and get_ipython() is not None
    return await mo.create_actor_pool(
        address, n_process=n_process, ports=ports, modules=modules,
        subprocess_start_method=subprocess_start_method,
        suspend_sigint=suspend_sigint,
        **kwargs)


async def create_worker_actor_pool(
        address: str,
        band_to_slots: Dict[str, int],
        n_io_process: int = 1,
        modules: List[str] = None,
        ports: List[int] = None,
        subprocess_start_method: str = None,
        **kwargs):
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
            envs.extend([dict() for _ in range(slot)])
            labels.extend([band] * slot)

    suspend_sigint = get_ipython is not None and get_ipython() is not None
    return await mo.create_actor_pool(
        address, n_process=n_process, ports=ports,
        n_io_process=n_io_process,
        labels=labels, envs=envs, modules=modules,
        subprocess_start_method=subprocess_start_method,
        suspend_sigint=suspend_sigint,
        **kwargs)
