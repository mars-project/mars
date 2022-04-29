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


from typing import List, Dict
from ....resource import Resource


def get_band_resources_from_config(execution_config: Dict) -> List[Dict[str, Resource]]:
    backend = execution_config["backend"]
    config = execution_config[backend]
    n_worker: int = config["n_worker"]
    n_cpu: int = config["n_cpu"]
    mem_bytes: int = config["mem_bytes"]
    cuda_devices: List[List[int]] = config["cuda_devices"]

    bands_to_resource = []
    worker_cpus = n_cpu // n_worker
    cuda_devices = cuda_devices or ([[]] * n_worker)
    if sum(len(devices) for devices in cuda_devices) == 0:
        assert worker_cpus > 0, (
            f"{n_cpu} cpus are not enough " f"for {n_worker}, try to decrease workers."
        )
    mem_bytes = mem_bytes // n_worker
    for _, devices in zip(range(n_worker), cuda_devices):
        worker_band_to_resource = dict()
        worker_band_to_resource["numa-0"] = Resource(
            num_cpus=worker_cpus, mem_bytes=mem_bytes
        )
        for i in devices:  # pragma: no cover
            worker_band_to_resource[f"gpu-{i}"] = Resource(num_gpus=1)
        bands_to_resource.append(worker_band_to_resource)
    return bands_to_resource
