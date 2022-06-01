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

cdef class Resource:
    cdef readonly:
        float num_cpus
        float num_gpus
        float mem_bytes

    def __init__(self, float num_cpus=0, float num_gpus=0, float mem_bytes=0):
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.mem_bytes = mem_bytes

    def __eq__(self, Resource other):
        cdef bint ret = (
            self.mem_bytes == other.mem_bytes
            and self.num_gpus == other.num_gpus
            and self.num_cpus == other.num_cpus
        )
        return ret

    cdef bint _le(self, Resource other) nogil:
        # memory first, then gpu, cpu last
        cdef bint ret = (
            self.mem_bytes <= other.mem_bytes
            and self.num_gpus <= other.num_gpus
            and self.num_cpus <= other.num_cpus
        )
        return ret

    def __gt__(self, Resource other):
        return not self._le(other)

    def __le__(self, Resource other):
        return self._le(other)

    def __add__(self, Resource other):
        return Resource(
            num_cpus=self.num_cpus + other.num_cpus,
            num_gpus=self.num_gpus + other.num_gpus,
            mem_bytes=self.mem_bytes + other.mem_bytes,
        )

    def __sub__(self, Resource other):
        return Resource(
            num_cpus=self.num_cpus - other.num_cpus,
            num_gpus=self.num_gpus - other.num_gpus,
            mem_bytes=self.mem_bytes - other.mem_bytes,
        )

    def __neg__(self):
        return Resource(
            num_cpus=-self.num_cpus,
            num_gpus=-self.num_gpus,
            mem_bytes=-self.mem_bytes,
        )

    def __repr__(self):
        return f"Resource(num_cpus={self.num_cpus}, num_gpus={self.num_gpus}, mem_bytes={self.mem_bytes})"

ZeroResource = Resource(num_cpus=0, num_gpus=0, mem_bytes=0)
