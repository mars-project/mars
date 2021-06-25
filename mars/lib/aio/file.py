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

import functools

from .base import AioBase, delegate_to_executor, \
    proxy_method_directly, proxy_property_directly


@delegate_to_executor(
    "close",
    "flush",
    "isatty",
    "read",
    "read1",
    "readinto",
    "readline",
    "readlines",
    "seek",
    "seekable",
    "tell",
    "truncate",
    "writable",
    "write",
    "writelines",
)
@proxy_method_directly("fileno", "readable")
@proxy_property_directly("closed", "name", "mode")
class AioFileObject(AioBase):
    def __aiter__(self):
        return self

    async def __anext__(self):
        """Simulate normal file iteration."""
        line = await self.readline()
        if line:
            return line
        else:
            raise StopAsyncIteration

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        self._file = None


@delegate_to_executor(
    "cat",
    "ls",
    "delete",
    "disk_usage",
    "stat",
    "rm",
    "mv",
    "rename",
    "mkdir",
    "exists",
    "isdir",
    "isfile",
    "read_parquet",
    "walk",
)
@proxy_property_directly("pathsep")
class AioFilesystem(AioBase):
    async def open(self, *args, **kwargs):
        func = functools.partial(self._file.open, *args, **kwargs)
        file = await self._loop.run_in_executor(self._executor, func)
        return AioFileObject(file)
