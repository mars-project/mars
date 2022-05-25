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

import asyncio
import contextlib
import sys

from .file import AioFileObject, AioFilesystem
from .isolation import Isolation, new_isolation, get_isolation, stop_isolation
from .lru import alru_cache
from .parallelism import AioEvent


if sys.version_info[:2] < (3, 9):
    from ._threads import to_thread

    asyncio.to_thread = to_thread
