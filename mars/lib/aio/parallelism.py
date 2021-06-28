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
import threading
import multiprocessing
from concurrent.futures import Executor
from typing import Union

from .base import AioBase, delegate_to_executor, proxy_method_directly

event_types = Union[threading.Event, multiprocessing.Event]


@delegate_to_executor(
    "wait"
)
@proxy_method_directly(
    "set",
    "is_set",
    "clear"
)
class AioEvent(AioBase):
    def __init__(self,
                 event: event_types = None,
                 loop: asyncio.BaseEventLoop = None,
                 executor: Executor = None):
        if event is None:
            event = threading.Event()
        super().__init__(event, loop=loop,
                         executor=executor)
