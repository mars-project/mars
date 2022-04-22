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

from concurrent.futures import Executor
from typing import Any, Callable, Dict, List, Tuple

def buffered(func: Callable) -> Callable: ...
def fast_id(obj: Any) -> int: ...

class Serializer:
    serializer_id: int
    def serial(self, obj: Any, context: Dict): ...
    def deserial(self, serialized: Tuple, context: Dict, subs: List[Any]): ...
    def on_deserial_error(
        self,
        serialized: Tuple,
        context: Dict,
        subs_serialized: List,
        error_index: int,
        exc: BaseException,
    ): ...
    @classmethod
    def register(cls, obj_type): ...
    @classmethod
    def unregister(cls, obj_type): ...

class Placeholder:
    id: int
    callbacks: List[Callable]
    def __init__(self, id_: int): ...
    def __hash__(self): ...
    def __eq__(self, other): ...

def serialize(obj: Any, context: Dict = None): ...
async def serialize_with_spawn(
    obj: Any,
    context: Dict = None,
    spawn_threshold: int = 100,
    executor: Executor = None,
): ...
def deserialize(headers: List, buffers: List, context: Dict = None): ...
