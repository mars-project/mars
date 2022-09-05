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

from typing import Any, Dict, List, Tuple

from ..utils import lazy_import
from .core import Serializer, buffered

ray = lazy_import("ray")


class RaySerializer(Serializer):
    """Return raw object to let ray do serialization."""

    @buffered
    def serial(self, obj: Any, context: Dict):
        return (obj,), [], True

    def deserial(self, serialized: Tuple, context: Dict, subs: List[Any]):
        assert not subs
        return serialized[0]


if ray:
    RaySerializer.register(object, "ray")
    RaySerializer.register(ray.ObjectRef, "ray")
    RaySerializer.register(ray.actor.ActorHandle, "ray")
