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

from typing import List, Dict, Any

from ..utils import lazy_import
from .core import Serializer, buffered, PickleSerializer
from .exception import ExceptionSerializer

ray = lazy_import('ray')


class RaySerializer(Serializer):
    """Return raw object to let ray do serialization."""
    serializer_name = 'ray'

    @buffered
    def serialize(self, obj: Any, context: Dict):
        header = {'o': obj}
        buffers = []
        return header, buffers

    def deserialize(self, header: Dict, buffers: List, context: Dict):
        assert not buffers
        return header['o']


def register_ray_serializers():
    PickleSerializer.unregister(object)
    ExceptionSerializer.unregister(Exception)
    RaySerializer.register(object)
    RaySerializer.register(ray.ObjectRef)
    RaySerializer.register(ray.actor.ActorHandle)


def unregister_ray_serializers():
    RaySerializer.unregister(ray.actor.ActorHandle)
    RaySerializer.unregister(ray.ObjectRef)
    RaySerializer.unregister(object)
    PickleSerializer.register(object)
    ExceptionSerializer.register(Exception)
