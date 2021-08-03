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

# import aio to ensure patch enabled for Python 3.6
from ..lib import aio
del aio

from . import debug
from .api import actor_ref, create_actor, has_actor, destroy_actor, \
    kill_actor, Actor, StatelessActor, create_actor_pool, setup_cluster, \
    wait_actor_pool_recovered
from .backends import allocate_strategy
from .backends.pool import MainActorPoolType
from .batch import extensible
from .core import ActorRef
from .debug import set_debug_options, DebugOptions
from .errors import ActorNotExist, ActorAlreadyExist, ServerClosed, Return
from .utils import create_actor_ref

# make sure methods are registered
from .backends import mars, ray, test
del mars, ray, test
