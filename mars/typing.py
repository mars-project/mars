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

try:
    from typing import Tuple, TypeVar
except ImportError:  # pragma: no cover
    # in some scenario (for instance, pycharm debug), `mars.typing`
    # could be mistakenly imported as builtin typing. Code below
    # resolves this issue.
    import os
    import sys

    _orig_sys_path = list(sys.path)
    _mars_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        sys.path = [p for p in sys.path if not p.startswith(_mars_path)]
        sys.modules.pop("typing", None)
        from typing import Tuple, TypeVar
    finally:
        sys.path = _orig_sys_path
        del _orig_sys_path, _mars_path

OperandType = TypeVar("OperandType")
TileableType = TypeVar("TileableType")
ChunkType = TypeVar("ChunkType")
EntityType = TypeVar("EntityType")
SessionType = TypeVar("SessionType")

ClusterType = TypeVar("ClusterType")
ClientType = TypeVar("ClientType")

BandType = Tuple[str, str]  # (band address, resource_type)
