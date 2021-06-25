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
    from pyarrow.fs import HadoopFileSystem as _ArrowHadoopFileSystem
    from .arrow import HadoopFileSystem
    del _ArrowHadoopFileSystem
except ImportError:  # pragma: no cover
    try:
        # pyarrow < 2.0.0
        from pyarrow import HadoopFileSystem
    except ImportError:
        HadoopFileSystem = None

from .core import register_filesystem


if HadoopFileSystem is not None:  # pragma: no branch
    register_filesystem('hdfs', HadoopFileSystem)
