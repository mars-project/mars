# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

from .partition import _validate_partition_arguments, TensorPartition


def argpartition(a, kth, axis=-1, kind='introselect', order=None, **kw):
    a, kth, axis, kind, order, need_align = _validate_partition_arguments(
        a, kth, axis, kind, order, kw)
    op = TensorPartition(kth=kth, axis=axis, kind=kind, order=order,
                         need_align=need_align, return_value=False,
                         return_indices=True, dtype=a.dtype, gpu=a.op.gpu)
    return op(a, kth)
