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

from .sort import _validate_sort_arguments, TensorSort


def argsort(a, axis=-1, kind=None, parallel_kind=None, psrs_kinds=None, order=None):
    a, axis, kind, parallel_kind, psrs_kinds, order = _validate_sort_arguments(
        a, axis, kind, parallel_kind, psrs_kinds, order)

    op = TensorSort(axis=axis ,kind=kind, parallel_kind=parallel_kind,
                    order=order, psrs_kinds=psrs_kinds,
                    return_value=False, return_indices=True,
                    dtype=a.dtype, gpu=a.op.gpu)
    return op(a)
