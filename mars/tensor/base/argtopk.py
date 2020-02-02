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

from ...operands import OperandStage
from .topk import _validate_topk_arguments, TensorTopk


def argtopk(a, k, axis=-1, largest=True, sorted=True, order=None,
            parallel_kind='auto', psrs_kinds=None):
    a, k, axis, largest, sorted, order, parallel_kind, psrs_kinds = \
        _validate_topk_arguments(a, k, axis, largest, sorted, order,
                                 parallel_kind, psrs_kinds)
    op = TensorTopk(k=k, axis=axis, largest=largest, sorted=sorted,
                    parallel_kind=parallel_kind, psrs_kinds=psrs_kinds,
                    dtype=a.dtype, return_value=False, return_indices=True,
                    stage=OperandStage.agg)
    return op(a)
