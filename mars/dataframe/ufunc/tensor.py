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

from ...utils import classproperty


_tensor_op_to_df_op = dict()


def register_tensor_ufunc(op):
    _tensor_op_to_df_op[op.tensor_op_type] = op


def get_tensor_ufunc_implementation(tensor_op):
    if tensor_op in _tensor_op_to_df_op:
        return _tensor_op_to_df_op[tensor_op]


class TensorUfuncMixin:
    __slots__ = ()

    @classproperty
    def tensor_op_type(self):
        raise NotImplementedError

    @classmethod
    def ufunc_call(cls, tensor_op, inputs, out, where, **kw):
        if out is not None:
            return NotImplemented
        if where is not None:
            raise NotImplementedError

        try:
            op = _tensor_op_to_df_op[tensor_op](**kw)
            return op(*inputs)
        except (KeyError, TypeError):
            return NotImplemented


def _tensor_ufunc(_, tensor_op, inputs, out, where, **kw):
    op = get_tensor_ufunc_implementation(tensor_op)
    if op is not None:
        return op.ufunc_call(tensor_op, inputs, out, where, **kw)
    return NotImplemented
