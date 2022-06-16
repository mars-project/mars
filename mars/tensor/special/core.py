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

import scipy.special as spspecial

from ...core import ExecutableTuple
from ... import opcodes
from ..datasource import tensor as astensor
from ..arithmetic.core import TensorUnaryOp, TensorBinOp, TensorMultiOp
from ..array_utils import (
    np,
    cp,
    issparse,
    sparse,
    convert_order,
    as_same_device,
    device,
)


_func_name_to_special_cls = {}


def _register_special_op(cls):
    if cls._func_name is not None:
        _func_name_to_special_cls[cls._func_name] = cls
    return cls


class TensorSpecialOperandMixin:
    _op_code_ = opcodes.SPECIAL
    _func_name = None

    def __new__(cls, *args, **kwargs):
        if cls._func_name is not None:
            return object.__new__(_func_name_to_special_cls[cls._func_name])
        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def _get_func(cls, xp):
        if xp is np:
            from scipy import special

            return getattr(special, cls._func_name)
        elif cp is not None and xp is cp:
            from cupyx.scipy import special

            return getattr(special, cls._func_name)
        else:
            assert xp is sparse
            return getattr(sparse, cls._func_name)


class TensorSpecialUnaryOp(TensorSpecialOperandMixin, TensorUnaryOp):
    pass


class TensorSpecialBinOp(TensorSpecialOperandMixin, TensorBinOp):
    pass


class TensorSpecialMultiOp(TensorSpecialOperandMixin, TensorMultiOp):
    @classmethod
    def _execute_gpu(cls, op, xp, *args, **kw):
        if kw.get("out") is not None:
            kw["out"] = xp.asarray(kw["out"])
        r = cls._get_func(xp)(*args, **kw)
        return convert_order(r, op.outputs[0].order.value)

    @classmethod
    def _execute_cpu(cls, op, xp, *args, **kw):
        kw["order"] = op.order
        if kw.get("out") is not None:
            kw["out"] = np.asarray(kw["out"])
        try:
            return cls._get_func(xp)(*args, **kw)
        except TypeError:
            kw.pop("order")
            r = cls._get_func(xp)(*args, **kw)
            if issparse(r):
                return r
            return convert_order(r, op.outputs[0].order.value)

    @classmethod
    def execute(cls, ctx, op):
        inputs, device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True
        )

        with device(device_id):
            kw = {"casting": op.casting} if op.out is not None else {}

            inputs_iter = iter(inputs)
            args = [a if np.isscalar(a) else next(inputs_iter) for a in op.args]
            if op.out is not None:
                kw["out"] = next(inputs_iter).copy()

            with np.errstate(**op.err):
                if op.is_gpu():
                    ret = cls._execute_gpu(op, xp, *args, **kw)
                else:
                    ret = cls._execute_cpu(op, xp, *args, **kw)

                if ret.dtype != op.dtype:
                    ret = ret.astype(op.dtype)
                ctx[op.outputs[0].key] = ret


class TensorTupleOp(TensorSpecialUnaryOp):
    @property
    def output_limit(self):
        return self._n_outputs

    def __call__(self, x, out=None):
        x = astensor(x)

        if out is not None:
            if not isinstance(out, ExecutableTuple):
                raise TypeError(
                    f"out should be ExecutableTuple object, got {type(out)} instead"
                )
            if len(out) != self._n_outputs:
                raise TypeError(
                    f"out should be an ExecutableTuple object with {self._n_outputs} elements, got {len(out)} instead"
                )

        func = getattr(spspecial, self._func_name)
        res = func(np.ones(x.shape, dtype=x.dtype))
        res_tensors = self.new_tensors(
            [x],
            kws=[
                {
                    "side": f"{self._func_name}[{i}]",
                    "dtype": output.dtype,
                    "shape": output.shape,
                }
                for i, output in enumerate(res)
            ],
        )

        if out is None:
            return ExecutableTuple(res_tensors)

        for res_tensor, out_tensor in zip(res_tensors, out):
            out_tensor.data = res_tensor.data
        return out

    @classmethod
    def execute(cls, ctx, op):
        inputs, device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True
        )

        with device(device_id):
            with np.errstate(**op.err):
                if op.is_gpu():
                    ret = cls._execute_gpu(op, xp, inputs[0])
                else:
                    ret = cls._execute_cpu(op, xp, inputs[0])

                for output, ret_element in zip(op.outputs, ret):
                    ctx[output.key] = ret_element
