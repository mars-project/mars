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


from ... import opcodes
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
_tuple_func_to_res = {}


def _register_special_op(cls):
    if cls._func_name is not None:
        if getattr(cls, "_output_index", None) is None:
            _func_name_to_special_cls[cls._func_name] = cls
        else:
            if cls._func_name not in _func_name_to_special_cls:
                _func_name_to_special_cls[cls._func_name] = [
                    None for _ in range(cls._func_outputs)
                ]
            _func_name_to_special_cls[cls._func_name][cls._output_index] = cls
    return cls


class TensorSpecialOperandMixin:
    _op_code_ = opcodes.SPECIAL
    _func_name = None

    def __new__(cls, *args, **kwargs):
        if cls._func_name is not None:
            if getattr(cls, "_output_index", None) is None:
                return object.__new__(_func_name_to_special_cls[cls._func_name])
            else:
                return object.__new__(
                    _func_name_to_special_cls[cls._func_name][cls._output_index]
                )

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


class TensorTupleElementOp(TensorSpecialUnaryOp):
    @classmethod
    def execute(cls, ctx, op):
        input_keys = [c.key for c in op.inputs]
        inputs, device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True
        )

        with device(device_id):
            kw = {"casting": op.casting} if op.out else {}

            if op.out and op.where:
                input_keys = input_keys[:-2]
                inputs, kw["out"], kw["where"] = (
                    inputs[:-2],
                    inputs[-2].copy(),
                    inputs[-1],
                )
            elif op.out:
                input_keys = input_keys[:-1]
                inputs, kw["out"] = inputs[:-1], inputs[-1].copy()
            elif op.where:
                input_keys = input_keys[:-1]
                inputs, kw["where"] = inputs[:-1], inputs[-1]

            with np.errstate(**op.err):
                is_func_ret_cached = False
                if (
                    cls._func_name in _tuple_func_to_res
                    and input_keys[0] in _tuple_func_to_res[cls._func_name]
                ):
                    is_func_ret_cached = True
                    ret = _tuple_func_to_res[cls._func_name][input_keys[0]]
                elif op.is_gpu():
                    ret = cls._execute_gpu(op, xp, inputs[0], **kw)
                else:
                    ret = cls._execute_cpu(op, xp, inputs[0], **kw)

                # during a single execution, if we use multiple components of the same
                # scipy calculation, we can store the overall result after the first computation.
                if not is_func_ret_cached:
                    if cls._func_name not in _tuple_func_to_res:
                        _tuple_func_to_res[cls._func_name] = {}
                    if input_keys[0] not in _tuple_func_to_res[cls._func_name]:
                        _tuple_func_to_res[cls._func_name][input_keys[0]] = ret

                ret = ret[cls._output_index]
                if ret.dtype != op.dtype:
                    ret = ret.astype(op.dtype)
                ctx[op.outputs[0].key] = ret
