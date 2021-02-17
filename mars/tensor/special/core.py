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


class _EmptyStub:
    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, item):
        return getattr(self._obj, item, None)


try:
    import scipy.special
    spspecial = _EmptyStub(scipy.special)
except ImportError:  # pragma: no cover
    spspecial = _EmptyStub(None)

from ... import opcodes
from ...serialize import DictField, KeyField, ListField, StringField, ValueType
from ..arithmetic.core import TensorUnaryOp, TensorBinOp, TensorElementWiseWithInputs
from ..array_utils import np, cp, sparse, convert_order, as_same_device, device
from ..core import Tensor
from ..datasource import tensor as astensor
from ..operands import TensorOperand
from ..utils import check_order, broadcast_shape, filter_inputs, check_out_param


_func_name_to_special_cls = {}


class TensorSpecialOperandMixin:
    _op_code_ = opcodes.SPECIAL
    _func_name = None

    def __new__(cls, *args, **kwargs):
        if cls._func_name is not None:
            return object.__new__(_func_name_to_special_cls[cls._func_name])
        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def _on_op_register_(cls):
        if cls._func_name is not None:
            _func_name_to_special_cls[cls._func_name] = cls

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


class TensorSpecialMultiOp(TensorSpecialOperandMixin, TensorElementWiseWithInputs,
                           TensorOperand):
    _ARG_COUNT = None

    _args = ListField('args')
    _out = KeyField('out')
    _where = KeyField('where')
    _casting = StringField('casting')
    _order = StringField('order')
    _err = DictField('err', ValueType.string, ValueType.string)

    def __init__(self, args=None, out=None, where=None, dtype=None, casting=None,
                 order=None, err=None, **kwargs):
        args = list(args or [None] * self._ARG_COUNT)
        super().__init__(_args=args, _out=out, _where=where, _order=order,
                         _dtype=dtype, _casting=casting, _er=err, **kwargs)
        if self._casting is None:
            self._casting = 'same_kind'
        if self._order is None:
            self._order = 'K'
        check_order(self._order)

    @property
    def args(self):
        return getattr(self, '_args', [None] * self._ARG_COUNT)

    @property
    def out(self):
        return getattr(self, '_out', None)

    @property
    def order(self):
        return getattr(self, '_order', None)

    @property
    def casting(self):
        return getattr(self, '_casting', None)

    @property
    def err(self):
        return getattr(self, '_err', dict())

    @classmethod
    def _is_sparse(cls, *args):
        return False

    def _set_sparse(self, inputs):
        inputs_iter = iter(inputs)
        args = list(self._args)
        for idx in range(len(self._args)):
            if not np.isscalar(self._args[idx]):
                args[idx] = next(inputs_iter)
        setattr(self, '_sparse', self._is_sparse(*args))

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(inputs)

        args = list(self._args)
        for idx in range(len(args)):
            if not np.isscalar(args[idx]):
                args[idx] = next(inputs_iter)
        self._args = args

        if getattr(self, '_out', None) is not None:
            self._out = next(inputs_iter)
        if getattr(self, '_where', None) is not None:
            self._where = next(inputs_iter)

    def _process_inputs(self, *args, out=None):
        self._args = [a if np.isscalar(a) else astensor(a) for a in args]

        if out is not None:
            if isinstance(out, Tensor):
                self._out = out
            else:
                raise TypeError(f'out should be Tensor object, got {type(out)} instead')

        return args + (out,)

    def __call__(self, *args, out=None):
        proc_inputs_results = self._process_inputs(*args, out=out)
        args = proc_inputs_results[:-2]
        out, where = proc_inputs_results[-2:]
        # check broadcast
        shapes = [() if np.isscalar(a) else a.shape for a in self._args]
        shape = broadcast_shape(*shapes)
        order = out.order if out is not None else None

        inputs = filter_inputs(list(args) + [out, where])
        t = self.new_tensor(inputs, shape, order=order)

        if out is None:
            return t

        check_out_param(out, t, getattr(self, '_casting'))
        out_shape, out_dtype = out.shape, out.dtype

        # if `out` is specified, use out's dtype and shape
        if t.shape != out_shape:
            t = self.new_tensor(inputs, out_shape, order=order)
        setattr(self, '_dtype', out_dtype)

        out.data = t.data
        return out

    @classmethod
    def _execute_gpu(cls, op, xp, *args, **kw):
        if kw.get('out') is not None:
            kw['out'] = xp.asarray(kw['out'])
        r = cls._get_func(xp)(*args, **kw)
        return convert_order(r, op.outputs[0].order.value)

    @classmethod
    def _execute_cpu(cls, op, xp, *args, **kw):
        kw['order'] = op.order
        if kw.get('out') is not None:
            kw['out'] = np.asarray(kw['out'])
        return cls._get_func(xp)(*args, **kw)

    @classmethod
    def execute(cls, ctx, op):
        inputs, device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            kw = {'casting': op.casting} if op.out is not None else {}

            inputs_iter = iter(inputs)
            args = [a if np.isscalar(a) else next(inputs_iter) for a in op.args]
            if op.out is not None:
                kw['out'] = next(inputs_iter).copy()

            with np.errstate(**op.err):
                if op.is_gpu():
                    ret = cls._execute_gpu(op, xp, *args, **kw)
                else:
                    ret = cls._execute_cpu(op, xp, *args, **kw)

                if ret.dtype != op.dtype:
                    ret = ret.astype(op.dtype)
                ctx[op.outputs[0].key] = ret
