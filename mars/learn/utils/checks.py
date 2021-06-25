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

import numpy as np
try:
    from sklearn import get_config as get_sklearn_config
except ImportError:  # pragma: no cover
    get_sklearn_config = None

from ... import opcodes as OperandDef
from ... import tensor as mt
from ...core import ENTITY_TYPE, get_output_types, recursive_tile
from ...core.operand import OperandStage
from ...config import options
from ...serialization.serializables import KeyField, StringField, BoolField, DataTypeField
from ...tensor.core import TensorOrder, TENSOR_CHUNK_TYPE
from ...tensor.array_utils import as_same_device, device, issparse, get_array_module
from ...utils import ceildiv
from ..operands import LearnOperand, LearnOperandMixin, OutputType


class CheckBase(LearnOperand, LearnOperandMixin):
    _input = KeyField('input')
    _value = KeyField('value')
    _err_msg = StringField('err_msg')

    def __init__(self, input=None, value=None, err_msg=None, output_types=None, **kw):
        super().__init__(_input=input, _value=value, _err_msg=err_msg,
                         _output_types=output_types, **kw)

    @property
    def input(self):
        return self._input

    @property
    def value(self):
        return self._value

    @property
    def err_msg(self):
        return self._err_msg

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if self._input is not None:
            self._input = self._inputs[0]
        if self._value is not None:
            self._value = self._inputs[-1]

    def __call__(self, x, value=None):
        # output input if value not specified
        self._value = value = value if value is not None else x
        self.output_types = get_output_types(value)
        self.stage = OperandStage.agg
        return self.new_tileable([x, value],
                                 kws=[value.params])

    @classmethod
    def tile(cls, op):
        combine_size = options.combine_size
        x, value = op.input, op.value
        check_chunks = []
        for i, chunk in enumerate(x.chunks):
            chunk_op = cls(err_msg=op.err_msg, stage=OperandStage.map,
                           output_types=[OutputType.tensor])
            check_chunk = chunk_op.new_chunk([chunk], shape=(),
                                             index=(i,),
                                             dtype=np.dtype(bool),
                                             order=TensorOrder.C_ORDER)
            check_chunks.append(check_chunk)

        while len(check_chunks) > 1:
            prev_check_chunks = check_chunks
            check_chunks = []
            chunk_size = ceildiv(len(prev_check_chunks), combine_size)
            for i in range(chunk_size):
                chunks = prev_check_chunks[i * combine_size: (i + 1) * combine_size]
                chunk_op = cls(err_msg=op.err_msg, stage=OperandStage.combine,
                               output_types=[OutputType.tensor])
                check_chunk = chunk_op.new_chunk(chunks, shape=(),
                                                 index=(i,),
                                                 dtype=np.dtype(bool),
                                                 order=TensorOrder.C_ORDER)
                check_chunks.append(check_chunk)

        check_chunk = check_chunks[0]
        out_chunks = []
        for val_chunk in value.chunks:
            chunk_op = cls(value=val_chunk, err_msg=op.err_msg, stage=OperandStage.agg,
                           output_types=op.output_types)
            out_chunk = chunk_op.new_chunk([check_chunk, val_chunk], kws=[val_chunk.params])
            out_chunks.append(out_chunk)

        new_op = op.copy()
        kw = op.outputs[0].params
        kw['chunks'] = out_chunks
        kw['nsplits'] = value.nsplits
        return new_op.new_tileables(op.inputs, kws=[kw])


class CheckNonNegative(CheckBase):
    _op_type_ = OperandDef.CHECK_NON_NEGATIVE

    _whom = StringField('whom')

    def __init__(self, input=None, value=None, whom=None, err_msg=None,
                 stage=None, gpu=None, output_types=None, **kw):
        super().__init__(input=input, value=value, _whom=whom,
                         err_msg=err_msg, stage=stage,
                         output_types=output_types,
                         gpu=gpu, **kw)
        if self._err_msg is None and self._whom is not None:
            self._err_msg = f"Negative values in data passed to {self._whom}"

    @property
    def whom(self):
        return self._whom

    @classmethod
    def _execute_tensor(cls, ctx, op):
        (x,), device_id, xp = as_same_device(
            [ctx[inp.key] for inp in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            if issparse(x) and x.nnz == 0:
                x_min = 0
            else:
                x_min = xp.min(x)

            if x_min < 0:
                raise ValueError(op.err_msg)

            ctx[op.outputs[0].key] = np.array(True)

    @classmethod
    def _execute_df(cls, ctx, op):
        x = ctx[op.inputs[0].key]
        x_min = x.min().min()
        if x_min < 0:
            raise ValueError(op.err_msg)

        ctx[op.outputs[0].key] = np.array(True)

    @classmethod
    def _execute_map(cls, ctx, op):
        if isinstance(op.inputs[0], TENSOR_CHUNK_TYPE):
            return cls._execute_tensor(ctx, op)
        else:
            return cls._execute_df(ctx, op)

    @classmethod
    def _execute_combine(cls, ctx, op):
        # just pass value cuz all inputs executed successfully
        ctx[op.outputs[0].key] = np.array(True)

    @classmethod
    def _execute_agg(cls, ctx, op):
        ctx[op.outputs[0].key] = ctx[op.value.key]

    @classmethod
    def execute(cls, ctx, op):
        if op.stage == OperandStage.map:
            return cls._execute_map(ctx, op)
        elif op.stage == OperandStage.combine:
            return cls._execute_combine(ctx, op)
        else:
            assert op.stage == OperandStage.agg
            return cls._execute_agg(ctx, op)


def check_non_negative_then_return_value(to_check, value, whom):
    op = CheckNonNegative(input=to_check, value=value, whom=whom)
    return op(to_check, value)


class AssertAllFinite(LearnOperand, LearnOperandMixin):
    _op_type_ = OperandDef.ASSERT_ALL_FINITE

    _x = KeyField('x')
    _allow_nan = BoolField('allow_nan')
    _msg_dtype = DataTypeField('msg_dtype')
    _check_only = BoolField('check_only')
    # chunks
    _is_finite = KeyField('is_finite')
    _check_nan = KeyField('check_nan')

    def __init__(self, x=None, allow_nan=None, msg_dtype=None,
                 check_only=None, is_finite=None, check_nan=None,
                 output_types=None, **kw):
        super().__init__(_x=x, _allow_nan=allow_nan, _msg_dtype=msg_dtype,
                         _check_only=check_only, _is_finite=is_finite,
                         _check_nan=check_nan, _output_types=output_types, **kw)

    @property
    def x(self):
        return self._x

    @property
    def allow_nan(self):
        return self._allow_nan

    @property
    def msg_dtype(self):
        return self._msg_dtype

    @property
    def check_only(self):
        return self._check_only

    @property
    def is_finite(self):
        return self._is_finite

    @property
    def check_nan(self):
        return self._check_nan

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs)
        for attr in ('_x', '_is_finite', '_check_nan'):
            if getattr(self, attr) is not None:
                setattr(self, attr, next(inputs_iter))

    @classmethod
    def _assume_finite(cls):
        assume_finite = options.learn.assume_finite
        if assume_finite is None and get_sklearn_config is not None:
            # get config from scikit-learn
            assume_finite = get_sklearn_config()['assume_finite']
        if assume_finite is None:  # pragma: no cover
            assume_finite = False

        return assume_finite

    def __call__(self, x):
        if self._assume_finite():
            # skip check
            if self._check_only:
                return
            else:
                return x

        if self._check_only:
            return self.new_tileable([x], dtype=np.dtype(bool),
                                     shape=(), order=TensorOrder.C_ORDER)
        else:
            return self.new_tileable([x], kws=[x.params])

    @classmethod
    def tile(cls, op):
        from .extmath import _safe_accumulator_op

        x = op.x
        out = op.outputs[0]
        is_float = x.dtype.kind in 'fc'
        combine_size = options.combine_size

        is_finite_chunk = check_nan_chunk = None
        if is_float:
            is_finite_chunk = (yield from recursive_tile(
                mt.isfinite(_safe_accumulator_op(mt.sum, x)))).chunks[0]
        elif x.dtype == np.dtype(object) and not op.allow_nan:
            check_nan_chunk = (yield from recursive_tile(
                (x != x).any())).chunks[0]

        map_chunks = []
        for c in x.chunks:
            chunk_op = op.copy().reset_key()
            chunk_op.stage = OperandStage.map
            chunk_op._is_finite = is_finite_chunk
            chunk_op._check_nan = check_nan_chunk
            chunk_inputs = [c]
            if is_finite_chunk is not None:
                chunk_inputs.append(is_finite_chunk)
            if check_nan_chunk is not None:
                chunk_inputs.append(check_nan_chunk)
            chunk_params = c.params
            if op.check_only:
                chunk_params['dtype'] = np.dtype(bool)
                chunk_params['shape'] = ()
                if len(x.chunks) == 1:
                    chunk_params['index'] = ()
            map_chunk = chunk_op.new_chunk(chunk_inputs, kws=[chunk_params])
            map_chunks.append(map_chunk)

        new_op = op.copy()
        if not op.check_only:
            params = out.params
            params['nsplits'] = x.nsplits
            params['chunks'] = map_chunks
            return new_op.new_tileables(op.inputs, kws=[params])

        out_chunks = map_chunks
        # if check only, we use tree reduction to aggregate to one chunk
        while len(out_chunks) > 1:
            size = ceildiv(len(out_chunks), combine_size)
            new_out_chunks = []
            for i in range(size):
                chunk_op = AssertAllFinite(
                    check_only=True, output_types=op.output_types,
                    stage=OperandStage.combine if size > 1 else OperandStage.agg)
                chunk_index = (i,) if size > 1 else ()
                out_chunk = chunk_op.new_chunk(
                    out_chunks[combine_size * i: combine_size * (i + 1)],
                    dtype=out.dtype, shape=(), index=chunk_index, order=out.order)
                new_out_chunks.append(out_chunk)
            out_chunks = new_out_chunks

        params = out.params
        params['nsplits'] = ()
        params['chunks'] = out_chunks
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    def _execute_map(cls, ctx, op):
        allow_nan = op.allow_nan
        msg_dtype = op.msg_dtype
        raw = x = ctx[op.x.key]
        xp = get_array_module(x, nosparse=True)

        if issparse(x):
            x = x.data
        # First try an O(n) time, O(1) space solution for the common case that
        # everything is finite; fall back to O(n) space np.isfinite to prevent
        # false positives from overflow in sum method. The sum is also calculated
        # safely to reduce dtype induced overflows.
        is_float = x.dtype.kind in 'fc'
        if is_float and ctx[op.is_finite.key]:
            pass
        elif is_float:
            msg_err = "Input contains {} or a value too large for {!r}."
            if (allow_nan and xp.isinf(x).any() or
                    not allow_nan and not xp.isfinite(x).all()):
                type_err = 'infinity' if allow_nan else 'NaN, infinity'
                raise ValueError(
                    msg_err.format
                    (type_err,
                     msg_dtype if msg_dtype is not None else x.dtype)
                )
        # for object dtype data, we only check for NaNs
        elif x.dtype == np.dtype(object) and not allow_nan:
            if ctx[op.check_nan.key]:
                raise ValueError("Input contains NaN")

        if op.check_only:
            result = np.array(True)
        else:
            result = raw
        ctx[op.outputs[0].key] = result

    @classmethod
    def _execute_combine_reduce(cls, ctx, op):
        # just return True
        ctx[op.outputs[0].key] = np.array(True)

    @classmethod
    def execute(cls, ctx, op):
        if op.stage == OperandStage.map:
            return cls._execute_map(ctx, op)
        else:
            assert op.stage in (OperandStage.combine, OperandStage.agg)
            return cls._execute_combine_reduce(ctx, op)


def assert_all_finite(X, allow_nan=False, msg_dtype=None, check_only=True):
    if not isinstance(X, ENTITY_TYPE):
        X = mt.asarray(X)

    if isinstance(X.op, AssertAllFinite) and X.op.allow_nan == allow_nan and \
            X.op.msg_dtype == msg_dtype and X.op.check_only == check_only:
        return X

    if check_only:
        output_types = [OutputType.tensor]
        sparse = False
    else:
        output_types = get_output_types(X)
        sparse = X.issparse()

    op = AssertAllFinite(x=X, allow_nan=allow_nan, msg_dtype=msg_dtype,
                         check_only=check_only, sparse=sparse,
                         output_types=output_types)
    return op(X)
