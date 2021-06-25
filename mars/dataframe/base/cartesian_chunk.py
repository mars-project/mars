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
import pandas as pd

from ... import opcodes
from ...core import recursive_tile
from ...core.custom_log import redirect_custom_log
from ...serialization.serializables import KeyField, FunctionField, TupleField, DictField
from ...utils import has_unknown_shape, enter_current_session, quiet_stdio
from ..operands import DataFrameOperand, DataFrameOperandMixin, OutputType
from ..utils import build_df, build_series, build_empty_df, parse_index, \
    validate_output_types


class DataFrameCartesianChunk(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.CARTESIAN_CHUNK

    _left = KeyField('left')
    _right = KeyField('right')
    _func = FunctionField('func')
    _args = TupleField('args')
    _kwargs = DictField('kwargs')

    def __init__(self, left=None, right=None, func=None, args=None, kwargs=None,
                 output_types=None, **kw):
        super().__init__(_left=left, _right=right, _func=func, _args=args,
                         _kwargs=kwargs, _output_types=output_types, **kw)
        if self.memory_scale is None:
            self.memory_scale = 2.0

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def func(self):
        return self._func

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._left = self._inputs[0]
        self._right = self._inputs[1]

    @staticmethod
    def _build_test_obj(obj):
        return build_df(obj, size=2) if obj.ndim == 2 else \
            build_series(obj, size=2, name=obj.name)

    def __call__(self, left, right, index=None, dtypes=None):
        test_left = self._build_test_obj(left)
        test_right = self._build_test_obj(right)
        output_type = self._output_types[0] if self._output_types else None

        # try run to infer meta
        try:
            with np.errstate(all='ignore'), quiet_stdio():
                obj = self._func(test_left, test_right, *self._args, **self._kwargs)
        except:  # noqa: E722  # nosec  # pylint: disable=bare-except
            if output_type == OutputType.series:
                obj = pd.Series([], dtype=np.dtype(object))
            elif output_type == OutputType.dataframe and dtypes is not None:
                obj = build_empty_df(dtypes)
            else:
                raise TypeError('Cannot determine `output_type`, '
                                'you have to specify it as `dataframe` or `series`, '
                                'for dataframe, `dtypes` is required as well '
                                'if output_type=\'dataframe\'')

        if getattr(obj, 'ndim', 0) == 1 or output_type == OutputType.series:
            shape = self._kwargs.pop('shape', (np.nan,))
            if index is None:
                index = obj.index
            index_value = parse_index(index, left, right,
                                      self._func, self._args, self._kwargs)
            return self.new_series([left, right], dtype=obj.dtype,
                                   shape=shape, index_value=index_value,
                                   name=obj.name)
        else:
            dtypes = dtypes if dtypes is not None else obj.dtypes
            # dataframe
            shape = (np.nan, len(dtypes))
            columns_value = parse_index(dtypes.index, store_data=True)
            if index is None:
                index = obj.index
            index_value = parse_index(index, left, right,
                                      self._func, self._args, self._kwargs)
            return self.new_dataframe([left, right], shape=shape,
                                      dtypes=dtypes, index_value=index_value,
                                      columns_value=columns_value)

    @classmethod
    def tile(cls, op: "DataFrameCartesianChunk"):
        left = op.left
        right = op.right
        out = op.outputs[0]

        if left.ndim == 2 and left.chunk_shape[1] > 1:
            if has_unknown_shape(left):
                yield
            # if left is a DataFrame, make sure 1 chunk on axis columns
            left = yield from recursive_tile(
                left.rechunk({1: left.shape[1]}))
        if right.ndim == 2 and right.chunk_shape[1] > 1:
            if has_unknown_shape(right):
                yield
            # if right is a DataFrame, make sure 1 chunk on axis columns
            right = yield from recursive_tile(
                right.rechunk({1: right.shape[1]}))

        out_chunks = []
        nsplits = [[]] if out.ndim == 1 else [[], [out.shape[1]]]
        i = 0
        for left_chunk in left.chunks:
            for right_chunk in right.chunks:
                chunk_op = op.copy().reset_key()
                chunk_op.tileable_op_key = op.key
                if op.output_types[0] == OutputType.dataframe:
                    shape = (np.nan, out.shape[1])
                    index_value = parse_index(out.index_value.to_pandas(),
                                              left_chunk, right_chunk,
                                              op.func, op.args, op.kwargs)
                    out_chunk = chunk_op.new_chunk([left_chunk, right_chunk],
                                                   shape=shape,
                                                   index_value=index_value,
                                                   columns_value=out.columns_value,
                                                   dtypes=out.dtypes,
                                                   index=(i, 0))
                    out_chunks.append(out_chunk)
                    nsplits[0].append(out_chunk.shape[0])
                else:
                    shape = (np.nan,)
                    index_value = parse_index(out.index_value.to_pandas(),
                                              left_chunk, right_chunk,
                                              op.func, op.args, op.kwargs)
                    out_chunk = chunk_op.new_chunk([left_chunk, right_chunk],
                                                   shape=shape,
                                                   index_value=index_value,
                                                   dtype=out.dtype,
                                                   name=out.name,
                                                   index=(i,))
                    out_chunks.append(out_chunk)
                    nsplits[0].append(out_chunk.shape[0])
                i += 1

        params = out.params
        params['nsplits'] = tuple(tuple(ns) for ns in nsplits)
        params['chunks'] = out_chunks
        new_op = op.copy()
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    @redirect_custom_log
    @enter_current_session
    def execute(cls, ctx, op: "DataFrameCartesianChunk"):
        left, right = ctx[op.left.key], ctx[op.right.key]
        ctx[op.outputs[0].key] = op.func(
            left, right, *op.args, **(op.kwargs or dict()))


def cartesian_chunk(left, right, func, args=(), **kwargs):
    output_type = kwargs.pop('output_type', None)
    output_types = kwargs.pop('output_types', None)
    object_type = kwargs.pop('object_type', None)
    output_types = validate_output_types(
        output_type=output_type, output_types=output_types, object_type=object_type)
    output_type = output_types[0] if output_types else None
    if output_type:
        output_types = [output_type]
    index = kwargs.pop('index', None)
    dtypes = kwargs.pop('dtypes', None)
    memory_scale = kwargs.pop('memory_scale', None)

    op = DataFrameCartesianChunk(left=left, right=right, func=func,
                                 args=args, kwargs=kwargs,
                                 output_types=output_types,
                                 memory_scale=memory_scale)
    return op(left, right, index=index, dtypes=dtypes)
