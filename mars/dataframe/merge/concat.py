# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

import pandas as pd
import numpy as np

from ...serialize import ValueType, ListField, StringField, BoolField, AnyField
from ... import opcodes as OperandDef
from ...utils import lazy_import
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType

cudf = lazy_import('cudf')


class DataFrameConcat(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.CONCATENATE

    _axis = AnyField('axis')
    _join = StringField('join')
    _join_axes = ListField('join_axes', ValueType.key)
    _ignore_index = BoolField('ignore_index')
    _keys = ListField('keys')
    _levels = ListField('levels')
    _names = ListField('names')
    _verify_integrity = BoolField('verify_integrity')
    _sort = BoolField('sort')
    _copy = BoolField('copy')

    def __init__(self, axis=None, join=None, join_axes=None, ignore_index=None,
                 keys=None, levels=None, names=None, verify_integrity=None,
                 sort=None, copy=None, sparse=None, object_type=None, **kw):
        super(DataFrameConcat, self).__init__(
            _axis=axis, _join=join, _join_axes=join_axes, _ignore_index=ignore_index,
            _keys=keys, _levels=levels, _names=names,
            _verify_integrity=verify_integrity, _sort=sort, _copy=copy,
            _sparse=sparse, _object_type=object_type, **kw)

    @property
    def axis(self):
        return self._axis

    @property
    def join(self):
        return self._join

    @property
    def join_axes(self):
        return self._join_axes

    @property
    def ignore_index(self):
        return self._ignore_index

    @property
    def keys(self):
        return self._keys

    @property
    def level(self):
        return self._levels

    @property
    def name(self):
        return self._name

    @property
    def verify_integrity(self):
        return self._verify_integrity

    @property
    def sort(self):
        return self._sort

    @property
    def copy_(self):
        return self._copy

    @classmethod
    def execute(cls, ctx, op):
        def _base_concat(chunk, inputs):
            # auto generated concat when executing a DataFrame, Series or Index
            if chunk.op.object_type == ObjectType.dataframe:
                return _auto_concat_dataframe_chunks(chunk, inputs)
            elif chunk.op.object_type == ObjectType.series:
                return _auto_concat_series_chunks(chunk, inputs)
            else:
                raise TypeError('Only DataFrameChunk, SeriesChunk and IndexChunk '
                                'can be automatically concatenated')

        def _auto_concat_dataframe_chunks(chunk, inputs):
            if chunk.op.axis is not None:
                return pd.concat(inputs, axis=op.axis)
            # auto generated concat when executing a DataFrame
            n_rows = max(inp.index[0] for inp in chunk.inputs) + 1
            n_cols = int(len(inputs) // n_rows)
            assert n_rows * n_cols == len(inputs)

            xdf = pd if isinstance(inputs[0], pd.DataFrame) else cudf

            concats = []
            for i in range(n_rows):
                concat = xdf.concat([inputs[i * n_cols + j] for j in range(n_cols)], axis=1)
                concats.append(concat)

            if xdf is pd:
                # The `sort=False` is to suppress a `FutureWarning` of pandas, when the index or column of chunks to
                # concatenate is not aligned, which may happens for certain ops.
                #
                # See also Note [Columns of Left Join] in test_merge_execution.py.
                ret = xdf.concat(concats, sort=False)
            else:
                ret = xdf.concat(concats)
            if getattr(chunk.index_value, 'should_be_monotonic', False):
                ret.sort_index(inplace=True)
            if getattr(chunk.columns, 'should_be_monotonic', False):
                ret.sort_index(axis=1, inplace=True)
            return ret

        def _auto_concat_series_chunks(chunk, inputs):
            # auto generated concat when executing a Series
            if all(np.isscalar(inp) for inp in inputs):
                return pd.Series(inputs)
            else:
                xdf = pd if isinstance(inputs[0], pd.Series) else cudf
                if chunk.op.axis is not None:
                    concat = xdf.concat(inputs, axis=chunk.op.axis)
                else:
                    concat = xdf.concat(inputs)
                if getattr(chunk.index_value, 'should_be_monotonic', False):
                    concat.sort_index(inplace=True)
                return concat

        chunk = op.outputs[0]
        inputs = [ctx[input.key] for input in op.inputs]

        if isinstance(inputs[0], tuple):
            ctx[chunk.key] = tuple(_base_concat(chunk, [input[i] for input in inputs])
                                   for i in range(len(inputs[0])))
        else:
            ctx[chunk.key] = _base_concat(chunk, inputs)
