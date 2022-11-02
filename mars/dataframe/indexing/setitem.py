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

import collections
import numpy as np
import pandas as pd
from pandas.api.types import is_list_like

from ... import opcodes
from ...core import OutputType, recursive_tile
from ...serialization.serializables import KeyField, AnyField
from ...tensor.core import TENSOR_TYPE
from ...utils import pd_release_version
from ..core import DATAFRAME_TYPE, SERIES_TYPE, DataFrame
from ..initializer import DataFrame as asframe, Series as asseries
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import parse_index, is_index_value_identical

# in pandas 1.0.x, __setitem__ with a list with missing items are not allowed
_allow_set_missing_list = pd_release_version[:2] >= (1, 1)


class DataFrameSetitem(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.INDEXSETVALUE

    _target = KeyField("target")
    _indexes = AnyField("indexes")
    _value = AnyField("value")

    def __init__(self, target=None, indexes=None, value=None, output_types=None, **kw):
        super().__init__(
            _target=target,
            _indexes=indexes,
            _value=value,
            _output_types=output_types,
            **kw,
        )
        if self.output_types is None:
            self.output_types = [OutputType.dataframe]

    @property
    def target(self):
        return self._target

    @property
    def indexes(self):
        return self._indexes

    @property
    def value(self):
        return self._value

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._target = self._inputs[0]
        if len(inputs) > 1:
            self._value = self._inputs[-1]

    @staticmethod
    def _is_scalar_tensor(t):
        return isinstance(t, TENSOR_TYPE) and t.ndim == 0

    def __call__(self, target: DataFrame, value):
        raw_target = target

        inputs = [target]
        if np.isscalar(value):
            value_dtype = np.array(value).dtype
        elif self._is_scalar_tensor(value):
            inputs.append(value)
            value_dtype = value.dtype
        else:
            if isinstance(value, (pd.Series, SERIES_TYPE)):
                value = asseries(value)
                value_dtype = value.dtype
            elif isinstance(value, (pd.DataFrame, DATAFRAME_TYPE)):
                if len(self.indexes) != value.shape[1]:  # pragma: no cover
                    raise ValueError("Columns must be same length as key")

                value = asframe(value)
                value_dtype = pd.Series(list(value.dtypes), index=self._indexes)
            elif is_list_like(value) or isinstance(value, TENSOR_TYPE):
                # convert to numpy to get actual dim and shape
                if is_list_like(value):
                    value = np.array(value)

                if value.ndim == 1:
                    value = asseries(value, index=target.index)
                    value_dtype = value.dtype
                else:
                    if len(self.indexes) != value.shape[1]:  # pragma: no cover
                        raise ValueError("Columns must be same length as key")

                    value = asframe(value, index=target.index)
                    value_dtype = pd.Series(list(value.dtypes), index=self._indexes)
            else:  # pragma: no cover
                raise TypeError(
                    "Wrong value type, could be one of scalar, Series or tensor"
                )

            if target.shape[0] == 0:
                # target empty, reindex target first
                target = target.reindex(value.index)
                inputs[0] = target
            elif value.index_value.key != target.index_value.key:
                # need reindex when target df is not empty and index different
                value = value.reindex(target.index)
            inputs.append(value)

        index_value = target.index_value
        dtypes = target.dtypes.copy(deep=True)

        try:
            dtypes.loc[self._indexes] = value_dtype
        except KeyError:
            # when some index not exist, try update one by one
            if isinstance(value_dtype, pd.Series):
                for idx in self._indexes:
                    dtypes.loc[idx] = value_dtype.loc[idx]
            else:
                for idx in self._indexes:
                    dtypes.loc[idx] = value_dtype

        columns_value = parse_index(dtypes.index, store_data=True)
        ret = self.new_dataframe(
            inputs,
            shape=(target.shape[0], len(dtypes)),
            dtypes=dtypes,
            index_value=index_value,
            columns_value=columns_value,
        )
        raw_target.data = ret.data

    @classmethod
    def tile(cls, op: "DataFrameSetitem"):
        from ..merge.concat import DataFrameConcat

        out = op.outputs[0]
        target = op.target
        value = op.value
        indexes = op.indexes
        columns = target.columns_value.to_pandas()
        last_column_index = target.chunk_shape[1] - 1
        is_value_scalar = np.isscalar(value) or cls._is_scalar_tensor(value)
        has_multiple_cols = getattr(out.dtypes[indexes], "ndim", 0) > 0
        target_index_to_value_index = collections.defaultdict(list)

        if has_multiple_cols:
            append_cols = [c for c in indexes if c not in columns]
        else:
            append_cols = [indexes] if indexes not in columns else []

        if not is_value_scalar:
            rechunk_arg = {}

            # check if all chunk's index_value are identical
            is_identical = is_index_value_identical(target, value)
            if not is_identical:
                # do rechunk
                if any(np.isnan(s) for s in target.nsplits[0]) or any(
                    np.isnan(s) for s in value.nsplits[0]
                ):  # pragma: no cover
                    yield

                rechunk_arg[0] = target.nsplits[0]

            if isinstance(value, DATAFRAME_TYPE):
                if len(append_cols) < len(indexes):
                    # rechunk in column dim given distribution of indexes in target chunks
                    target_col_to_chunk_index = {
                        col: head_chunk.index[1]
                        for head_chunk in target.cix[0, :]
                        for col in head_chunk.dtypes.keys()
                    }
                    value_chunk_indexes = [
                        target_col_to_chunk_index.get(vc, None) for vc in indexes
                    ]
                    col_nsplits = []
                    last_cidx = value_chunk_indexes[0]
                    match_idxes = []
                    for cidx, idx in zip(value_chunk_indexes, indexes):
                        if cidx != last_cidx:
                            target_index_to_value_index[last_cidx].append(
                                len(col_nsplits)
                            )
                            col_nsplits.append(len(match_idxes))
                            last_cidx = cidx
                            match_idxes = [idx]
                        else:
                            match_idxes.append(idx)
                    target_index_to_value_index[last_cidx].append(len(col_nsplits))
                    col_nsplits.append(len(match_idxes))

                    # merge last column indexes and keep column order
                    last_value_index = target_index_to_value_index.pop(
                        last_column_index, []
                    )
                    append_value_index = target_index_to_value_index.pop(None, [])
                    target_index_to_value_index[None] = (
                        last_value_index + append_value_index
                    )

                    rechunk_arg[1] = col_nsplits
                else:
                    target_index_to_value_index[None] = [0]
                    rechunk_arg[1] = [len(append_cols)]

            if rechunk_arg:
                value = yield from recursive_tile(value.rechunk(rechunk_arg))

        out_chunks = []
        nsplits = [list(ns) for ns in target.nsplits]
        nsplits[1][-1] += len(append_cols)
        nsplits = tuple(tuple(ns) for ns in nsplits)

        for c in target.chunks:
            result_chunk = c

            if has_multiple_cols:
                new_indexes = [vc for vc in indexes if vc in c.dtypes]
            else:
                new_indexes = [indexes] if indexes in c.dtypes else []

            if c.index[-1] == last_column_index:
                new_indexes.extend(append_cols)

            if new_indexes:
                # update needed on current chunk
                chunk_op = op.copy().reset_key()
                chunk_op._indexes = new_indexes if has_multiple_cols else new_indexes[0]

                if pd.api.types.is_scalar(value):
                    chunk_inputs = [c]
                elif is_value_scalar:
                    chunk_inputs = [c, value.chunks[0]]
                else:
                    # get proper chunk from value chunks
                    if has_multiple_cols:
                        value_chunks = []
                        target_index = (
                            None if c.index[-1] == last_column_index else c.index[1]
                        )
                        for value_index in target_index_to_value_index[target_index]:
                            value_chunk = value.cix[c.index[0], value_index]
                            value_chunks.append(value_chunk)
                        if len(value_chunks) == 1:
                            value_chunk = value_chunks[0]
                        else:
                            # concat multiple columns by order
                            shape = (
                                value_chunks[0].shape[0],
                                sum(c.shape[1] for c in value_chunks),
                            )
                            dtypes = pd.concat([c.dtypes for c in value_chunks])
                            concat_op = DataFrameConcat(output_types=op.output_types)
                            value_chunk = concat_op.new_chunk(
                                value_chunks, shape=shape, dtypes=dtypes
                            )
                    else:
                        value_chunk = value.cix[
                            c.index[0],
                        ]

                    chunk_inputs = [c, value_chunk]

                shape = c.shape
                if append_cols and c.index[-1] == last_column_index:
                    # some columns appended at the last column of chunks
                    shape = (shape[0], shape[1] + len(append_cols))

                result_chunk = chunk_op.new_chunk(
                    chunk_inputs,
                    shape=shape,
                    index=c.index,
                )
                result_chunk._set_tileable_meta(
                    tileable_key=out.key,
                    nsplits=nsplits,
                    index_value=out.index_value,
                    columns_value=out.columns_value,
                    dtypes=out.dtypes,
                )
            out_chunks.append(result_chunk)

        params = out.params
        params["nsplits"] = nsplits
        params["chunks"] = out_chunks
        new_op = op.copy()
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    def estimate_size(cls, ctx: dict, op: "DataFrameSetitem"):
        result_size = ctx[op.target.key][0]
        ctx[op.outputs[0].key] = (result_size, result_size)

    @classmethod
    def execute(cls, ctx, op: "DataFrameSetitem"):
        target = ctx[op.target.key]
        # only deep copy when updating
        indexes = (
            (op.indexes,)
            if not isinstance(op.indexes, (tuple, list, set))
            else op.indexes
        )
        deep = bool(set(indexes) & set(target.columns))
        target = ctx[op.target.key].copy(deep=deep)
        value = ctx[op.value.key] if not np.isscalar(op.value) else op.value
        try:
            target[op.indexes] = value
        except KeyError:
            if _allow_set_missing_list:  # pragma: no cover
                raise
            else:
                existing = set(target.columns)
                new_columns = target.columns.append(
                    pd.Index([idx for idx in op.indexes if idx not in existing])
                )
                target = target.reindex(new_columns, axis=1)
                target[op.indexes] = value

        ctx[op.outputs[0].key] = target


def dataframe_setitem(df, col, value):
    op = DataFrameSetitem(target=df, indexes=col, value=value)
    return op(df, value)
