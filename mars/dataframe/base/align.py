# Copyright 1999-2022 Alibaba Group Holding Ltd.
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

from typing import Any, Optional, Union

import numpy as np

from ... import opcodes
from ...core import get_output_types
from ...serialization.serializables import (
    AnyField,
    KeyField,
    StringField,
    Int16Field,
    Int64Field,
)
from ...typing import TileableType
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..align import (
    align_dataframe_dataframe,
    align_dataframe_series,
    align_series_series,
)
from ..core import IndexValue
from ..utils import validate_axis, parse_index, build_empty_df, is_index_value_identical


class DataFrameAlign(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.ALIGN

    lhs = KeyField("lhs")
    rhs = KeyField("rhs")
    join = StringField("join")
    axis = Int16Field("axis")
    level = AnyField("level")
    fill_value = AnyField("fill_value")
    method = StringField("method")
    limit = Int64Field("limit")
    fill_axis = Int16Field("fill_axis")
    broadcast_axis = Int16Field("broadcast_axis")

    @property
    def output_limit(self) -> int:
        return 2

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self.lhs = inputs[0]
        self.rhs = inputs[1]

    def __call__(self, lhs: TileableType, rhs: TileableType):
        self._output_types = get_output_types(lhs, rhs)
        if lhs.ndim == rhs.ndim:
            if lhs.ndim == 1:
                return self._call_series_series(lhs, rhs)
            else:
                return self._call_dataframe_dataframe(lhs, rhs)
        else:
            if lhs.ndim == 1:
                return self._call_dataframe_series(rhs, lhs)[::-1]
            else:
                return self._call_dataframe_series(lhs, rhs)

    def _call_dataframe_dataframe(self, lhs: TileableType, rhs: TileableType):
        altered = False
        l_shape = list(lhs.shape)
        r_shape = list(rhs.shape)
        if (self.axis is None or self.axis == 0) and not is_index_value_identical(
            lhs.index_value, rhs.index_value
        ):
            l_idx_val = r_idx_val = self._merge_index(
                lhs.index_value, rhs.index_value, how=self.method
            )
            l_shape[0] = r_shape[0] = np.nan
            altered = True
        else:
            l_idx_val, r_idx_val = lhs.index_value, rhs.index_value

        if (
            self.axis is None or self.axis == 1
        ) and lhs.columns_value.key != rhs.columns_value.key:
            l_empty = build_empty_df(lhs.dtypes)
            r_empty = build_empty_df(rhs.dtypes)
            aligned, _ = l_empty.align(r_empty, axis=1)
            l_dtypes, r_dtypes = aligned.dtypes
            l_col_val = r_col_val = parse_index(aligned.columns, store_data=True)
            l_shape[1] = r_shape[1] = np.nan
            altered = True
        else:
            l_dtypes, r_dtypes = lhs.dtypes, rhs.dtypes
            l_col_val, r_col_val = lhs.columns_value, rhs.columns_value

        if not altered:
            return lhs, rhs

        l_kws = {
            "index_value": l_idx_val,
            "dtypes": l_dtypes,
            "shape": tuple(l_shape),
            "columns_value": l_col_val,
        }
        r_kws = {
            "index_value": r_idx_val,
            "dtypes": r_dtypes,
            "shape": tuple(r_shape),
            "columns_value": r_col_val,
        }
        return self.new_tileables([lhs, rhs], kws=[l_kws, r_kws])

    def _call_dataframe_series(self, lhs: TileableType, rhs: TileableType):
        l_shape = list(lhs.shape)
        if self.axis == 0:
            dtypes = lhs.dtypes
            col_val = lhs.columns_value
            l_idx_val = r_idx_val = self._merge_index(
                lhs.index_value, rhs.index_value, how=self.join
            )
            l_shape[0] = r_size = np.nan
        else:
            l_idx_val = lhs.index_value
            series_index = rhs.index_value.to_pandas()
            if not rhs.index_value.has_value:
                dtypes = None
                l_shape[1] = r_size = np.nan
                col_val, r_idx_val = self._merge_index(
                    lhs.columns_value, rhs.index_value, how=self.join
                )
            else:
                dtypes = lhs.dtypes.reindex(
                    lhs.dtypes.index.join(series_index, how=self.join)
                ).fillna(np.dtype(np.float_))
                l_shape[1] = r_size = len(dtypes)
                col_val = r_idx_val = parse_index(dtypes.index, store_data=True)

        l_kws = {
            "index_value": l_idx_val,
            "dtypes": dtypes,
            "shape": tuple(l_shape),
            "columns_value": col_val,
        }
        r_kws = {
            "index_value": r_idx_val,
            "shape": (r_size,),
            "dtype": rhs.dtype,
        }
        return self.new_tileables([lhs, rhs], kws=[l_kws, r_kws])

    def _call_series_series(self, lhs: TileableType, rhs: TileableType):
        if is_index_value_identical(lhs.index_value, rhs.index_value):
            return lhs, rhs

        idx = self._merge_index(lhs.index_value, rhs.index_value, how=self.join)
        return self.new_tileables([lhs, rhs], index_value=idx, shape=(np.nan,))

    @staticmethod
    def _merge_index(
        left_index_value: IndexValue, right_index_value: IndexValue, how: str = "outer"
    ):
        if left_index_value.key == right_index_value.key:
            return left_index_value

        left_pd = left_index_value.to_pandas()
        right_pd = right_index_value.to_pandas()

        if not left_index_value.has_value or not right_index_value.has_value:
            left_pd = left_pd[:0]
            right_pd = right_pd[:0]
            store_data = False
        else:
            store_data = True

        joined = left_pd.join(right_pd, how=how)
        if store_data:
            return parse_index(joined, store_data=store_data)
        else:
            return parse_index(
                joined,
                {left_index_value.key, right_index_value.key},
                store_data=store_data,
            )

    @classmethod
    def _select_nsplits(
        cls, op: "DataFrameAlign", tileable: TileableType, val_to_replace: list
    ):
        if op.axis is None:
            return val_to_replace[: tileable.ndim]
        else:
            attr_val = tileable.nsplits
            return [
                tuple(val_to_replace[i]) if i == op.axis else attr_val[i]
                for i in range(len(attr_val))
            ]

    @classmethod
    def _build_tiled_kw(
        cls, op: "DataFrameAlign", idx: int, chunks: list, nsplits: list
    ):
        in_tileable = op.inputs[idx]
        out_tileable = op.outputs[idx]
        kw = out_tileable.params.copy()
        kw.update(
            {
                "chunks": chunks,
                "nsplits": tuple(cls._select_nsplits(op, in_tileable, nsplits)),
            }
        )
        return kw

    @classmethod
    def tile(cls, op: "DataFrameAlign"):
        if op.lhs.ndim == op.rhs.ndim:
            if op.lhs.ndim == 2:
                nsplits, _, left_chunks, right_chunks = align_dataframe_dataframe(
                    op.lhs, op.rhs, axis=op.axis
                )
            else:
                nsplits, _, left_chunks, right_chunks = align_series_series(
                    op.lhs, op.rhs
                )
        else:
            if op.lhs.ndim == 2:
                nsplits, _, left_chunks, right_chunks = align_dataframe_series(
                    op.lhs, op.rhs, axis=op.axis
                )
            else:
                nsplits, _, right_chunks, left_chunks = align_dataframe_series(
                    op.rhs, op.lhs, axis=op.axis
                )

        kws = [
            cls._build_tiled_kw(op, 0, left_chunks, nsplits),
            cls._build_tiled_kw(op, 1, right_chunks, nsplits),
        ]
        return op.copy().new_tileables(op.inputs, kws=kws)


def align(
    df,
    other,
    join: str = "outer",
    axis: Union[int, str, None] = None,
    level: Union[int, str, None] = None,
    copy: bool = True,
    fill_value: Any = None,
    method: str = None,
    limit: Optional[int] = None,
    fill_axis: Union[int, str] = 0,
    broadcast_axis: Union[int, str] = None,
):
    axis = validate_axis(axis) if axis is not None else None

    if join != "outer":
        raise NotImplementedError("Non-outer join method not supported")

    locals_vals = locals()
    for var_name in ["level", "fill_value", "method", "broadcast_axis", "limit"]:
        if locals_vals[var_name] is not None:
            raise NotImplementedError(f"Non-none {var_name} not supported")

    if df.ndim != other.ndim and axis is None:
        raise ValueError("Must specify axis=0 or 1")

    op = DataFrameAlign(
        join=join,
        axis=axis,
        level=level,
        copy=copy,
        fill_value=fill_value,
        metohd=method,
        limit=limit,
        fill_axis=fill_axis,
        broadcast_axis=broadcast_axis,
    )
    return op(df, other)
