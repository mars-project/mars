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
from ...core import get_output_types, recursive_tile, OutputType
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
from ..utils import validate_axis, parse_index, build_empty_df


class _NoNeedToAlign(Exception):
    pass


class DataFrameAlign(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.ALIGN

    lhs = KeyField("lhs")
    rhs = KeyField("rhs")
    join = StringField("join", default=None)
    axis = Int16Field("axis", default=None)
    level = AnyField("level", default=None)
    fill_value = AnyField("fill_value", default=None)
    method = StringField("method", default=None)
    limit = Int64Field("limit", default=None)
    fill_axis = Int16Field("fill_axis", default=None)
    broadcast_axis = Int16Field("broadcast_axis", default=None)

    @property
    def output_limit(self) -> int:
        return 2

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self.lhs = inputs[0]
        self.rhs = inputs[1]

    def __call__(self, lhs: TileableType, rhs: TileableType):
        if self.broadcast_axis != 1 or lhs.ndim == rhs.ndim:
            self._output_types = get_output_types(lhs, rhs)
        else:
            self._output_types = [OutputType.dataframe, OutputType.dataframe]

        if lhs.ndim == rhs.ndim:
            if lhs.ndim == 1:
                return self._call_series_series(lhs, rhs)
            else:
                return self._call_dataframe_dataframe(lhs, rhs)
        else:
            if lhs.ndim == 1:
                # join order need to be reversed if not symmetric
                asym_joins = {"left", "right"} - {self.join}
                if len(asym_joins) == 1:  # self.join in {"left", "right"}
                    self.join = asym_joins.pop()
                # need to put dataframe first
                self._output_types = get_output_types(rhs, lhs)
                return self._call_dataframe_series(rhs, lhs)[::-1]
            else:
                return self._call_dataframe_series(lhs, rhs)

    def _call_dataframe_dataframe(self, lhs: TileableType, rhs: TileableType):
        l_shape = list(lhs.shape)
        r_shape = list(rhs.shape)
        if self.axis is None or self.axis == 0:
            l_idx_val = r_idx_val = self._merge_index(
                lhs.index_value, rhs.index_value, how=self.join
            )
            l_shape[0] = r_shape[0] = np.nan
        else:
            l_idx_val, r_idx_val = lhs.index_value, rhs.index_value

        if self.axis is None or self.axis == 1:
            l_empty = build_empty_df(lhs.dtypes)
            r_empty = build_empty_df(rhs.dtypes)
            aligned, _ = l_empty.align(r_empty, axis=1)
            l_dtypes = r_dtypes = aligned.dtypes
            l_col_val = r_col_val = parse_index(aligned.columns, store_data=True)
            l_shape[1] = r_shape[1] = len(l_dtypes)
        else:
            l_dtypes, r_dtypes = lhs.dtypes, rhs.dtypes
            l_col_val, r_col_val = lhs.columns_value, rhs.columns_value

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
        if self.axis == 0 or self.broadcast_axis == 1:
            dtypes = lhs.dtypes
            col_val = lhs.columns_value
            l_idx_val = r_idx_val = self._merge_index(
                lhs.index_value, rhs.index_value, how=self.join
            )
            l_shape[0] = r_size = np.nan
        else:
            l_idx_val = lhs.index_value
            if not rhs.index_value.has_value():
                dtypes = None
                l_shape[1] = r_size = np.nan
                col_val = r_idx_val = self._merge_index(
                    lhs.columns_value, rhs.index_value, how=self.join
                )
            else:
                series_index = rhs.index_value.to_pandas()
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
        if self.broadcast_axis == 1:
            r_kws = {
                "index_value": r_idx_val,
                "dtypes": dtypes,
                "shape": tuple(l_shape),
                "columns_value": col_val,
            }
        else:
            r_kws = {
                "index_value": r_idx_val,
                "shape": (r_size,),
                "dtype": rhs.dtype,
            }
        return self.new_tileables([lhs, rhs], kws=[l_kws, r_kws])

    def _call_series_series(self, lhs: TileableType, rhs: TileableType):
        idx = self._merge_index(lhs.index_value, rhs.index_value, how=self.join)
        kws = [
            {"index_value": idx, "shape": (np.nan,), "dtype": lhs.dtype},
            {"index_value": idx, "shape": (np.nan,), "dtype": rhs.dtype},
        ]
        return self.new_tileables([lhs, rhs], kws=kws)

    @staticmethod
    def _merge_index(
        left_index_value: IndexValue, right_index_value: IndexValue, how: str = "outer"
    ):
        left_pd = left_index_value.to_pandas()
        right_pd = right_index_value.to_pandas()

        if not left_index_value.has_value() or not right_index_value.has_value():
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
            axis = op.axis % tileable.ndim
            return [
                tuple(val_to_replace[op.axis]) if i == axis else attr_val[i]
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
    def _check_align_needed(
        cls, op: "DataFrameAlign", left_chunks: list, right_chunks: list
    ):
        lhs, rhs = op.lhs, op.rhs
        if all(lc.key == rc.key for lc, rc in zip(lhs.chunks, left_chunks)) and all(
            lc.key == rc.key for lc, rc in zip(rhs.chunks, right_chunks)
        ):
            raise _NoNeedToAlign

    @classmethod
    def _tile_dataframe_dataframe(cls, op: "DataFrameAlign"):
        lhs, rhs = op.lhs, op.rhs
        nsplits, chunk_shapes, left_chunks, right_chunks = align_dataframe_dataframe(
            lhs, rhs, axis=op.axis
        )
        cls._check_align_needed(op, left_chunks, right_chunks)

        left_chunk_array = np.array(left_chunks, dtype="O").reshape(chunk_shapes[0])
        right_chunk_array = np.array(right_chunks, dtype="O").reshape(chunk_shapes[1])

        left_idx_to_chunk = dict()
        l_chunks, r_chunks = [], []

        iterator = np.nditer(right_chunk_array, flags=["refs_ok", "multi_index"])
        for rc_obj in iterator:
            rc = rc_obj.tolist()
            r_index = iterator.multi_index
            l_index = tuple(r_index[i] % chunk_shapes[0][i] for i in (0, 1))
            lc = left_chunk_array[l_index]

            kws = [lc.params, rc.params]
            kws[0]["index"] = l_index
            kws[1]["index"] = r_index

            chunk_op = op.copy().reset_key()
            l_chunk, r_chunk = chunk_op.new_chunks([lc, rc], kws=kws)
            left_idx_to_chunk[l_index] = l_chunk
            r_chunks.append(r_chunk)

        iterator = np.nditer(left_chunk_array, flags=["refs_ok", "multi_index"])
        for lc_obj in iterator:
            lc = lc_obj.tolist()
            l_index = iterator.multi_index
            try:
                l_chunk = left_idx_to_chunk[l_index]
                l_chunks.append(l_chunk)
                continue
            except KeyError:
                pass

            r_index = tuple(l_index[i] % chunk_shapes[1][i] for i in (0, 1))
            rc = right_chunk_array[r_index]

            kws = [lc.params, rc.params]
            kws[0]["index"] = l_index

            chunk_op = op.copy().reset_key()
            l_chunk, _r_chunk = chunk_op.new_chunks([lc, rc], kws=kws)
            l_chunks.append(l_chunk)

        return nsplits, l_chunks, r_chunks

    @classmethod
    def _tile_dataframe_series(cls, op: "DataFrameAlign"):
        lhs, rhs = op.lhs, op.rhs
        nsplits, left_chunk_shape, left_chunks, right_chunks = align_dataframe_series(
            lhs, rhs, axis=op.axis
        )
        cls._check_align_needed(op, left_chunks, right_chunks)

        left_chunk_array = np.array(left_chunks, dtype="O").reshape(left_chunk_shape)
        axis = op.axis if op.broadcast_axis != 1 else 0
        l_chunks, r_chunks = [], []
        iterator = np.nditer(left_chunk_array, flags=["refs_ok", "multi_index"])
        for c_obj in iterator:
            c = c_obj.tolist()
            l_index = iterator.multi_index

            right_chunk = right_chunks[l_index[axis]]
            kws = [c.params, right_chunk.params]
            kws[0]["index"] = l_index
            if op.broadcast_axis != 1:
                kws[1]["index"] = (l_index[axis],)
            else:
                kws[1]["index"] = l_index

            chunk_op = op.copy().reset_key()
            l_chunk, r_chunk = chunk_op.new_chunks([c, right_chunk], kws=kws)

            l_chunks.append(l_chunk)
            if op.broadcast_axis == 1 or l_index[1 - axis] == 0:
                r_chunks.append(r_chunk)

        return nsplits, l_chunks, r_chunks

    @classmethod
    def _tile_series_series(cls, op: "DataFrameAlign"):
        nsplits, _, left_chunks, right_chunks = align_series_series(op.lhs, op.rhs)
        cls._check_align_needed(op, left_chunks, right_chunks)

        l_chunks, r_chunks = [], []
        for idx, (lc, rc) in enumerate(zip(left_chunks, right_chunks)):
            kws = [lc.params, rc.params]
            kws[0]["index"] = kws[1]["index"] = (idx,)

            chunk_op = op.copy().reset_key()
            l_chunk, r_chunk = chunk_op.new_chunks([lc, rc], kws=kws)
            l_chunks.append(l_chunk)
            r_chunks.append(r_chunk)
        return nsplits, l_chunks, r_chunks

    @classmethod
    def _tile_with_fillna(cls, tileable: TileableType):
        op = tileable.op
        if op.method is None:
            return tileable
        axis = op.fill_axis if tileable.ndim == 2 else 0
        tileable = tileable.fillna(method=op.method, limit=op.limit, axis=axis)
        return (yield from recursive_tile(tileable))

    @classmethod
    def _make_direct_output_kws(cls, left: TileableType, right: TileableType):
        kws = [left.params, right.params]
        kws[0].update(dict(chunks=left.chunks, nsplits=left.nsplits))
        kws[1].update(dict(chunks=right.chunks, nsplits=right.nsplits))
        return kws

    @classmethod
    def tile(cls, op: "DataFrameAlign"):
        try:
            if op.lhs.ndim == op.rhs.ndim:
                if op.lhs.ndim == 2:
                    nsplits, left_chunks, right_chunks = cls._tile_dataframe_dataframe(
                        op
                    )
                else:
                    nsplits, left_chunks, right_chunks = cls._tile_series_series(op)
            else:
                nsplits, left_chunks, right_chunks = cls._tile_dataframe_series(op)
        except _NoNeedToAlign:
            kws = cls._make_direct_output_kws(op.lhs, op.rhs)
        else:
            kws = [
                cls._build_tiled_kw(op, 0, left_chunks, nsplits),
                cls._build_tiled_kw(op, 1, right_chunks, nsplits),
            ]
        new_left, new_right = op.copy().new_tileables(op.inputs, kws=kws)

        new_left_filled = yield from cls._tile_with_fillna(new_left)
        new_right_filled = yield from cls._tile_with_fillna(new_right)
        if new_left_filled is not new_left or new_right_filled is not new_right:
            kws = cls._make_direct_output_kws(new_left_filled, new_right_filled)
            new_left, new_right = op.copy().new_tileables(op.inputs, kws=kws)

        return [new_left, new_right]

    @classmethod
    def execute(cls, ctx, op: "DataFrameAlign"):
        lhs_val = ctx[op.lhs.key]
        rhs_val = ctx[op.rhs.key]
        l_res, r_res = lhs_val.align(
            rhs_val,
            axis=op.axis,
            join=op.join,
            fill_value=op.fill_value,
            broadcast_axis=op.broadcast_axis,
        )
        ctx[op.outputs[0].key] = l_res
        ctx[op.outputs[1].key] = r_res


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
    """
    Align two objects on their axes with the specified join method.

    Join method is specified for each axis Index.

    Parameters
    ----------
    other : DataFrame or Series
    join : {'outer', 'inner', 'left', 'right'}, default 'outer'
    axis : allowed axis of the other object, default None
        Align on index (0), columns (1), or both (None).
    level : int or level name, default None
        Broadcast across a level, matching Index values on the
        passed MultiIndex level.
    copy : bool, default True
        Always returns new objects. If copy=False and no reindexing is
        required then original objects are returned.
    fill_value : scalar, default np.NaN
        Value to use for missing values. Defaults to NaN, but can be any
        "compatible" value.
    method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
        Method to use for filling holes in reindexed Series:

        - pad / ffill: propagate last valid observation forward to next valid.
        - backfill / bfill: use NEXT valid observation to fill gap.

    limit : int, default None
        If method is specified, this is the maximum number of consecutive
        NaN values to forward/backward fill. In other words, if there is
        a gap with more than this number of consecutive NaNs, it will only
        be partially filled. If method is not specified, this is the
        maximum number of entries along the entire axis where NaNs will be
        filled. Must be greater than 0 if not None.
    fill_axis : {0 or 'index', 1 or 'columns'}, default 0
        Filling axis, method and limit.
    broadcast_axis : {0 or 'index', 1 or 'columns'}, default None
        Broadcast values along this axis, if aligning two objects of
        different dimensions.

    Notes
    -----
    Currently argument `level` is not supported.

    Returns
    -------
    (left, right) : (DataFrame, type of other)
        Aligned objects.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> import mars.dataframe as md
    >>> df = md.DataFrame(
    ...     [[1, 2, 3, 4], [6, 7, 8, 9]], columns=["D", "B", "E", "A"], index=[1, 2]
    ... )
    >>> other = md.DataFrame(
    ...     [[10, 20, 30, 40], [60, 70, 80, 90], [600, 700, 800, 900]],
    ...     columns=["A", "B", "C", "D"],
    ...     index=[2, 3, 4],
    ... )
    >>> df.execute()
       D  B  E  A
    1  1  2  3  4
    2  6  7  8  9
    >>> other.execute()
        A    B    C    D
    2   10   20   30   40
    3   60   70   80   90
    4  600  700  800  900

    Align on columns:

    >>> left, right = df.align(other, join="outer", axis=1)
    >>> left.execute()
       A  B   C  D  E
    1  4  2 NaN  1  3
    2  9  7 NaN  6  8
    >>> right.execute()
        A    B    C    D   E
    2   10   20   30   40 NaN
    3   60   70   80   90 NaN
    4  600  700  800  900 NaN

    We can also align on the index:

    >>> left, right = df.align(other, join="outer", axis=0)
    >>> left.execute()
        D    B    E    A
    1  1.0  2.0  3.0  4.0
    2  6.0  7.0  8.0  9.0
    3  NaN  NaN  NaN  NaN
    4  NaN  NaN  NaN  NaN
    >>> right.execute()
        A      B      C      D
    1    NaN    NaN    NaN    NaN
    2   10.0   20.0   30.0   40.0
    3   60.0   70.0   80.0   90.0
    4  600.0  700.0  800.0  900.0

    Finally, the default `axis=None` will align on both index and columns:

    >>> left, right = df.align(other, join="outer", axis=None)
    >>> left.execute()
         A    B   C    D    E
    1  4.0  2.0 NaN  1.0  3.0
    2  9.0  7.0 NaN  6.0  8.0
    3  NaN  NaN NaN  NaN  NaN
    4  NaN  NaN NaN  NaN  NaN
    >>> right.execute()
           A      B      C      D   E
    1    NaN    NaN    NaN    NaN NaN
    2   10.0   20.0   30.0   40.0 NaN
    3   60.0   70.0   80.0   90.0 NaN
    4  600.0  700.0  800.0  900.0 NaN
    """
    axis = validate_axis(axis) if axis is not None else None
    fill_axis = validate_axis(fill_axis) if fill_axis is not None else None
    broadcast_axis = (
        validate_axis(broadcast_axis) if broadcast_axis is not None else None
    )

    if level is not None:
        raise NotImplementedError(f"Argument `level` not supported")
    if df.ndim != other.ndim and axis is None:
        raise ValueError("Must specify axis=0 or 1")

    op = DataFrameAlign(
        join=join,
        axis=axis,
        level=level,
        copy=copy,
        fill_value=fill_value,
        method=method,
        limit=limit,
        fill_axis=fill_axis,
        broadcast_axis=broadcast_axis,
    )
    return op(df, other)
