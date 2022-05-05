import numpy as np
import pandas as pd

from mars.dataframe.operands import DataFrameOperandMixin
from mars.dataframe.sort.psrs import DataFramePSRSChunkOperand
from mars.utils import lazy_import
from ..utils import is_cudf

from ... import opcodes as OperandDef
from ...core import OutputType
from ...core.operand import MapReduceOperand, OperandStage
from ...serialization.serializables import StringField, Int32Field, BoolField, ListField, FieldTypes

cudf = lazy_import("cudf", globals=globals())


class _Largest:
    """
    This util class resolve TypeError when
    comparing strings with None values
    """

    def __lt__(self, other):
        return False

    def __gt__(self, other):
        return self is not other


_largest = _Largest()


def _series_to_df(in_series, xdf):
    in_df = in_series.to_frame()
    if in_series.name is not None:
        in_df.columns = xdf.Index([in_series.name])
    return in_df


class DataFrameGroupbyConcatPivot(DataFramePSRSChunkOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.GROUPBY_SORT_PIVOT

    @property
    def output_limit(self):
        return 1

    @classmethod
    def execute(cls, ctx, op):
        inputs = [ctx[c.key] for c in op.inputs if len(ctx[c.key]) > 0]
        if len(inputs) == 0:
            # corner case: nothing sampled, we need to do nothing
            ctx[op.outputs[-1].key] = ctx[op.inputs[0].key]
            return

        xdf = pd if isinstance(inputs[0], (pd.DataFrame, pd.Series)) else cudf

        a = xdf.concat(inputs, axis=0)
        a = a.sort_index()
        index = a.index.drop_duplicates()

        p = len(inputs)
        if len(index) < p:
            num = p // len(index) + 1
            index = index.append([index] * (num - 1))

        index = index.sort_values()

        values = index.values

        slc = np.linspace(
            p - 1, len(index) - 1, num=len(op.inputs) - 1, endpoint=False
        ).astype(int)
        out = values[slc]
        ctx[op.outputs[-1].key] = out


class DataFramePSRSGroupbySample(DataFramePSRSChunkOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.GROUPBY_SORT_REGULAR_SAMPLE

    @property
    def output_limit(self):
        return 1

    @classmethod
    def execute(cls, ctx, op):
        a = ctx[op.inputs[0].key][0]
        xdf = pd if isinstance(a, (pd.DataFrame, pd.Series)) else cudf
        if len(a) == 0:
            # when chunk is empty, return the empty chunk itself
            ctx[op.outputs[0].key] = a
            return
        if isinstance(a, xdf.Series) and op.output_types[0] == OutputType.dataframe:
            a = _series_to_df(a, xdf)

        n = op.n_partition
        if a.shape[0] < n:
            num = n // a.shape[0] + 1
            a = xdf.concat([a] * num).sort_index()

        w = a.shape[0] * 1.0 / (n + 1)

        slc = np.linspace(max(w - 1, 0), a.shape[0] - 1, num=n, endpoint=False).astype(
            int
        )

        out = a.iloc[slc]
        ctx[op.outputs[-1].key] = out


class DataFrameGroupbySortShuffle(MapReduceOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.GROUPBY_SORT_SHUFFLE

    _sort_type = StringField("sort_type")

    # for shuffle map
    _axis = Int32Field("axis")
    _by = ListField("by")
    _ascending = BoolField("ascending")
    _inplace = BoolField("inplace")
    _na_position = StringField("na_position")
    _n_partition = Int32Field("n_partition")

    # for sort_index
    _level = ListField("level")
    _sort_remaining = BoolField("sort_remaining")

    # for shuffle reduce
    _kind = StringField("kind")

    def __init__(
            self,
            sort_type=None,
            by=None,
            axis=None,
            ascending=None,
            n_partition=None,
            na_position=None,
            inplace=None,
            kind=None,
            level=None,
            sort_remaining=None,
            output_types=None,
            **kw
    ):
        super().__init__(
            _sort_type=sort_type,
            _by=by,
            _axis=axis,
            _ascending=ascending,
            _n_partition=n_partition,
            _na_position=na_position,
            _inplace=inplace,
            _kind=kind,
            _level=level,
            _sort_remaining=sort_remaining,
            _output_types=output_types,
            **kw
        )

    @property
    def sort_type(self):
        return self._sort_type

    @property
    def by(self):
        return self._by

    @property
    def axis(self):
        return self._axis

    @property
    def ascending(self):
        return self._ascending

    @property
    def inplace(self):
        return self._inplace

    @property
    def na_position(self):
        return self._na_position

    @property
    def level(self):
        return self._level

    @property
    def sort_remaining(self):
        return self._sort_remaining

    @property
    def n_partition(self):
        return self._n_partition

    @property
    def kind(self):
        return self._kind

    @property
    def output_limit(self):
        return 1

    @classmethod
    def _execute_dataframe_map(cls, ctx, op):
        df, pivots = [ctx[c.key] for c in op.inputs]
        out = op.outputs[0]

        def _get_out_df(p_index, in_df):
            if p_index == 0:
                out_df = in_df.loc[: pivots[p_index]]
            elif p_index == op.n_partition - 1:
                out_df = in_df.loc[pivots[p_index - 1]:].drop(
                    index=pivots[p_index - 1], errors="ignore"
                )
            else:
                out_df = in_df.loc[pivots[p_index - 1]: pivots[p_index]].drop(
                    index=pivots[p_index - 1], errors="ignore"
                )
            return out_df

        for i in range(op.n_partition):
            index = (i, 0)
            if isinstance(df, tuple):
                out_df = tuple(_get_out_df(i, x) for x in df)
            else:
                out_df = _get_out_df(i, df)
            ctx[out.key, index] = out_df

    @classmethod
    def _execute_map(cls, ctx, op):
        a = [ctx[c.key] for c in op.inputs][0]
        if isinstance(a, tuple):
            a = a[0]
        cls._execute_dataframe_map(ctx, op)

    @classmethod
    def _execute_reduce(cls, ctx, op: "DataFramePSRSShuffle"):
        out_chunk = op.outputs[0]
        raw_inputs = list(op.iter_mapper_data(ctx, pop=False))
        by = op.by
        xdf = cudf if op.gpu else pd

        r = []

        if isinstance(raw_inputs[0], tuple):
            tuple_len = len(raw_inputs[0])
            for i in range(tuple_len):
                r.append(xdf.concat([inp[i] for inp in raw_inputs], axis=0))
            r = tuple(r)
        else:
            r = xdf.concat(raw_inputs, axis=0)

        if isinstance(r, tuple):
            ctx[op.outputs[0].key] = r + (by,)
        else:
            ctx[op.outputs[0].key] = (r, by)

    @classmethod
    def estimate_size(cls, ctx, op):
        super().estimate_size(ctx, op)
        result = ctx[op.outputs[0].key]
        if op.stage == OperandStage.reduce:
            ctx[op.outputs[0].key] = (result[0], result[1] * 1.5)
        else:
            ctx[op.outputs[0].key] = result

    @classmethod
    def execute(cls, ctx, op):
        if op.stage == OperandStage.map:
            cls._execute_map(ctx, op)
        else:
            cls._execute_reduce(ctx, op)
