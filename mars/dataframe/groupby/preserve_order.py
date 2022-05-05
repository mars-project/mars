import numpy as np
import pandas as pd
from pandas import MultiIndex

from ... import opcodes as OperandDef
from mars.dataframe.operands import DataFrameOperandMixin, DataFrameOperand
from mars.utils import lazy_import
from ...core.operand import OperandStage, MapReduceOperand
from ...serialization.serializables import Int32Field, AnyField, StringField, ListField, BoolField

cudf = lazy_import("cudf", globals=globals())


class DataFrameOrderPreserveIndexOperand(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.GROUPBY_SORT_ORDER_INDEX

    _index_prefix = Int32Field("index_prefix")

    def __init__(self, output_types=None, index_prefix=None, *args, **kwargs):
        super().__init__(_output_types=output_types, _index_prefix=index_prefix, *args, **kwargs)

    @property
    def index_prefix(self):
        return self._index_prefix

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

        min_table = xdf.DataFrame({"min_col": np.arange(0, len((a))), "index": op.index_prefix} , index=a.index)

        ctx[op.outputs[-1].key] = min_table


class DataFrameOrderPreservePivotOperand(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.GROUPBY_SORT_ORDER_PIVOT

    _n_partition = Int32Field("n_partition")
    _by = AnyField("by")

    def __init__(self, n_partition=None, output_types=None, by=None, *args, **kwargs):
        super().__init__(_n_partition=n_partition, _output_types=output_types, _by=by, *args, **kwargs)

    @property
    def by(self):
        return self._by

    @property
    def output_limit(self):
        return 2

    @classmethod
    def execute(cls, ctx, op):
        inputs = [ctx[c.key] for c in op.inputs if len(ctx[c.key]) > 0]
        if len(inputs) == 0:
            # corner case: nothing sampled, we need to do nothing
            ctx[op.outputs[0].key] = ctx[op.outputs[-1].key] = ctx[op.inputs[0].key]
            return

        xdf = pd if isinstance(inputs[0], (pd.DataFrame, pd.Series)) else cudf

        a = xdf.concat(inputs, axis=0)
        a = a.sort_index()
        # a = a.groupby(op.by).min(['index', 'min_col'])
        a_group = a.groupby(op.by).groups
        a_list = []
        for g in a_group:
            group_df = a.loc[g]
            group_min_index = group_df['index'].min()
            group_min_col = group_df.loc[group_df['index'] == group_min_index]['min_col'].min()
            if isinstance(a.axes[0], MultiIndex):
                index = pd.MultiIndex.from_tuples([g], names=group_df.index.names)
            else:
                index = pd.Index([g], name=group_df.index.names)
            a_list_df = pd.DataFrame({"index" : group_min_index, "min_col" : group_min_col}, index=index)
            a_list.append(a_list_df)

        a = pd.concat(a_list)

        ctx[op.outputs[0].key] = a

        sort_values_df = a.sort_values(['index', 'min_col'])

        p = len(inputs)
        if len(sort_values_df) < p:
            num = p // len(a) + 1
            sort_values_df = sort_values_df.append([sort_values_df] * (num - 1))

        sort_values_df = sort_values_df.sort_values(['index', 'min_col'])

        w = sort_values_df.shape[0] * 1.0 / (p + 1)

        values = sort_values_df[['index', 'min_col']].values

        slc = np.linspace(
            max(w-1, 0), len(sort_values_df) - 1, num=len(op.inputs) - 1, endpoint=False
        ).astype(int)
        out = values[slc]
        ctx[op.outputs[-1].key] = out

class DataFrameGroupbyOrderPresShuffle(MapReduceOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.GROUPBY_SORT_SHUFFLE

    _by = ListField("by")
    _n_partition = Int32Field("n_partition")

    def __init__(
            self,
            by=None,
            n_partition=None,
            output_types=None,
            **kw
    ):
        super().__init__(
            _by=by,
            _n_partition=n_partition,
            _output_types=output_types,
            **kw
        )

    @property
    def by(self):
        return self._by

    @property
    def n_partition(self):
        return self._n_partition

    @property
    def output_limit(self):
        return 1


    @classmethod
    def _execute_dataframe_map(cls, ctx, op):
        df, pivots, min_table = [ctx[c.key] for c in op.inputs]
        out = op.outputs[0]
        if isinstance(df, tuple):
            ijoin_df = tuple(x.join(min_table, how="inner") for x in df)
        else:
            ijoin_df = df.join(min_table, how="inner")

        if isinstance(df, tuple):
            for i in range(len(df)):
                ijoin_df[i].index = ijoin_df[i].index.rename(df[i].index.names) if isinstance(df[i].index, MultiIndex) else ijoin_df[i].index.rename(df[i].index.name)
        else:
            ijoin_df.index = ijoin_df.index.rename(df.index.names) if isinstance(df.index, MultiIndex) else ijoin_df.index.rename(df.index.name)

        def _get_out_df(p_index, in_df):
            if p_index == 0:
                index_upper = pivots[p_index][0]+1
                intermediary_dfs = []
                for i in range(0, index_upper):
                    if i == index_upper-1:
                        intermediary_dfs.append(in_df.loc[in_df['index'] == i].loc[in_df['min_col'] < pivots[p_index][1]])
                    else:
                        intermediary_dfs.append(in_df.loc[in_df['index'] == i])
            elif p_index == op.n_partition - 1:
                intermediary_dfs = []
                index_lower = pivots[p_index-1][0]
                index_upper = in_df['index'].max() + 1
                for i in range(index_lower, index_upper):
                    if i == index_lower:
                        intermediary_dfs.append(in_df.loc[in_df['index'] == i].loc[in_df['min_col'] >= pivots[p_index-1][1]])
                    else:
                        intermediary_dfs.append(in_df.loc[in_df['index'] == i])
            else:
                intermediary_dfs = []
                index_lower = pivots[p_index - 1][0]
                index_upper = pivots[p_index][0]+1
                if index_upper == index_lower + 1:
                    intermediary_dfs.append(
                        in_df.loc[in_df['index'] == index_lower].loc[
                            (in_df['min_col'] >= pivots[p_index - 1][1]) & (in_df['min_col'] < pivots[p_index][1])])
                else:
                    for i in range(index_lower, index_upper):
                        if i == index_lower:
                            if index_lower != index_upper:
                                intermediary_dfs.append(in_df.loc[in_df['index'] == i].loc[in_df['min_col'] >= pivots[p_index-1][1]])
                        elif i == index_upper-1:
                            intermediary_dfs.append(in_df.loc[in_df['index'] == i].loc[in_df['min_col'] < pivots[p_index][1]])
                        else:
                            intermediary_dfs.append(in_df.loc[in_df['index'] == i])
            if len(intermediary_dfs) > 0:
                out_df = pd.concat(intermediary_dfs)
            else:
                # out_df = pd.DataFrame(columns=in_df.columns)
                # out_df.index = out_df.index.rename(in_df.index.names) if isinstance(in_df.index, MultiIndex) else out_df.index.rename(in_df.index.name)
                out_df = None
            return out_df

        for i in range(op.n_partition):
            index = (i, 0)
            if isinstance(df, tuple):
                out_df = tuple(_get_out_df(i, x) for x in ijoin_df)
            else:
                out_df = _get_out_df(i, ijoin_df)
            if out_df is not None:
                ctx[out.key, index] = out_df

    @classmethod
    def _execute_map(cls, ctx, op):
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
                concat_df = xdf.concat([inp[i] for inp in raw_inputs], axis=0)
                concat_df = concat_df.sort_values(["index", "min_col"]).drop(columns=["index", "min_col"])
                r.append(concat_df)
            r = tuple(r)
        else:
            concat_df = xdf.concat(raw_inputs, axis=0)
            concat_df = concat_df.sort_values(["index", "min_col"]).drop(columns=["index", "min_col"])
            r = concat_df

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
