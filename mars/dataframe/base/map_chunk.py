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
from ...core import recursive_tile, get_output_types, ENTITY_TYPE, CHUNK_TYPE
from ...core.custom_log import redirect_custom_log
from ...serialization.serializables import (
    KeyField,
    FunctionField,
    TupleField,
    DictField,
    BoolField,
    StringField,
    AnyField,
)
from ...utils import (
    enter_current_session,
    has_unknown_shape,
    quiet_stdio,
    find_objects,
    replace_objects,
)
from ..operands import DataFrameOperand, DataFrameOperandMixin, OutputType
from ..utils import (
    build_df,
    build_empty_df,
    build_series,
    parse_index,
    validate_output_types,
    build_empty_series,
    clean_up_func,
    restore_func,
)


class DataFrameMapChunk(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.MAP_CHUNK

    _input = KeyField("input")
    _func = FunctionField("func")
    _args = TupleField("args")
    _kwargs = DictField("kwargs")
    _with_chunk_index = BoolField("with_chunk_index")
    _logic_key = StringField("logic_key")
    _func_key = AnyField("func_key")
    _need_clean_up_func = BoolField("need_clean_up_func")

    def __init__(
        self,
        input=None,
        func=None,
        args=None,
        kwargs=None,
        output_types=None,
        with_chunk_index=None,
        logic_key=None,
        func_key=None,
        need_clean_up_func=False,
        **kw,
    ):
        super().__init__(
            _input=input,
            _func=func,
            _args=args,
            _kwargs=kwargs,
            _output_types=output_types,
            _with_chunk_index=with_chunk_index,
            _logic_key=logic_key,
            _func_key=func_key,
            _need_clean_up_func=need_clean_up_func,
            **kw,
        )

    @property
    def input(self):
        return self._input

    @property
    def func(self):
        return self._func

    @func.setter
    def func(self, func):
        self._func = func

    @property
    def logic_key(self):
        return self._logic_key

    @logic_key.setter
    def logic_key(self, logic_key):
        self._logic_key = logic_key

    @property
    def func_key(self):
        return self._func_key

    @func_key.setter
    def func_key(self, func_key):
        self._func_key = func_key

    @property
    def need_clean_up_func(self):
        return self._need_clean_up_func

    @need_clean_up_func.setter
    def need_clean_up_func(self, need_clean_up_func: bool):
        self._need_clean_up_func = need_clean_up_func

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def with_chunk_index(self):
        return self._with_chunk_index

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        old_inputs = find_objects(self._args, ENTITY_TYPE) + find_objects(
            self._kwargs, ENTITY_TYPE
        )
        mapping = {o: n for o, n in zip(old_inputs, self._inputs[1:])}
        self._args = replace_objects(self._args, mapping)
        self._kwargs = replace_objects(self._kwargs, mapping)
        self._input = self._inputs[0]

    def _infer_attrs_by_call(self, df_or_series):
        test_obj = (
            build_df(df_or_series, size=2)
            if df_or_series.ndim == 2
            else build_series(df_or_series, size=2, name=df_or_series.name)
        )
        kwargs = self.kwargs or dict()
        if self.with_chunk_index:
            kwargs["chunk_index"] = (0,) * df_or_series.ndim
        with np.errstate(all="ignore"), quiet_stdio():
            obj = self._func(test_obj, *self._args, **kwargs)

        if obj.ndim == 2:
            output_type = OutputType.dataframe
            dtypes = obj.dtypes
            if obj.shape == test_obj.shape:
                shape = (df_or_series.shape[0], len(dtypes))
            else:  # pragma: no cover
                shape = (np.nan, len(dtypes))
        else:
            output_type = OutputType.series
            dtypes = pd.Series([obj.dtype], name=obj.name)
            if obj.shape == test_obj.shape:
                shape = df_or_series.shape
            else:
                shape = (np.nan,)

        index_value = parse_index(
            obj.index, df_or_series, self._func, self._args, self._kwargs
        )
        return {
            "output_type": output_type,
            "index_value": index_value,
            "shape": shape,
            "dtypes": dtypes,
        }

    def __call__(self, df_or_series, index=None, dtypes=None):
        output_type = (
            self.output_types[0]
            if self.output_types
            else get_output_types(df_or_series)[0]
        )
        shape = self._kwargs.pop("shape", None)

        if output_type == OutputType.df_or_series:
            return self.new_df_or_series([df_or_series])
        elif dtypes is not None:
            index = index if index is not None else pd.RangeIndex(-1)
            index_value = parse_index(
                index, df_or_series, self._func, self._args, self._kwargs
            )
            if shape is None:  # pragma: no branch
                shape = (
                    (np.nan,)
                    if output_type == OutputType.series
                    else (np.nan, len(dtypes))
                )
        else:
            # try run to infer meta
            try:
                attrs = self._infer_attrs_by_call(df_or_series)
                output_type = attrs["output_type"]
                index_value = attrs["index_value"]
                shape = attrs["shape"]
                dtypes = attrs["dtypes"]
            except:  # noqa: E722  # nosec
                raise TypeError(
                    "Cannot determine `output_type`, "
                    "you have to specify it as `dataframe` or `series`, "
                    "for dataframe, `dtypes` is required as well "
                    "if output_type='dataframe'"
                )

        inputs = (
            [df_or_series]
            + find_objects(self.args, ENTITY_TYPE)
            + find_objects(self.kwargs, ENTITY_TYPE)
        )
        if output_type == OutputType.series:
            return self.new_series(
                inputs,
                dtype=dtypes.iloc[0],
                shape=shape,
                index_value=index_value,
                name=dtypes.name,
            )
        else:
            # dataframe
            columns_value = parse_index(dtypes.index, store_data=True)
            return self.new_dataframe(
                inputs,
                shape=shape,
                dtypes=dtypes,
                index_value=index_value,
                columns_value=columns_value,
            )

    @classmethod
    def tile(cls, op: "DataFrameMapChunk"):
        clean_up_func(op)
        inp = op.input
        out = op.outputs[0]
        out_type = op.output_types[0]

        if inp.ndim == 2 and inp.chunk_shape[1] > 1:
            if has_unknown_shape(inp):
                yield
            # if input is a DataFrame, make sure 1 chunk on axis columns
            inp = yield from recursive_tile(inp.rechunk({1: inp.shape[1]}))
        arg_input_chunks = []
        for other_inp in op.inputs[1:]:
            other_inp = yield from recursive_tile(other_inp.rechunk(other_inp.shape))
            arg_input_chunks.append(other_inp.chunks[0])

        out_chunks = []
        if out_type == OutputType.dataframe:
            nsplits = [[], [out.shape[1]]]
            pd_out_index = out.index_value.to_pandas()
        elif out_type == OutputType.series:
            nsplits = [[]]
            pd_out_index = out.index_value.to_pandas()
        else:
            # DataFrameOrSeries
            nsplits = None
            pd_out_index = None
        for chunk in inp.chunks:
            chunk_op = op.copy().reset_key()
            chunk_op.tileable_op_key = op.key
            if out_type == OutputType.df_or_series:
                if inp.ndim == 2:
                    collapse_axis = 1
                else:
                    collapse_axis = None
                out_chunks.append(
                    chunk_op.new_chunk(
                        [chunk], index=chunk.index, collapse_axis=collapse_axis
                    )
                )
            elif out_type == OutputType.dataframe:
                if np.isnan(out.shape[0]):
                    shape = (np.nan, out.shape[1])
                else:
                    shape = (chunk.shape[0], out.shape[1])
                index_value = parse_index(pd_out_index, chunk, op.key)
                out_chunk = chunk_op.new_chunk(
                    [chunk] + arg_input_chunks,
                    shape=shape,
                    dtypes=out.dtypes,
                    index_value=index_value,
                    columns_value=out.columns_value,
                    index=(chunk.index[0], 0),
                )
                out_chunks.append(out_chunk)
                nsplits[0].append(out_chunk.shape[0])
            else:
                if np.isnan(out.shape[0]):
                    shape = (np.nan,)
                else:
                    shape = (chunk.shape[0],)
                index_value = parse_index(pd_out_index, chunk, op.key)
                out_chunk = chunk_op.new_chunk(
                    [chunk] + arg_input_chunks,
                    shape=shape,
                    index_value=index_value,
                    name=out.name,
                    dtype=out.dtype,
                    index=(chunk.index[0],),
                )
                out_chunks.append(out_chunk)
                nsplits[0].append(out_chunk.shape[0])

        params = out.params
        params["nsplits"] = tuple(tuple(ns) for ns in nsplits) if nsplits else nsplits
        params["chunks"] = out_chunks
        new_op = op.copy()
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    @redirect_custom_log
    @enter_current_session
    def execute(cls, ctx, op: "DataFrameMapChunk"):
        restore_func(ctx, op)
        inp = ctx[op.input.key]
        out = op.outputs[0]
        if len(inp) == 0:
            if op.output_types[0] == OutputType.dataframe:
                ctx[out.key] = build_empty_df(out.dtypes)
            elif op.output_types[0] == OutputType.series:
                ctx[out.key] = build_empty_series(out.dtype, name=out.name)
            else:
                raise ValueError(f"Chunk can not be empty except for dataframe/series.")
            return

        kwargs = op.kwargs or dict()
        if op.with_chunk_index:
            kwargs["chunk_index"] = out.index
        args = op.args or tuple()
        chunks = find_objects(args, CHUNK_TYPE) + find_objects(kwargs, CHUNK_TYPE)
        mapping = {chunk: ctx[chunk.key] for chunk in chunks}
        args = replace_objects(args, mapping)
        kwargs = replace_objects(kwargs, mapping)
        ctx[out.key] = op.func(inp, *args, **kwargs)


def map_chunk(df_or_series, func, args=(), kwargs=None, skip_infer=False, **kw):
    """
    Apply function to each chunk.

    Parameters
    ----------
    func : function
        Function to apply to each chunk.
    args : tuple
        Positional arguments to pass to func in addition to the array/series.
    kwargs: Dict
        Additional keyword arguments to pass as keywords arguments to func.
    skip_infer: bool, default False
        Whether infer dtypes when dtypes or output_type is not specified.

    Returns
    -------
    Series or DataFrame
        Result of applying ``func`` to each chunk of the DataFrame or Series.

    See Also
    --------
    DataFrame.apply : Perform any type of operations.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> df = md.DataFrame([[4, 9]] * 3, columns=['A', 'B'])
    >>> df.execute()
       A  B
    0  4  9
    1  4  9
    2  4  9

    Output type including Series or DataFrame will be auto inferred.

    >>> df.map_chunk(lambda c: c['A'] + c['B']).execute()
    0    13
    1    13
    2    13
    dtype: int64

    You can specify ``output_type`` by yourself if auto infer failed.

    >>> import pandas as pd
    >>> import numpy as np
    >>> df['c'] = ['s1', 's2', 's3']
    >>> df.map_chunk(lambda c: pd.concat([c['A'], c['c'].str.slice(1).astype(int)], axis=1)).execute()
    Traceback (most recent call last):
    TypeError: Cannot determine `output_type`, you have to specify it as `dataframe` or `series`...
    >>> df.map_chunk(lambda c: pd.concat([c['A'], c['c'].str.slice(1).astype(int)], axis=1),
    >>>              output_type='dataframe', dtypes=pd.Series([np.dtype(object), np.dtype(int)])).execute()
       A  c
    0  4  1
    1  4  2
    2  4  3
    """
    output_type = kw.pop("output_type", None)
    output_types = kw.pop("output_types", None)
    object_type = kw.pop("object_type", None)
    output_types = validate_output_types(
        output_type=output_type, output_types=output_types, object_type=object_type
    )
    output_type = output_types[0] if output_types else None
    if output_type:
        output_types = [output_type]
    elif skip_infer:
        output_types = [OutputType.df_or_series]
    index = kw.pop("index", None)
    dtypes = kw.pop("dtypes", None)
    with_chunk_index = kw.pop("with_chunk_index", False)
    if kw:  # pragma: no cover
        raise TypeError(f"Unknown kwargs: {kw}")

    op = DataFrameMapChunk(
        input=df_or_series,
        func=func,
        args=args,
        kwargs=kwargs or {},
        output_types=output_types,
        with_chunk_index=with_chunk_index,
    )
    return op(df_or_series, index=index, dtypes=dtypes)
