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

import copy
import itertools
import random
from collections.abc import Iterable

import numpy as np
import pandas as pd

from ... import opcodes
from ...core import ENTITY_TYPE, OutputType, get_output_types, recursive_tile
from ...core.operand import OperandStage, MapReduceOperand
from ...serialization.serializables import BoolField, DictField, Float32Field, KeyField, \
    Int32Field, Int64Field, NDArrayField, StringField
from ...tensor.operands import TensorShuffleProxy
from ...tensor.random import RandomStateField
from ...tensor.utils import gen_random_seeds
from ...utils import has_unknown_shape
from ..initializer import Series as asseries
from ..operands import DataFrameOperandMixin, DataFrameOperand
from ..utils import parse_index

_ILOC_COL_HEADER = '_gsamp_iloc_col_'
_WEIGHT_COL_HEADER = '_gsamp_weight_col_'


# code adapted from pandas.core.groupby.groupby.DataFrameGroupBy.sample
def _sample_groupby_iter(groupby, obj_index, n, frac, replace, weights,
                         random_state=None, errors='ignore'):
    if weights is None:
        ws = [None] * groupby.ngroups
    elif not isinstance(weights, Iterable) or isinstance(weights, str):
        ws = [weights] * groupby.ngroups
    else:
        weights = pd.Series(weights, index=obj_index)
        ws = [weights.iloc[idx] for idx in groupby.indices.values()]

    group_iterator = groupby.grouper.get_iterator(groupby._selected_obj)
    if not replace and errors == 'ignore':
        for (_, obj), w in zip(group_iterator, ws):
            yield obj.sample(
                n=n, frac=frac, replace=replace, weights=w, random_state=random_state
            ) if len(obj) > n else obj
    else:
        for (_, obj), w in zip(group_iterator, ws):
            yield obj.sample(
                n=n, frac=frac, replace=replace, weights=w, random_state=random_state
            )


class GroupBySampleILoc(DataFrameOperand, DataFrameOperandMixin):
    _op_code_ = opcodes.GROUPBY_SAMPLE_ILOC
    _op_module_ = 'dataframe.groupby'

    _groupby_params = DictField('groupby_params')
    _size = Int64Field('size')
    _frac = Float32Field('frac')
    _replace = BoolField('replace')
    _weights = KeyField('weights')
    _seed = Int32Field('seed')
    _random_state = RandomStateField('random_state')
    _errors = StringField('errors')

    _random_col_id = Int32Field('random_col_id')

    # for chunks
    # num of instances for chunks
    _left_iloc_bound = Int64Field('left_iloc_bound')

    def __init__(self, groupby_params=None, size=None, frac=None, replace=None,
                 weights=None, random_state=None, seed=None, errors=None,
                 left_iloc_bound=None, random_col_id=None, **kw):
        super().__init__(_groupby_params=groupby_params, _size=size, _frac=frac,
                         _seed=seed, _replace=replace, _weights=weights,
                         _random_state=random_state, _errors=errors,
                         _left_iloc_bound=left_iloc_bound,
                         _random_col_id=random_col_id, **kw)
        if self._random_col_id is None:
            self._random_col_id = random.randint(10000, 99999)

    @property
    def groupby_params(self):
        return self._groupby_params

    @property
    def size(self):
        return self._size

    @property
    def frac(self):
        return self._frac

    @property
    def replace(self):
        return self._replace

    @property
    def weights(self):
        return self._weights

    @property
    def seed(self):
        return self._seed

    @property
    def random_state(self):
        if self._random_state is None:
            self._random_state = np.random.RandomState(self.seed)
        return self._random_state

    @property
    def errors(self):
        return self._errors

    @property
    def left_iloc_bound(self):
        return self._left_iloc_bound

    @property
    def random_col_id(self):
        return self._random_col_id

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        input_iter = iter(inputs)
        next(input_iter)
        if isinstance(self.weights, ENTITY_TYPE):
            self._weights = next(input_iter)

    def __call__(self, df):
        self._output_types = [OutputType.tensor]
        inp_tileables = [df]
        if self.weights is not None:
            inp_tileables.append(self.weights)
        return self.new_tileable(inp_tileables, dtype=np.dtype(np.int_),
                                 shape=(np.nan,))

    @classmethod
    def tile(cls, op: 'GroupBySampleILoc'):
        in_df = op.inputs[0]
        out_tensor = op.outputs[0]
        iloc_col_header = _ILOC_COL_HEADER + str(op.random_col_id)
        weight_col_header = _WEIGHT_COL_HEADER + str(op.random_col_id)

        if has_unknown_shape(in_df):
            yield

        if op.weights is None:
            weights_iter = itertools.repeat(None)
        else:
            weights_iter = iter(op.weights.chunks)

        if isinstance(op.groupby_params['by'], list):
            map_cols = list(op.groupby_params['by'])
        else:  # pragma: no cover
            map_cols = []

        dtypes = in_df.dtypes.copy()
        dtypes.at[iloc_col_header] = np.dtype(np.int_)
        map_cols.append(iloc_col_header)
        if op.weights is not None:
            dtypes.at[weight_col_header] = op.weights.dtype
            map_cols.append(weight_col_header)

        new_dtypes = dtypes[map_cols]
        new_columns_value = parse_index(new_dtypes.index, store_data=True)

        map_chunks = []
        left_ilocs = np.array((0,) + in_df.nsplits[0]).cumsum()
        for inp_chunk, weight_chunk in zip(in_df.chunks, weights_iter):
            new_op = op.copy().reset_key()
            new_op._left_iloc_bound = int(left_ilocs[inp_chunk.index[0]])
            new_op.stage = OperandStage.map
            new_op._output_types = [OutputType.dataframe]

            inp_chunks = [inp_chunk]
            if weight_chunk is not None:
                inp_chunks.append(weight_chunk)
            params = inp_chunk.params
            params.update(dict(
                dtypes=new_dtypes, columns_value=new_columns_value,
                shape=(inp_chunk.shape[0], len(new_dtypes)),
                index=inp_chunk.index,
            ))
            map_chunks.append(new_op.new_chunk(inp_chunks, **params))

        new_op = op.copy().reset_key()
        new_op._output_types = [OutputType.dataframe]
        params = in_df.params
        params.update(dict(
            chunks=map_chunks,
            nsplits=(in_df.nsplits[0], (len(new_dtypes),)),
            dtypes=new_dtypes, columns_value=new_columns_value,
            shape=(in_df.shape[0], len(new_dtypes)),
        ))
        map_df = new_op.new_tileable(op.inputs, **params)

        groupby_params = op.groupby_params.copy()
        groupby_params.pop('selection', None)
        grouped = yield from recursive_tile(
            map_df.groupby(**groupby_params))

        result_chunks = []
        seeds = gen_random_seeds(len(grouped.chunks), op.random_state)
        for group_chunk, seed in zip(grouped.chunks, seeds):
            new_op = op.copy().reset_key()
            new_op.stage = OperandStage.reduce
            new_op._weights = None
            new_op._random_state = None
            new_op._seed = seed

            result_chunks.append(new_op.new_chunk(
                [group_chunk], shape=(np.nan,), index=(group_chunk.index[0],),
                dtype=out_tensor.dtype
            ))

        new_op = op.copy().reset_key()
        params = out_tensor.params
        params.update(dict(
            chunks=result_chunks, nsplits=((np.nan,) * len(result_chunks),)
        ))
        return new_op.new_tileables(op.inputs, **params)

    @classmethod
    def execute(cls, ctx, op: 'GroupBySampleILoc'):
        in_data = ctx[op.inputs[0].key]
        iloc_col = _ILOC_COL_HEADER + str(op.random_col_id)
        weight_col = _WEIGHT_COL_HEADER + str(op.random_col_id)
        if op.stage == OperandStage.map:
            if op.weights is not None:
                ret = pd.DataFrame({
                    iloc_col: np.arange(op.left_iloc_bound, op.left_iloc_bound + len(in_data)),
                    weight_col: ctx[op.weights.key],
                }, index=in_data.index)
            else:
                ret = pd.DataFrame({
                    iloc_col: np.arange(op.left_iloc_bound, op.left_iloc_bound + len(in_data)),
                }, index=in_data.index)

            if isinstance(op.groupby_params['by'], list):
                ret = pd.concat([in_data[op.groupby_params['by']], ret], axis=1)

            ctx[op.outputs[0].key] = ret
        else:
            if weight_col not in in_data.obj.columns:
                weight_col = None

            if len(in_data.obj) == 0 or in_data.ngroups == 0:
                ctx[op.outputs[0].key] = np.array([], dtype=np.int_)
            else:
                ctx[op.outputs[0].key] = np.concatenate([
                    sample_pd[iloc_col].to_numpy()
                    for sample_pd in _sample_groupby_iter(
                        in_data, in_data.obj.index, n=op.size, frac=op.frac, replace=op.replace,
                        weights=weight_col, random_state=op.random_state, errors=op.errors
                    )
                ])


class GroupBySample(MapReduceOperand, DataFrameOperandMixin):
    _op_code_ = opcodes.RAND_SAMPLE
    _op_module_ = 'dataframe.groupby'

    _groupby_params = DictField('groupby_params')
    _size = Int64Field('size')
    _frac = Float32Field('frac')
    _replace = BoolField('replace')
    _weights = KeyField('weights')
    _seed = Int32Field('seed')
    _random_state = RandomStateField('random_state')
    _errors = StringField('errors')

    # for chunks
    # num of instances for chunks
    _input_nsplits = NDArrayField('input_nsplits')

    def __init__(self, groupby_params=None, size=None, frac=None, replace=None,
                 weights=None, random_state=None, seed=None,
                 errors=None, input_nsplits=None, **kw):
        super().__init__(_groupby_params=groupby_params, _size=size, _frac=frac,
                         _seed=seed, _replace=replace, _weights=weights,
                         _random_state=random_state, _errors=errors,
                         _input_nsplits=input_nsplits, **kw)

    @property
    def groupby_params(self):
        return self._groupby_params

    @property
    def size(self):
        return self._size

    @property
    def frac(self):
        return self._frac

    @property
    def replace(self):
        return self._replace

    @property
    def weights(self):
        return self._weights

    @property
    def seed(self):
        return self._seed

    @property
    def random_state(self):
        return self._random_state

    @property
    def errors(self):
        return self._errors

    @property
    def input_nsplits(self):
        return self._input_nsplits

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        input_iter = iter(inputs)
        next(input_iter)
        if isinstance(self.weights, ENTITY_TYPE):
            self._weights = next(input_iter)

    def __call__(self, groupby):
        df = groupby
        while df.op.output_types[0] not in (OutputType.dataframe, OutputType.series):
            df = df.inputs[0]

        selection = groupby.op.groupby_params.pop('selection', None)
        if df.ndim > 1 and selection:
            if isinstance(selection, tuple) and selection not in df.dtypes:
                selection = list(selection)
            result_df = df[selection]
        else:
            result_df = df

        params = result_df.params
        params['shape'] = (np.nan,) if result_df.ndim == 1 else (np.nan, result_df.shape[-1])
        params['index_value'] = parse_index(result_df.index_value.to_pandas()[:0])

        input_dfs = [df]
        if isinstance(self.weights, ENTITY_TYPE):
            input_dfs.append(self.weights)

        self._output_types = get_output_types(result_df)
        return self.new_tileable(input_dfs, **params)

    @classmethod
    def _tile_one_chunk(cls, op: "GroupBySample", in_df, weights):
        out = op.outputs[0]

        input_dfs = [in_df]
        if isinstance(weights, ENTITY_TYPE):
            input_dfs.append(weights)

        params = out.params
        chunk_op = op.copy().reset_key()
        if isinstance(weights, ENTITY_TYPE):
            chunk_op._weights = weights
        params['index'] = (0,) * out.ndim
        chunk = chunk_op.new_chunk([c.chunks[0] for c in input_dfs], **params)

        df_op = op.copy().reset_key()
        return df_op.new_tileables(
            input_dfs, chunks=[chunk], nsplits=((s,) for s in out.shape), **params)

    @classmethod
    def _tile_distributed(cls, op: "GroupBySample", in_df, weights):
        out_df = op.outputs[0]
        if has_unknown_shape(in_df):
            yield

        sample_iloc_op = GroupBySampleILoc(
            groupby_params=op.groupby_params, size=op.size, frac=op.frac, replace=op.replace,
            weights=weights, random_state=op.random_state, errors=op.errors, seed=None,
            left_iloc_bound=None
        )
        sampled_iloc = yield from recursive_tile(sample_iloc_op(in_df))

        map_chunks = []
        for c in sampled_iloc.chunks:
            new_op = op.copy().reset_key()
            new_op.stage = OperandStage.map
            new_op._weights = None
            new_op._output_types = [OutputType.tensor]
            new_op._input_nsplits = np.array(in_df.nsplits[0])

            map_chunks.append(new_op.new_chunk(
                [c], dtype=sampled_iloc.dtype, shape=(np.nan,), index=c.index))

        proxy_chunk = TensorShuffleProxy(dtype=sampled_iloc.dtype).new_chunk(
            map_chunks, shape=())

        reduce_chunks = []
        for src_chunk in in_df.chunks:
            new_op = op.copy().reset_key()
            new_op._weights = None
            new_op._output_types = [OutputType.tensor]
            new_op.stage = OperandStage.reduce
            new_op.reducer_index = (src_chunk.index[0],)
            new_op._input_nsplits = np.array(in_df.nsplits[0])

            reduce_chunks.append(new_op.new_chunk(
                [proxy_chunk], index=src_chunk.index, dtype=sampled_iloc.dtype, shape=(np.nan,)))

        combine_chunks = []
        for src_chunk, reduce_chunk in zip(in_df.chunks, reduce_chunks):
            new_op = op.copy().reset_key()
            new_op.stage = OperandStage.combine
            new_op._weights = None

            params = out_df.params
            if out_df.ndim == 2:
                params.update(dict(
                    index=src_chunk.index, dtypes=out_df.dtypes,
                    shape=(np.nan, out_df.shape[1]),
                    columns_value=out_df.columns_value,
                ))
            else:
                params.update(dict(
                    index=(src_chunk.index[0],), dtype=out_df.dtype,
                    shape=(np.nan,), name=out_df.name,
                ))
            combine_chunks.append(new_op.new_chunk([src_chunk, reduce_chunk], **params))

        new_op = op.copy().reset_key()
        if out_df.ndim == 2:
            new_nsplits = ((np.nan,) * in_df.chunk_shape[0], (out_df.shape[1],))
        else:
            new_nsplits = ((np.nan,) * in_df.chunk_shape[0],)
        return new_op.new_tileables(out_df.inputs, chunks=combine_chunks, nsplits=new_nsplits,
                                    **out_df.params)

    @classmethod
    def tile(cls, op: 'GroupBySample'):
        in_df = op.inputs[0]
        if in_df.ndim == 2:
            in_df = yield from recursive_tile(
                in_df.rechunk({1: (in_df.shape[1],)}))

        weights = op.weights
        if isinstance(weights, ENTITY_TYPE):
            weights = yield from recursive_tile(
                weights.rechunk({0: in_df.nsplits[0]}))

        if len(in_df.chunks) == 1:
            return cls._tile_one_chunk(op, in_df, weights)
        return (yield from cls._tile_distributed(op, in_df, weights))

    @classmethod
    def execute(cls, ctx, op: 'GroupBySample'):
        out_df = op.outputs[0]

        if op.stage == OperandStage.map:
            in_data = ctx[op.inputs[0].key]
            in_data = np.sort(in_data)
            input_nsplits = np.copy(op.input_nsplits).tolist()
            pos_array = np.cumsum([0] + input_nsplits)
            poses = np.searchsorted(in_data, pos_array).tolist()
            for idx, (left, right) in enumerate(zip(poses, poses[1:])):
                ctx[op.outputs[0].key, (idx,)] = in_data[left:right]
        elif op.stage == OperandStage.reduce:
            in_indexes = list(op.iter_mapper_data(ctx))
            idx = np.sort(np.concatenate(in_indexes))
            if op.outputs[0].index[0] > 0:
                acc_nsplits = np.cumsum(op.input_nsplits)
                idx -= acc_nsplits[op.outputs[0].index[0] - 1]
            ctx[op.outputs[0].key] = idx
        elif op.stage == OperandStage.combine:
            in_data = ctx[op.inputs[0].key]
            idx = ctx[op.inputs[1].key]
            selection = op.groupby_params.get('selection')
            if selection:
                in_data = in_data[selection]
            ctx[op.outputs[0].key] = in_data.iloc[idx]
        else:
            in_data = ctx[op.inputs[0].key]
            weights = op.weights
            if isinstance(weights, ENTITY_TYPE):
                weights = ctx[weights.key]
            params = op.groupby_params.copy()
            selection = params.pop('selection', None)

            grouped = in_data.groupby(**params)
            if selection is not None:
                grouped = grouped[selection]

            result = pd.concat([
                sample_df for sample_df in _sample_groupby_iter(
                    grouped, in_data.index, n=op.size, frac=op.frac, replace=op.replace,
                    weights=weights, random_state=op.random_state, errors=op.errors,
                )
            ])
            ctx[out_df.key] = result


def groupby_sample(groupby, n=None, frac=None, replace=False, weights=None,
                   random_state=None, errors='ignore'):
    """
    Return a random sample of items from each group.

    You can use `random_state` for reproducibility.

    Parameters
    ----------
    n : int, optional
        Number of items to return for each group. Cannot be used with
        `frac` and must be no larger than the smallest group unless
        `replace` is True. Default is one if `frac` is None.
    frac : float, optional
        Fraction of items to return. Cannot be used with `n`.
    replace : bool, default False
        Allow or disallow sampling of the same row more than once.
    weights : list-like, optional
        Default None results in equal probability weighting.
        If passed a list-like then values must have the same length as
        the underlying DataFrame or Series object and will be used as
        sampling probabilities after normalization within each group.
        Values must be non-negative with at least one positive element
        within each group.
    random_state : int, array-like, BitGenerator, np.random.RandomState, optional
        If int, array-like, or BitGenerator (NumPy>=1.17), seed for
        random number generator
        If np.random.RandomState, use as numpy RandomState object.
    errors : {'ignore', 'raise'}, default 'ignore'
        If ignore, errors will not be raised when `replace` is False
        and size of some group is less than `n`.

    Returns
    -------
    Series or DataFrame
        A new object of same type as caller containing items randomly
        sampled within each group from the caller object.

    See Also
    --------
    DataFrame.sample: Generate random samples from a DataFrame object.
    numpy.random.choice: Generate a random sample from a given 1-D numpy
        array.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> df = md.DataFrame(
    ...     {"a": ["red"] * 2 + ["blue"] * 2 + ["black"] * 2, "b": range(6)}
    ... )
    >>> df.execute()
           a  b
    0    red  0
    1    red  1
    2   blue  2
    3   blue  3
    4  black  4
    5  black  5

    Select one row at random for each distinct value in column a. The
    `random_state` argument can be used to guarantee reproducibility:

    >>> df.groupby("a").sample(n=1, random_state=1).execute()
           a  b
    4  black  4
    2   blue  2
    1    red  1

    Set `frac` to sample fixed proportions rather than counts:

    >>> df.groupby("a")["b"].sample(frac=0.5, random_state=2).execute()
    5    5
    2    2
    0    0
    Name: b, dtype: int64

    Control sample probabilities within groups by setting weights:

    >>> df.groupby("a").sample(
    ...     n=1,
    ...     weights=[1, 1, 1, 0, 0, 1],
    ...     random_state=1,
    ... ).execute()
           a  b
    5  black  5
    2   blue  2
    0    red  0
    """
    groupby_params = groupby.op.groupby_params.copy()
    groupby_params.pop('as_index', None)

    if weights is not None and not isinstance(weights, ENTITY_TYPE):
        weights = asseries(weights)

    n = 1 if n is None and frac is None else n
    rs = copy.deepcopy(
        random_state.to_numpy() if hasattr(random_state, 'to_numpy') else random_state)
    op = GroupBySample(size=n, frac=frac, replace=replace, weights=weights,
                       random_state=rs, groupby_params=groupby_params,
                       errors=errors)
    return op(groupby)
