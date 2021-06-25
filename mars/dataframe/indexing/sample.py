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

import numpy as np

from ... import opcodes
from ...core import ENTITY_TYPE, get_output_types, recursive_tile
from ...serialization.serializables import BoolField, AnyField, Int8Field, Int64Field, Float64Field, \
    KeyField
from ...tensor import searchsorted
from ...tensor.base import TensorMapChunk
from ...tensor.merge import TensorConcatenate
from ...tensor.random import RandomState as TensorRandomState, RandomStateField
from ...tensor.utils import normalize_chunk_sizes, gen_random_seeds
from ...utils import has_unknown_shape, ceildiv
from ..operands import DataFrameOperandMixin, DataFrameOperand
from ..utils import validate_axis, parse_index


class DataFrameSample(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.RAND_SAMPLE

    _size = Int64Field('size')
    _frac = Float64Field('frac')
    _replace = BoolField('replace')
    _weights = AnyField('weights')
    _axis = Int8Field('axis')
    _seed = Int64Field('seed')
    _random_state = RandomStateField('random_state')
    _always_multinomial = BoolField('always_multinomial')

    # for chunks
    # num of instances for chunks
    _chunk_samples = KeyField('chunk_samples')

    def __init__(self, size=None, frac=None, replace=None, weights=None, seed=None,
                 axis=None, random_state=None, always_multinomial=None,
                 chunk_samples=None, **kw):
        super().__init__(_size=size, _frac=frac, _replace=replace, _weights=weights,
                         _seed=seed, _axis=axis, _random_state=random_state,
                         _always_multinomial=always_multinomial,
                         _chunk_samples=chunk_samples, **kw)

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
    def axis(self):
        return self._axis

    @property
    def random_state(self):
        if self._random_state is None:
            self._random_state = np.random.RandomState(self.seed)
        return self._random_state

    @property
    def always_multinomial(self):
        return self._always_multinomial

    @property
    def chunk_samples(self):
        return self._chunk_samples

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        it = iter(inputs)
        next(it)
        if isinstance(self.weights, ENTITY_TYPE):
            self._weights = next(it)
        if isinstance(self.chunk_samples, ENTITY_TYPE):
            self._chunk_samples = next(it)

    def __call__(self, df):
        params = df.params
        new_shape = list(df.shape)

        if self.frac is not None and not np.isnan(df.shape[self.axis]):
            self._size = int(self.frac * df.shape[self.axis])
            self._frac = None

        if self.size is not None:
            new_shape[self.axis] = self.size
        params['shape'] = tuple(new_shape)
        params['index_value'] = parse_index(df.index_value.to_pandas()[:0])

        input_dfs = [df]
        if isinstance(self.weights, ENTITY_TYPE):
            input_dfs.append(self.weights)

        self._output_types = get_output_types(df)
        return self.new_tileable(input_dfs, **params)

    @classmethod
    def _tile_one_chunk(cls, op: "DataFrameSample", in_df, weights):
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
    def _tile_multinomial(cls, op: "DataFrameSample", in_df, weights):
        out_data = op.outputs[0]
        input_dfs = [in_df]
        size = op.size

        weight_chunks = itertools.repeat(None)
        if isinstance(op.weights, ENTITY_TYPE):
            input_dfs.append(weights)
            weight_chunks = weights.chunks

        chunks = []
        new_nsplits = list(in_df.nsplits)
        rs = op.random_state
        seeds = gen_random_seeds(len(in_df.chunks), op.random_state)
        if weights is None:
            # weights is None, use nsplits to sample num of instances for each chunk
            probs = np.array(in_df.nsplits[op.axis])
            probs = 1.0 * probs / probs.sum()
            chunk_sizes = rs.multinomial(size, probs)
            new_nsplits[op.axis] = tuple(int(s) for s in chunk_sizes if s > 0)

            chunk_idx = 0
            for data_chunk, chunk_size, seed in zip(in_df.chunks, chunk_sizes, seeds):
                if chunk_size == 0:
                    continue

                chunk_op = op.copy().reset_key()
                chunk_op._random_state = None
                chunk_op._seed = seed
                chunk_op._size = int(chunk_size)

                params = data_chunk.params
                params['index_value'] = parse_index(
                    params['index_value'].to_pandas()[:0])
                new_shape = list(data_chunk.shape)
                new_shape[op.axis] = int(chunk_size)
                params['shape'] = tuple(new_shape)

                idx_list = [0] * data_chunk.ndim
                idx_list[op.axis] = chunk_idx
                params['index'] = tuple(idx_list)

                chunks.append(chunk_op.new_chunk([data_chunk], **params))
                chunk_idx += 1
        else:
            mn_seed = gen_random_seeds(1, op.random_state)[0]

            # weights is specified, use weights to sample num of instances for each chunk
            chunk_weights = yield from recursive_tile(
                weights.to_tensor().map_chunk(lambda x: x.sum(keepdims=True)))
            chunk_weights_chunk = TensorConcatenate(dtype=chunk_weights.dtype).new_chunk(
                chunk_weights.chunks, shape=(len(chunk_weights.chunks),), index=(0,))
            chunk_samples = TensorMapChunk(
                func=lambda x: np.random.RandomState(mn_seed).multinomial(size, x / x.sum())
            ).new_chunk(
                [chunk_weights_chunk], shape=(len(chunk_weights.chunks),), index=(0,)
            )
            new_nsplits[op.axis] = (np.nan,) * len(chunk_weights.chunks)
            for chunk_idx, (data_chunk, weight_chunk, seed) \
                    in enumerate(zip(in_df.chunks, weight_chunks, seeds)):
                input_chunks = [data_chunk]

                chunk_op = op.copy().reset_key()
                chunk_op._size = None
                chunk_op._random_state = None
                chunk_op._seed = seed
                chunk_op._chunk_samples = chunk_samples
                if weight_chunk is not None:
                    chunk_op._weights = weight_chunk
                    input_chunks.append(weight_chunk)

                params = data_chunk.params
                params['index_value'] = parse_index(
                    params['index_value'].to_pandas()[:0])
                new_shape = list(data_chunk.shape)
                new_shape[op.axis] = np.nan
                params['shape'] = tuple(new_shape)

                idx_list = [0] * data_chunk.ndim
                idx_list[op.axis] = chunk_idx
                params['index'] = tuple(idx_list)

                chunks.append(chunk_op.new_chunk(
                    input_chunks + [chunk_samples], **params))

        params = out_data.params
        new_shape = list(in_df.shape)
        new_shape[op.axis] = size
        params['shape'] = tuple(new_shape)

        df_op = op.copy().reset_key()
        return df_op.new_tileables(input_dfs, chunks=chunks, nsplits=tuple(new_nsplits), **params)

    @classmethod
    def _tile_reservoirs(cls, op: "DataFrameSample", in_df, weights):
        out_data = op.outputs[0]
        input_dfs = [in_df]
        size = op.size

        weight_chunks = itertools.repeat(None)
        if isinstance(weights, ENTITY_TYPE):
            input_dfs.append(weights)
            weight_chunks = weights.chunks

        if any(cs < size for cs in in_df.nsplits[op.axis]):
            # make sure all chunk > m
            n_records = in_df.shape[op.axis]
            n_chunk = min(max(ceildiv(n_records, size), 1), in_df.chunk_shape[0])
            chunk_size = ceildiv(in_df.shape[op.axis], n_chunk)
            chunk_sizes = list(normalize_chunk_sizes(n_records, chunk_size)[0])
            if chunk_sizes[-1] < size and len(chunk_sizes) > 1:
                # the last chunk may still less than m
                # merge it into previous one
                chunk_sizes[-2] += chunk_sizes[-1]
                chunk_sizes = chunk_sizes[:-1]
            in_df = yield from recursive_tile(
                in_df.rechunk({0: tuple(chunk_sizes)}))
            if isinstance(weights, ENTITY_TYPE):
                weights = yield from recursive_tile(
                    weights.rechunk({0: tuple(chunk_sizes)}))
            if len(chunk_sizes) == 1:
                return cls._tile_one_chunk(op, in_df, weights)

        # for each chunk in a, do regular sampling
        sampled_chunks = []
        seeds = gen_random_seeds(len(in_df.chunks), op.random_state)
        for data_chunk, weights_chunk, seed in zip(in_df.chunks, weight_chunks, seeds):
            input_chunks = [data_chunk]

            chunk_op = op.copy().reset_key()
            chunk_op._random_state = None
            chunk_op._seed = seed
            if isinstance(op.weights, ENTITY_TYPE):
                input_chunks.append(weights_chunk)
                chunk_op._weights = weights_chunk

            params = data_chunk.params
            new_shape = list(data_chunk.shape)
            new_shape[op.axis] = size
            params['shape'] = tuple(new_shape)
            sampled_chunks.append(chunk_op.new_chunk(input_chunks, **params))

        # generate a random variable for samples in every chunk
        state = TensorRandomState.from_numpy(op.random_state)
        indices = state.rand(size)

        if weights is None:
            # weights not specified, use nsplits to calculate cumulative probability
            # to distribute samples in each chunk
            cum_offsets = np.cumsum(in_df.nsplits[op.axis])
            cum_offsets = cum_offsets * 1.0 / cum_offsets[-1]
        else:
            # weights specified, use weights to calculate cumulative probability
            # to distribute samples in each chunk
            chunk_weights = yield from recursive_tile(
                weights.to_tensor().map_chunk(lambda x: x.sum(keepdims=True)))
            chunk_weights_chunk = TensorConcatenate(dtype=chunk_weights.dtype).new_chunk(
                chunk_weights.chunks, shape=(len(chunk_weights.chunks),), index=(0,))

            cum_chunk = TensorMapChunk(func=lambda x: (x / x.sum()).cumsum()).new_chunk(
                [chunk_weights_chunk], shape=(len(chunk_weights.chunks),), index=(0,)
            )
            cum_offsets = TensorMapChunk(func=cum_chunk.op.func).new_tensor(
                [weights], chunks=[cum_chunk], nsplits=((s,) for s in cum_chunk.shape),
                **cum_chunk.params)

        index_chunks = []
        # seek which chunk the final sample will select
        chunk_sel = yield from recursive_tile(
            searchsorted(cum_offsets, indices, side='right'))
        # for every chunk, select samples with bool indexing
        for idx, sampled_chunk in enumerate(sampled_chunks):
            chunk_index = chunk_sel.map_chunk(func=lambda x, i: x == i, args=(idx,),
                                              elementwise=True, dtype=bool)
            sampled_df_op = sampled_chunk.op.copy().reset_key()
            sampled_chunk._index = (0,) * sampled_chunk.ndim
            sampled_df = sampled_df_op.new_tileable(
                input_dfs, chunks=[sampled_chunk], nsplits=((s,) for s in sampled_chunk.shape),
                **sampled_chunk.params)
            index_chunk = (yield from recursive_tile(
                sampled_df.iloc[chunk_index])).chunks[0]

            chunk_idx = [0] * sampled_chunk.ndim
            chunk_idx[op.axis] = idx
            index_chunk._index = tuple(chunk_idx)
            index_chunks.append(index_chunk)

        params = out_data.params
        new_shape = list(in_df.shape)
        new_shape[op.axis] = size
        params['shape'] = tuple(new_shape)

        new_nsplits = list(in_df.nsplits)
        new_nsplits[op.axis] = (np.nan,) * len(index_chunks)

        df_op = op.copy().reset_key()
        return df_op.new_tileables(
            input_dfs, chunks=index_chunks, nsplits=tuple(new_nsplits), **params)

    @classmethod
    def tile(cls, op: "DataFrameSample"):
        if has_unknown_shape(*op.inputs):
            yield

        in_df = op.inputs[0]
        if in_df.ndim == 2:
            in_df = yield from recursive_tile(
                in_df.rechunk({(1 - op.axis): (in_df.shape[1 - op.axis],)}))

        if op.size is None:
            op._size = int(op.frac * in_df.shape[op.axis])

        weights = op.weights
        if isinstance(weights, ENTITY_TYPE):
            weights = yield from recursive_tile(
                weights.rechunk({0: in_df.nsplits[op.axis]}))
        elif in_df.ndim > 1 and weights in in_df.dtypes.index:
            weights = yield from recursive_tile(in_df[weights])

        if len(in_df.chunks) == 1:
            return cls._tile_one_chunk(op, in_df, weights)

        if op.replace or op.always_multinomial:
            return (yield from cls._tile_multinomial(op, in_df, weights))
        else:
            return (yield from cls._tile_reservoirs(op, in_df, weights))

    @classmethod
    def execute(cls, ctx, op: "DataFrameSample"):
        in_data = ctx[op.inputs[0].key]
        weights = op.weights
        if isinstance(weights, ENTITY_TYPE):
            weights = ctx[weights.key]

        size = op.size
        chunk_samples = op.chunk_samples
        if isinstance(chunk_samples, ENTITY_TYPE):
            chunk_samples = ctx[chunk_samples.key]
        if chunk_samples is not None:
            size = chunk_samples[op.inputs[0].index[op.axis]]

        ctx[op.outputs[0].key] = in_data.sample(
            n=size, frac=op.frac, replace=op.replace, weights=weights,
            random_state=op.random_state, axis=op.axis
        )


def sample(df_or_series, n=None, frac=None, replace=False, weights=None, random_state=None,
           axis=None, always_multinomial=False):
    """
    Return a random sample of items from an axis of object.

    You can use `random_state` for reproducibility.

    Parameters
    ----------
    n : int, optional
        Number of items from axis to return. Cannot be used with `frac`.
        Default = 1 if `frac` = None.
    frac : float, optional
        Fraction of axis items to return. Cannot be used with `n`.
    replace : bool, default False
        Allow or disallow sampling of the same row more than once.
    weights : str or ndarray-like, optional
        Default 'None' results in equal probability weighting.
        If passed a Series, will align with target object on index. Index
        values in weights not found in sampled object will be ignored and
        index values in sampled object not in weights will be assigned
        weights of zero.
        If called on a DataFrame, will accept the name of a column
        when axis = 0.
        Unless weights are a Series, weights must be same length as axis
        being sampled.
        If weights do not sum to 1, they will be normalized to sum to 1.
        Missing values in the weights column will be treated as zero.
        Infinite values not allowed.
    random_state : int, array-like, BitGenerator, np.random.RandomState, optional
        If int, array-like, or BitGenerator (NumPy>=1.17), seed for
        random number generator
        If np.random.RandomState, use as numpy RandomState object.
    axis : {0 or ‘index’, 1 or ‘columns’, None}, default None
        Axis to sample. Accepts axis number or name. Default is stat axis
        for given data type (0 for Series and DataFrames).
    always_multinomial : bool, default False
        If True, always treat distribution of sample counts between data chunks
        as multinomial distribution. This will accelerate sampling when data
        is huge, but may affect randomness of samples when number of instances
        is not very large.

    Returns
    -------
    Series or DataFrame
        A new object of same type as caller containing `n` items randomly
        sampled from the caller object.

    See Also
    --------
    DataFrameGroupBy.sample: Generates random samples from each group of a
        DataFrame object.
    SeriesGroupBy.sample: Generates random samples from each group of a
        Series object.
    numpy.random.choice: Generates a random sample from a given 1-D numpy
        array.

    Notes
    -----
    If `frac` > 1, `replacement` should be set to `True`.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> df = md.DataFrame({'num_legs': [2, 4, 8, 0],
    ...                    'num_wings': [2, 0, 0, 0],
    ...                    'num_specimen_seen': [10, 2, 1, 8]},
    ...                   index=['falcon', 'dog', 'spider', 'fish'])
    >>> df.execute()
            num_legs  num_wings  num_specimen_seen
    falcon         2          2                 10
    dog            4          0                  2
    spider         8          0                  1
    fish           0          0                  8

    Extract 3 random elements from the ``Series`` ``df['num_legs']``:
    Note that we use `random_state` to ensure the reproducibility of
    the examples.

    >>> df['num_legs'].sample(n=3, random_state=1).execute()
    fish      0
    spider    8
    falcon    2
    Name: num_legs, dtype: int64

    A random 50% sample of the ``DataFrame`` with replacement:

    >>> df.sample(frac=0.5, replace=True, random_state=1).execute()
          num_legs  num_wings  num_specimen_seen
    dog          4          0                  2
    fish         0          0                  8

    An upsample sample of the ``DataFrame`` with replacement:
    Note that `replace` parameter has to be `True` for `frac` parameter > 1.

    >>> df.sample(frac=2, replace=True, random_state=1).execute()
            num_legs  num_wings  num_specimen_seen
    dog            4          0                  2
    fish           0          0                  8
    falcon         2          2                 10
    falcon         2          2                 10
    fish           0          0                  8
    dog            4          0                  2
    fish           0          0                  8
    dog            4          0                  2

    Using a DataFrame column as weights. Rows with larger value in the
    `num_specimen_seen` column are more likely to be sampled.

    >>> df.sample(n=2, weights='num_specimen_seen', random_state=1).execute()
            num_legs  num_wings  num_specimen_seen
    falcon         2          2                 10
    fish           0          0                  8

    """
    axis = validate_axis(axis or 0, df_or_series)
    if axis == 1:
        raise NotImplementedError('Currently cannot sample over columns')
    rs = copy.deepcopy(
        random_state.to_numpy() if hasattr(random_state, 'to_numpy') else random_state)
    op = DataFrameSample(size=n, frac=frac, replace=replace, weights=weights,
                         random_state=rs, axis=axis, always_multinomial=always_multinomial)
    return op(df_or_series)
