# Copyright 1999-2020 Alibaba Group Holding Ltd.
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
from pandas.api.types import is_list_like

from ... import opcodes
from ...config import options
from ...operands import OperandStage
from ...serialize import KeyField, AnyField, BoolField, StringField, Int32Field
from ...tiles import TilesError
from ...utils import check_chunks_unknown_shape, ceildiv, get_shuffle_input_keys_idxes, lazy_import
from ..initializer import DataFrame as asdataframe
from ..operands import DataFrameMapReduceOperand, DataFrameOperandMixin, \
    DataFrameShuffleProxy, ObjectType
from ..merge import DataFrameConcat
from ..utils import parse_index, hash_dataframe_on, gen_unknown_index_value, standardize_range_index

cudf = lazy_import('cudf', globals=globals())


class DataFrameDropDuplicates(DataFrameMapReduceOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.DROP_DUPLICATES

    _input = KeyField('input')
    _subset = AnyField('subset')
    _keep = AnyField('keep')
    _ignore_index = BoolField('ignore_index')
    _method = StringField('method')
    # subset chunk, used for method 'subset_tree'
    _subset_chunk = KeyField('subset_chunk')
    # shuffle phase, used in shuffle method
    _shuffle_size = Int32Field('shuffle_size')
    _shuffle_phase = StringField('shuffle_phase')

    def __init__(self, subset=None, keep=None, ignore_index=None,
                 object_type=None, method=None, subset_chunk=None,
                 shuffle_size=None, shuffle_phase=None, shuffle_key=None, **kw):
        super().__init__(_subset=subset, _keep=keep, _ignore_index=ignore_index,
                         _object_type=object_type, _method=method,
                         _subset_chunk=subset_chunk,
                         _shuffle_size=shuffle_size,
                         _shuffle_phase=shuffle_phase,
                         _shuffle_key=shuffle_key, **kw)

    @property
    def input(self):
        return self._input

    @property
    def subset(self):
        return self._subset

    @property
    def keep(self):
        return self._keep

    @property
    def ignore_index(self):
        return self._ignore_index

    @property
    def method(self):
        return self._method

    @property
    def subset_chunk(self):
        return self._subset_chunk

    @property
    def shuffle_size(self):
        return self._shuffle_size

    @property
    def shuffle_phase(self):
        return self._shuffle_phase

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]
        if self._subset_chunk is not None:
            self._subset_chunk = self._inputs[1]

    def __call__(self, inp, inplace=False):
        self._object_type = inp.op.object_type
        params = inp.params
        if self._ignore_index:
            params['index_value'] = parse_index(pd.RangeIndex(-1))
        else:
            params['index_value'] = gen_unknown_index_value(
                params['index_value'], self._keep, self._subset, type(self).__name__)
        shape_list = list(params['shape'])
        shape_list[0] = np.nan
        params['shape'] = tuple(shape_list)

        ret = self.new_tileable([inp], kws=[params])
        if inplace:
            inp.data = ret.data
        return ret

    @classmethod
    def _tile_one_chunk(cls, op: "DataFrameDropDuplicates"):
        inp = op.input
        out = op.outputs[0]

        chunk_op = op.copy().reset_key()
        chunk_op._method = None
        chunk_params = out.params
        chunk_params['index'] = (0,) * out.ndim
        chunk_shape_list = list(chunk_params['shape'])
        chunk_shape_list[0] = np.nan
        chunk_params['shape'] = tuple(chunk_shape_list)
        in_chunk = inp.chunks[0]
        chunk_params['index_value'] = gen_unknown_index_value(
            chunk_params['index_value'], in_chunk)
        chunk = chunk_op.new_chunk([in_chunk], kws=[chunk_params])

        params = out.params
        params['chunks'] = [chunk]
        params['nsplits'] = tuple((s,) for s in chunk.shape)
        new_op = op.copy()
        return new_op.new_tileables([inp], kws=[params])

    @classmethod
    def _gen_map_chunks(cls, op: "DataFrameDropDuplicates", inp, method, **kw):
        chunks = inp.chunks
        map_chunks = []
        for c in chunks:
            chunk_op = op.copy().reset_key()
            chunk_op._method = method
            chunk_op._stage = OperandStage.map
            for k, v in kw.items():
                setattr(chunk_op, k, v)
            chunk_params = c.params
            chunk_shape_list = list(chunk_params['shape'])
            chunk_shape_list[0] = np.nan
            chunk_params['shape'] = tuple(chunk_shape_list)
            chunk_params['index_value'] = \
                gen_unknown_index_value(chunk_params['index_value'], c)
            map_chunks.append(chunk_op.new_chunk([c], kws=[chunk_params]))
        return map_chunks

    @classmethod
    def _gen_tree(cls, op: "DataFrameDropDuplicates", inp, method):
        combine_size = options.combine_size
        new_chunks = cls._gen_map_chunks(op, inp, method)
        while len(new_chunks) > 1:
            out_chunk_size = ceildiv(len(new_chunks), combine_size)
            out_chunks = []
            for i in range(out_chunk_size):
                in_chunks = new_chunks[i * combine_size: (i + 1) * combine_size]
                if in_chunks[0].ndim == 2:
                    kw = dict(dtypes=in_chunks[0].dtypes)
                else:
                    kw = dict(dtype=in_chunks[0].dtype)
                concat_chunk = DataFrameConcat(
                    object_type=in_chunks[0].op.object_type).new_chunk(
                    in_chunks, **kw)
                chunk_op = op.copy().reset_key()
                chunk_op._method = method
                chunk_op._stage = \
                    OperandStage.combine if out_chunk_size > 1 else OperandStage.agg
                params = in_chunks[0].params
                chunk_shape_list = list(params['shape'])
                chunk_shape_list[0] = sum(c.shape[0] for c in in_chunks)
                params['shape'] = tuple(chunk_shape_list)
                chunk_index_list = list(params['index'])
                chunk_index_list[0] = i
                params['index'] = tuple(chunk_index_list)
                params['index_value'] = gen_unknown_index_value(
                    params['index_value'], *in_chunks)
                out_chunks.append(chunk_op.new_chunk([concat_chunk], kws=[params]))
            new_chunks = out_chunks

        return new_chunks

    @classmethod
    def _tile_tree(cls, op: "DataFrameDropDuplicates", inp):
        out = op.outputs[0]

        params = out.params
        params['chunks'] = cls._gen_tree(op, inp, 'tree')
        params['nsplits'] = ((np.nan,),) + inp.nsplits[1:]
        new_op = op.copy()
        return new_op.new_tileables([inp], kws=[params])

    @classmethod
    def _tile_subset_tree(cls, op: "DataFrameDropDuplicates", inp):
        # subset is available for DataFrame only
        inp = asdataframe(inp)
        out = op.outputs[0]
        subset = op.subset
        if subset is None:
            subset = inp.dtypes.index.tolist()
        # select subset first
        subset_df = inp[subset]._inplace_tile()
        # tree aggregate subset
        subset_chunk = cls._gen_tree(op, subset_df, 'subset_tree')[0]

        out_chunks = []
        for c in inp.chunks:
            chunk_op = op.copy().reset_key()
            chunk_op._method = 'subset_tree'
            chunk_op._subset_chunk = subset_chunk
            chunk_params = c.params
            chunk_shape_list = list(chunk_params['shape'])
            chunk_shape_list[0] = np.nan
            chunk_params['shape'] = tuple(chunk_shape_list)
            out_chunks.append(chunk_op.new_chunk([c, subset_chunk],
                                                 kws=[chunk_params]))

        new_op = op.copy()
        params = out.params
        params['chunks'] = out_chunks
        params['nsplits'] = ((np.nan,) * len(inp.chunks),) + inp.nsplits[1:]
        return new_op.new_tileables([inp], kws=[params])

    @classmethod
    def _tile_shuffle(cls, op: "DataFrameDropDuplicates", inp):
        out = op.outputs[0]

        map_chunks = cls._gen_map_chunks(op, inp, 'shuffle',
                                         _shuffle_size=inp.chunk_shape[0])
        proxy_chunk = DataFrameShuffleProxy(
            object_type=map_chunks[0].op.object_type).new_chunk(map_chunks, shape=())
        reduce_chunks = []
        for i in range(len(map_chunks)):
            reduce_op = op.copy().reset_key()
            reduce_op._method = 'shuffle'
            reduce_op._stage = OperandStage.reduce
            reduce_op._shuffle_phase = 'drop_duplicates'
            reduce_op._shuffle_key = str(i)
            reduce_op._shuffle_size = inp.chunk_shape[0]
            reduce_chunk_params = map_chunks[0].params
            reduce_chunk_params['index'] = (i,) + reduce_chunk_params['index'][1:]
            reduce_chunks.append(
                reduce_op.new_chunk([proxy_chunk], kws=[reduce_chunk_params]))

        put_back_proxy_chunk = DataFrameShuffleProxy(
            object_type=map_chunks[0].op.object_type).new_chunk(reduce_chunks, shape=())
        put_back_chunks = []
        for i in range(len(map_chunks)):
            put_back_op = op.copy().reset_key()
            put_back_op._method = 'shuffle'
            put_back_op._stage = OperandStage.reduce
            put_back_op._shuffle_phase = 'put_back'
            put_back_op._shuffle_key = str(i)
            put_back_chunk_params = map_chunks[0].params
            put_back_chunk_params['index'] = (i,) + put_back_chunk_params['index'][1:]
            put_back_chunks.append(
                put_back_op.new_chunk([put_back_proxy_chunk], kws=[put_back_chunk_params]))

        if op.ignore_index:
            put_back_chunks = standardize_range_index(put_back_chunks)

        new_op = op.copy()
        params = out.params
        params['chunks'] = put_back_chunks
        params['nsplits'] = ((np.nan,) * len(put_back_chunks),) + inp.nsplits[1:]
        return new_op.new_tileables([inp], kws=[params])

    @classmethod
    def tile(cls, op: "DataFrameDropDuplicates"):
        inp = op.input

        if len(inp.chunks) == 1:
            # one chunk
            return cls._tile_one_chunk(op)

        if inp.ndim == 2 and inp.chunk_shape[1] > 1:
            check_chunks_unknown_shape([inp], TilesError)
            inp = inp.rechunk({1: inp.shape[1]})._inplace_tile()

        default_tile = cls._tile_tree

        if op.method == 'auto':
            # if method == 'auto', pick appropriate method
            if np.isnan(inp.shape[0]) or op.subset is None:
                # if any unknown shape exist,
                # choose merge method
                return default_tile(op, inp)

            # check subset data to see if it's small enough
            subset_dtypes = inp.dtypes[op.subset]
            memory_usage = 0.
            for s_dtype in subset_dtypes:
                if s_dtype.kind == 'O' or not hasattr(s_dtype, 'itemsize'):
                    # object, just use default tile
                    return default_tile(op, inp)
                else:
                    memory_usage += s_dtype.itemsize * inp.shape[0]
            if memory_usage <= options.chunk_store_limit:
                # if subset is small enough, use method 'subset_tree'
                return cls._tile_subset_tree(op, inp)
            else:
                return default_tile(op, inp)
        elif op.method == 'subset_tree':
            return cls._tile_subset_tree(op, inp)
        elif op.method == 'tree':
            return cls._tile_tree(op, inp)
        else:
            assert op.method == 'shuffle'
            return cls._tile_shuffle(op, inp)

    @classmethod
    def _get_xdf(cls, x):
        if cudf is None:
            return pd
        elif isinstance(x, (pd.Index, pd.Series, pd.DataFrame)):  # pragma: no cover
            return pd
        else:  # pragma: no cover
            return cudf

    @classmethod
    def _drop_duplicates(cls, inp, op, subset=None, keep=None, ignore_index=None):
        if ignore_index is None:
            ignore_index = op.ignore_index
        if subset is None:
            subset = op.subset
        if keep is None:
            keep = op.keep
        if inp.ndim == 2:
            try:
                return inp.drop_duplicates(subset=subset,
                                           keep=keep,
                                           ignore_index=ignore_index)
            except TypeError:
                # no ignore_index for pandas < 1.0
                ret = inp.drop_duplicates(subset=subset,
                                          keep=keep)
                if ignore_index:
                    ret.reset_index(drop=True, inplace=True)
                return ret
        else:
            return inp.drop_duplicates(keep=keep)

    @classmethod
    def _execute_chunk(cls, ctx, op):
        inp = ctx[op.input.key]
        ctx[op.outputs[0].key] = cls._drop_duplicates(inp, op)

    @classmethod
    def _execute_subset_tree_map(cls, ctx, op):
        out = op.outputs[0]
        idx = out.index[0]
        # input contains subset only
        inp = ctx[op.input.key]
        xdf = cls._get_xdf(inp)

        # index would be (chunk_index, i)
        index = xdf.MultiIndex.from_arrays(
            [np.full(inp.shape[0], idx), np.arange(inp.shape[0])],
            names=['_chunk_index_', '_i_'])
        inp = inp.set_index(index)
        ctx[out.key] = cls._drop_duplicates(inp, op, ignore_index=False)

    @classmethod
    def _execute_subset_tree_combine(cls, ctx, op):
        inp = ctx[op.input.key]
        ctx[op.outputs[0].key] = cls._drop_duplicates(inp, op,
                                                      ignore_index=False)

    @classmethod
    def _execute_subset_tree_agg(cls, ctx, op):
        inp = ctx[op.input.key]
        ret = cls._drop_duplicates(inp, op, ignore_index=False)
        ret = ret.index.to_frame()
        ret.reset_index(drop=True, inplace=True)
        ctx[op.outputs[0].key] = ret

    @classmethod
    def _execute_subset_tree_post(cls, ctx, op):
        inp = ctx[op.input.key]
        out = op.outputs[0]
        idx = op.outputs[0].index[0]
        subset = ctx[op.subset_chunk.key]
        selected = subset[subset['_chunk_index_'] == idx]['_i_']
        ret = inp.iloc[selected]
        if op.ignore_index:
            prev_size = (subset['_chunk_index_'] < out.index[0]).sum()
            ret.index = pd.RangeIndex(prev_size, prev_size + len(ret))
        ctx[op.outputs[0].key] = ret

    @classmethod
    def _execute_shuffle_map(cls, ctx, op):
        out = op.outputs[0]
        shuffle_size = op.shuffle_size
        subset = op.subset

        inp = ctx[op.input.key]
        dropped = cls._drop_duplicates(inp, op)
        if dropped.ndim == 1:
            dropped = dropped.to_frame()
            subset = dropped.columns.tolist()
        else:
            if subset is None:
                subset = dropped.columns.tolist()
        dropped['_chunk_index_'] = out.index[0]
        dropped['_i_'] = np.arange(dropped.shape[0])
        hashed = hash_dataframe_on(dropped, subset, shuffle_size)
        for i, data in enumerate(hashed):
            ctx[(out.key, str(i))] = dropped.loc[data]

    @classmethod
    def _execute_shuffle_reduce(cls, ctx, op):
        out = op.outputs[0]

        input_keys, _ = get_shuffle_input_keys_idxes(op.inputs[0])
        inputs = [ctx[(inp_key, str(out.index[0]))] for inp_key in input_keys]
        xdf = cls._get_xdf(inputs[0])
        inp = xdf.concat(inputs)
        dropped = cls._drop_duplicates(inp, op,
                                       subset=[c for c in inp.columns
                                               if c not in ('_chunk_index_', '_i_')],
                                       keep=op.keep, ignore_index=op.ignore_index)
        for i in range(op.shuffle_size):
            filtered = dropped[dropped['_chunk_index_'] == i]
            del filtered['_chunk_index_']
            ctx[(out.key, str(i))] = filtered

    @classmethod
    def _execute_shuffle_put_back(cls, ctx, op):
        out = op.outputs[0]

        input_keys, _ = get_shuffle_input_keys_idxes(op.inputs[0])
        inputs = [ctx[(inp_key, str(out.index[0]))] for inp_key in input_keys]
        xdf = cls._get_xdf(inputs[0])
        inp = xdf.concat(inputs)
        inp.sort_values('_i_', inplace=True)
        del inp['_i_']

        if out.op.object_type == ObjectType.index:
            assert inp.shape[1] == 1
            ret = xdf.Index(inp.iloc[:, 0])
        elif out.op.object_type == ObjectType.series:
            assert inp.shape[1] == 1
            ret = inp.iloc[:, 0]
        else:
            ret = inp

        if op.ignore_index:
            ret.reset_index(drop=True, inplace=True)
        ctx[op.outputs[0].key] = ret

    @classmethod
    def execute(cls, ctx, op):
        if op.method is None:
            # one chunk
            cls._execute_chunk(ctx, op)
        elif op.method == 'tree':
            # tree
            cls._execute_chunk(ctx, op)
        elif op.method == 'subset_tree':
            # subset tree
            if op.stage == OperandStage.map:
                cls._execute_subset_tree_map(ctx, op)
            elif op.stage == OperandStage.combine:
                cls._execute_subset_tree_combine(ctx, op)
            elif op.stage == OperandStage.agg:
                cls._execute_subset_tree_agg(ctx, op)
            else:
                # post
                cls._execute_subset_tree_post(ctx, op)
        else:
            assert op.method == 'shuffle'
            if op.stage == OperandStage.map:
                cls._execute_shuffle_map(ctx, op)
            elif op.shuffle_phase == 'drop_duplicates':
                cls._execute_shuffle_reduce(ctx, op)
            else:
                assert op.shuffle_phase == 'put_back'
                cls._execute_shuffle_put_back(ctx, op)


def _validate_subset(df, subset):
    if subset is None:
        return subset
    if not is_list_like(subset):
        subset = [subset]
    else:
        subset = list(subset)

    for s in subset:
        if s not in df.dtypes:
            raise KeyError(pd.Index([s]))

    return subset


def df_drop_duplicates(df, subset=None, keep='first',
                       inplace=False, ignore_index=False, method='auto'):
    """
    Return DataFrame with duplicate rows removed.

    Considering certain columns is optional. Indexes, including time indexes
    are ignored.

    Parameters
    ----------
    subset : column label or sequence of labels, optional
        Only consider certain columns for identifying duplicates, by
        default use all of the columns.
    keep : {'first', 'last', False}, default 'first'
        Determines which duplicates (if any) to keep.
        - ``first`` : Drop duplicates except for the first occurrence.
        - ``last`` : Drop duplicates except for the last occurrence.
        - False : Drop all duplicates.
    inplace : bool, default False
        Whether to drop duplicates in place or to return a copy.
    ignore_index : bool, default False
        If True, the resulting axis will be labeled 0, 1, …, n - 1.

    Returns
    -------
    DataFrame
        DataFrame with duplicates removed or None if ``inplace=True``.
    """
    if method not in ('auto', 'tree', 'subset_tree', 'shuffle', None):
        raise ValueError("method could only be one of "
                         "'auto', 'tree', 'subset_tree', 'shuffle' or None")
    subset = _validate_subset(df, subset)
    op = DataFrameDropDuplicates(subset=subset, keep=keep,
                                 ignore_index=ignore_index,
                                 method=method)
    return op(df, inplace=inplace)


def series_drop_duplicates(series, keep='first', inplace=False, method='auto'):
    """
    Return Series with duplicate values removed.

    Parameters
    ----------
    keep : {'first', 'last', ``False``}, default 'first'
        Method to handle dropping duplicates:

        - 'first' : Drop duplicates except for the first occurrence.
        - 'last' : Drop duplicates except for the last occurrence.
        - ``False`` : Drop all duplicates.

    inplace : bool, default ``False``
        If ``True``, performs operation inplace and returns None.

    Returns
    -------
    Series
        Series with duplicates dropped.

    See Also
    --------
    Index.drop_duplicates : Equivalent method on Index.
    DataFrame.drop_duplicates : Equivalent method on DataFrame.
    Series.duplicated : Related method on Series, indicating duplicate
        Series values.

    Examples
    --------
    Generate a Series with duplicated entries.

    >>> import mars.dataframe as md
    >>> s = md.Series(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'],
    ...               name='animal')
    >>> s.execute()
    0      lama
    1       cow
    2      lama
    3    beetle
    4      lama
    5     hippo
    Name: animal, dtype: object

    With the 'keep' parameter, the selection behaviour of duplicated values
    can be changed. The value 'first' keeps the first occurrence for each
    set of duplicated entries. The default value of keep is 'first'.

    >>> s.drop_duplicates().execute()
    0      lama
    1       cow
    3    beetle
    5     hippo
    Name: animal, dtype: object

    The value 'last' for parameter 'keep' keeps the last occurrence for
    each set of duplicated entries.

    >>> s.drop_duplicates(keep='last').execute()
    1       cow
    3    beetle
    4      lama
    5     hippo
    Name: animal, dtype: object

    The value ``False`` for parameter 'keep' discards all sets of
    duplicated entries. Setting the value of 'inplace' to ``True`` performs
    the operation inplace and returns ``None``.

    >>> s.drop_duplicates(keep=False, inplace=True)
    >>> s.execute()
    1       cow
    3    beetle
    5     hippo
    Name: animal, dtype: object
    """
    if method not in ('auto', 'tree', 'shuffle', None):
        raise ValueError("method could only be one of "
                         "'auto', 'tree', 'shuffle' or None")
    op = DataFrameDropDuplicates(keep=keep, method=method)
    return op(series, inplace=inplace)


def index_drop_duplicates(index, keep='first', method='auto'):
    """
    Return Index with duplicate values removed.

    Parameters
    ----------
    keep : {'first', 'last', ``False``}, default 'first'
        - 'first' : Drop duplicates except for the first occurrence.
        - 'last' : Drop duplicates except for the last occurrence.
        - ``False`` : Drop all duplicates.

    Returns
    -------
    deduplicated : Index

    See Also
    --------
    Series.drop_duplicates : Equivalent method on Series.
    DataFrame.drop_duplicates : Equivalent method on DataFrame.
    Index.duplicated : Related method on Index, indicating duplicate
        Index values.

    Examples
    --------
    Generate an pandas.Index with duplicate values.

    >>> import mars.dataframe as md

    >>> idx = md.Index(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'])

    The `keep` parameter controls  which duplicate values are removed.
    The value 'first' keeps the first occurrence for each
    set of duplicated entries. The default value of keep is 'first'.

    >>> idx.drop_duplicates(keep='first').execute()
    Index(['lama', 'cow', 'beetle', 'hippo'], dtype='object')

    The value 'last' keeps the last occurrence for each set of duplicated
    entries.

    >>> idx.drop_duplicates(keep='last').execute()
    Index(['cow', 'beetle', 'lama', 'hippo'], dtype='object')

    The value ``False`` discards all sets of duplicated entries.

    >>> idx.drop_duplicates(keep=False).execute()
    Index(['cow', 'beetle', 'hippo'], dtype='object')
    """
    if method not in ('auto', 'tree', 'shuffle', None):
        raise ValueError("method could only be one of "
                         "'auto', 'tree', 'shuffle' or None")
    op = DataFrameDropDuplicates(keep=keep, method=method)
    return op(index)
