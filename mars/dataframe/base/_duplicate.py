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
from pandas.api.types import is_list_like

from ...config import options
from ...core import OutputType, recursive_tile
from ...core.operand import OperandStage, MapReduceOperand
from ...serialization.serializables import AnyField, Int32Field, StringField, KeyField
from ...utils import ceildiv, has_unknown_shape, lazy_import
from ..initializer import DataFrame as asdataframe
from ..operands import DataFrameOperandMixin, DataFrameShuffleProxy

cudf = lazy_import('cudf', globals=globals())


class DuplicateOperand(MapReduceOperand, DataFrameOperandMixin):
    _input = KeyField('input')
    _subset = AnyField('subset')
    _keep = AnyField('keep')
    _method = StringField('method')

    # subset chunk, used for method 'subset_tree'
    _subset_chunk = KeyField('subset_chunk')
    # shuffle phase, used in shuffle method
    _shuffle_size = Int32Field('shuffle_size')

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
    def method(self):
        return self._method

    @property
    def subset_chunk(self):
        return self._subset_chunk

    @property
    def shuffle_size(self):
        return self._shuffle_size

    @classmethod
    def _get_shape(cls, input_shape, op):  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def _gen_tileable_params(cls, op: "DuplicateOperand", input_params):  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def _gen_chunk_params(cls, op: "DuplicateOperand", input_chunk):  # pragma: no cover
        raise NotImplementedError

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]
        if self._subset_chunk is not None:
            self._subset_chunk = self._inputs[1]

    @classmethod
    def _tile_one_chunk(cls, op: "DuplicateOperand"):
        inp = op.input
        out = op.outputs[0]

        chunk_op = op.copy().reset_key()
        chunk_op._method = None
        in_chunk = inp.chunks[0]
        chunk_params = cls._gen_chunk_params(chunk_op, in_chunk)
        chunk = chunk_op.new_chunk([in_chunk], kws=[chunk_params])

        params = out.params
        params['chunks'] = [chunk]
        params['nsplits'] = tuple((s,) for s in chunk.shape)
        new_op = op.copy()
        return new_op.new_tileables([inp], kws=[params])

    @classmethod
    def _get_map_output_types(cls, input_chunk, method: str):
        raise NotImplementedError

    @classmethod
    def _gen_map_chunks(cls, op: "DuplicateOperand", inp, method, **kw):
        chunks = inp.chunks
        map_chunks = []
        for c in chunks:
            chunk_op = op.copy().reset_key()
            chunk_op._output_types = cls._get_map_output_types(c, method)
            chunk_op._method = method
            chunk_op.stage = OperandStage.map
            for k, v in kw.items():
                setattr(chunk_op, k, v)
            chunk_params = cls._gen_chunk_params(chunk_op, c)
            map_chunks.append(chunk_op.new_chunk([c], kws=[chunk_params]))
        return map_chunks

    @classmethod
    def _gen_tree_chunks(cls, op: "DuplicateOperand", inp, method):
        from ..merge import DataFrameConcat

        out = op.outputs[0]
        combine_size = options.combine_size
        new_chunks = cls._gen_map_chunks(op, inp, method)
        while len(new_chunks) > 1:
            out_chunk_size = ceildiv(len(new_chunks), combine_size)
            out_chunks = []
            for i in range(out_chunk_size):
                in_chunks = new_chunks[i * combine_size: (i + 1) * combine_size]
                s = sum(c.shape[0] for c in in_chunks)
                if in_chunks[0].ndim == 2:
                    kw = dict(dtypes=in_chunks[0].dtypes,
                              index_value=in_chunks[0].index_value,
                              columns_value=in_chunks[0].columns_value,
                              shape=(s, in_chunks[0].shape[1]),
                              index=(i, 0))
                else:
                    kw = dict(dtype=in_chunks[0].dtype,
                              index_value=in_chunks[0].index_value,
                              name=in_chunks[0].name,
                              shape=(s,),
                              index=(i,))
                concat_chunk = DataFrameConcat(
                    output_types=in_chunks[0].op.output_types).new_chunk(
                    in_chunks, **kw)
                chunk_op = op.copy().reset_key()
                chunk_op._method = method
                chunk_op.stage = \
                    OperandStage.combine if out_chunk_size > 1 else OperandStage.agg
                if out_chunk_size > 1 and method == 'tree':
                    # for tree, chunks except last one should be dataframes,
                    chunk_op._output_types = \
                        [OutputType.dataframe] if out_chunk_size > 1 else \
                            out.op.output_types
                elif method == 'subset_tree':
                    # `subset_tree` will tile chunks that are always dataframes
                    chunk_op._output_types = [OutputType.dataframe]
                params = cls._gen_chunk_params(chunk_op, concat_chunk)
                if out.ndim == 1 and out_chunk_size == 1:
                    params['name'] = out.name
                out_chunks.append(chunk_op.new_chunk([concat_chunk], kws=[params]))
            new_chunks = out_chunks

        return new_chunks

    @classmethod
    def _tile_tree(cls, op: "DuplicateOperand", inp):
        out = op.outputs[0]

        params = out.params
        params['chunks'] = chunks = cls._gen_tree_chunks(op, inp, 'tree')
        params['nsplits'] = tuple((s,) for s in chunks[0].shape)
        new_op = op.copy()
        return new_op.new_tileables([inp], kws=[params])

    @classmethod
    def _tile_subset_tree(cls, op: "DuplicateOperand", inp):
        # subset is available for DataFrame only
        inp = asdataframe(inp)
        out = op.outputs[0]
        subset = op.subset
        if subset is None:
            subset = inp.dtypes.index.tolist()
        # select subset first
        subset_df = yield from recursive_tile(inp[subset])
        # tree aggregate subset
        subset_chunk = cls._gen_tree_chunks(op, subset_df, 'subset_tree')[0]

        out_chunks = []
        for c in inp.chunks:
            chunk_op = op.copy().reset_key()
            chunk_op._method = 'subset_tree'
            chunk_op._subset_chunk = subset_chunk
            chunk_params = cls._gen_chunk_params(chunk_op, c)
            out_chunks.append(chunk_op.new_chunk([c, subset_chunk],
                                                 kws=[chunk_params]))

        new_op = op.copy()
        params = out.params
        params['chunks'] = out_chunks
        splits = tuple(c.shape[0] for c in out_chunks)
        if out.ndim == 2:
            params['nsplits'] = (splits, inp.nsplits[1])
        else:
            params['nsplits'] = (splits,)
        return new_op.new_tileables([inp], kws=[params])

    @classmethod
    def _tile_shuffle(cls, op: "DuplicateOperand", inp):
        out = op.outputs[0]

        map_chunks = cls._gen_map_chunks(op, inp, 'shuffle',
                                         _shuffle_size=inp.chunk_shape[0])
        proxy_chunk = DataFrameShuffleProxy(
            output_types=map_chunks[0].op.output_types).new_chunk(map_chunks, shape=())
        reduce_chunks = []
        for i in range(len(map_chunks)):
            reduce_op = op.copy().reset_key()
            reduce_op._method = 'shuffle'
            reduce_op.stage = OperandStage.reduce
            reduce_op.reducer_phase = 'drop_duplicates'
            reduce_op._shuffle_size = inp.chunk_shape[0]
            reduce_op._output_types = [OutputType.dataframe]
            reduce_chunk_params = map_chunks[0].params
            reduce_chunk_params['index'] = (i,) + reduce_chunk_params['index'][1:]
            reduce_chunks.append(
                reduce_op.new_chunk([proxy_chunk], kws=[reduce_chunk_params]))

        put_back_proxy_chunk = DataFrameShuffleProxy(
            output_types=map_chunks[0].op.output_types).new_chunk(reduce_chunks, shape=())
        put_back_chunks = []
        for i in range(len(map_chunks)):
            put_back_op = op.copy().reset_key()
            put_back_op._method = 'shuffle'
            put_back_op.stage = OperandStage.reduce
            put_back_op.reducer_phase = 'put_back'
            put_back_op.reducer_index = (i,)
            if out.ndim == 2:
                put_back_chunk_params = map_chunks[i].params
            else:
                put_back_chunk_params = out.params.copy()
                map_chunk_params = map_chunks[i].params
                put_back_chunk_params['index_value'] = map_chunk_params['index_value']
                put_back_chunk_params['index'] = map_chunk_params['index'][:1]
            if out.ndim == 1:
                put_back_chunk_params['index'] = (i,)
            else:
                put_back_chunk_params['index'] = (i,) + put_back_chunk_params['index'][1:]
            put_back_chunk_params['shape'] = cls._get_shape(
                map_chunks[i].op.input.shape, op)
            put_back_chunks.append(
                put_back_op.new_chunk([put_back_proxy_chunk], kws=[put_back_chunk_params]))

        new_op = op.copy()
        params = out.params
        params['chunks'] = put_back_chunks
        split = tuple(c.shape[0] for c in put_back_chunks)
        if out.ndim == 2:
            params['nsplits'] = (split, inp.nsplits[1])
        else:
            params['nsplits'] = (split,)
        return new_op.new_tileables([inp], kws=[params])

    @classmethod
    def tile(cls, op: "DuplicateOperand"):
        inp = op.input

        if len(inp.chunks) == 1:
            # one chunk
            return cls._tile_one_chunk(op)

        if inp.ndim == 2 and inp.chunk_shape[1] > 1:
            if has_unknown_shape(inp):
                yield
            inp = yield from recursive_tile(
                inp.rechunk({1: inp.shape[1]}))

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
                r = yield from cls._tile_subset_tree(op, inp)
                return r
            else:
                return default_tile(op, inp)
        elif op.method == 'subset_tree':
            r = yield from cls._tile_subset_tree(op, inp)
            return r
        elif op.method == 'tree':
            return cls._tile_tree(op, inp)
        else:
            assert op.method == 'shuffle'
            return cls._tile_shuffle(op, inp)

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
    def _get_xdf(cls, x):
        if cudf is None:
            return pd
        elif isinstance(x, (pd.Index, pd.Series, pd.DataFrame)):  # pragma: no cover
            return pd
        else:  # pragma: no cover
            return cudf

    @classmethod
    def _execute_subset_tree_map(cls, ctx, op):
        out = op.outputs[0]
        idx = out.index[0]
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


def validate_subset(df, subset):
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
