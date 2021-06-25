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

import itertools
from collections import defaultdict
from collections.abc import Iterable
from functools import reduce

import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ...core import get_output_types, recursive_tile
from ...core.operand import OperandStage, MapReduceOperand
from ...dataframe.utils import parse_index
from ...lib import sparse
from ...serialization.serializables import FieldTypes, TupleField, KeyField
from ...tensor.utils import validate_axis, check_random_state, gen_random_seeds, decide_unify_split
from ...tensor.array_utils import get_array_module
from ...utils import tokenize, lazy_import, has_unknown_shape
from ...core import ExecutableTuple
from ..operands import LearnOperandMixin, OutputType, LearnShuffleProxy
from ..utils import convert_to_tensor_or_dataframe


cudf = lazy_import('cudf')


def _shuffle_index_value(op, index_value, chunk_index=None):
    key = tokenize((op._values_, chunk_index, index_value.key))
    return parse_index(pd.Index([], index_value.to_pandas().dtype), key=key)


def _safe_slice(obj, slc, output_type):
    if output_type == OutputType.tensor:
        return obj[slc]
    else:
        return obj.iloc[slc]


class LearnShuffle(MapReduceOperand, LearnOperandMixin):
    _op_type_ = OperandDef.PERMUTATION

    _axes = TupleField('axes', FieldTypes.int32)
    _seeds = TupleField('seeds', FieldTypes.uint32)

    _input = KeyField('input')
    _reduce_sizes = TupleField('reduce_sizes', FieldTypes.uint32)

    def __init__(self, axes=None, seeds=None, output_types=None, reduce_sizes=None, **kw):
        super().__init__(_axes=axes, _seeds=seeds, _output_types=output_types,
                         _reduce_sizes=reduce_sizes, **kw)

    @property
    def axes(self):
        return self._axes

    @property
    def seeds(self):
        return self._seeds

    @property
    def input(self):
        return self._input

    @property
    def reduce_sizes(self):
        return self._reduce_sizes

    @property
    def output_limit(self):
        if self.stage is None:
            return len(self.output_types)
        return 1

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, arrays):
        params = self._calc_params([ar.params for ar in arrays])
        return self.new_tileables(arrays, kws=params)

    def _shuffle_index_value(self, index_value):
        return _shuffle_index_value(self, index_value)

    def _shuffle_dtypes(self, dtypes):
        seed = self.seeds[self.axes.index(1)]
        rs = np.random.RandomState(seed)
        shuffled_dtypes = dtypes[rs.permutation(np.arange(len(dtypes)))]
        return shuffled_dtypes

    def _calc_params(self, params):
        axes = set(self.axes)
        for i, output_type, param in zip(itertools.count(0), self.output_types, params):
            if output_type == OutputType.dataframe:
                if 0 in axes:
                    param['index_value'] = self._shuffle_index_value(param['index_value'])
                if 1 in axes:
                    dtypes = param['dtypes'] = self._shuffle_dtypes(param['dtypes'])
                    param['columns_value'] = parse_index(dtypes.index, store_data=True)
            elif output_type == OutputType.series:
                if 0 in axes:
                    param['index_value'] = self._shuffle_index_value(param['index_value'])
            param['_position_'] = i
        return params

    @staticmethod
    def _safe_rechunk(tileable, ax_nsplit):
        do_rechunk = False
        for ax, nsplit in ax_nsplit.items():
            if ax >= tileable.ndim:
                continue
            if tuple(tileable.nsplits[ax]) != tuple(nsplit):
                do_rechunk = True
        if do_rechunk:
            return (yield from recursive_tile(tileable.rechunk(ax_nsplit)))
        else:
            return tileable

    @classmethod
    def _calc_chunk_params(cls, in_chunk, axes, chunk_shape, output, output_type,
                           chunk_op, no_shuffle: bool):
        params = {'index': in_chunk.index}
        if output_type == OutputType.tensor:
            shape_c = list(in_chunk.shape)
            for ax in axes:
                if not no_shuffle and chunk_shape[ax] > 1:
                    shape_c[ax] = np.nan
            params['shape'] = tuple(shape_c)
            params['dtype'] = in_chunk.dtype
            params['order'] = output.order
        elif output_type == OutputType.dataframe:
            shape_c = list(in_chunk.shape)
            if 0 in axes:
                if not no_shuffle and chunk_shape[0] > 1:
                    shape_c[0] = np.nan
            params['shape'] = tuple(shape_c)
            if 1 not in axes:
                params['dtypes'] = in_chunk.dtypes
                params['columns_value'] = in_chunk.columns_value
            else:
                params['dtypes'] = output.dtypes
                params['columns_value'] = output.columns_value
            params['index_value'] = _shuffle_index_value(chunk_op, in_chunk.index_value, in_chunk.index)
        else:
            assert output_type == OutputType.series
            if no_shuffle:
                params['shape'] = in_chunk.shape
            else:
                params['shape'] = (np.nan,)
            params['name'] = in_chunk.name
            params['index_value'] = _shuffle_index_value(chunk_op, in_chunk.index_value, in_chunk.index)
            params['dtype'] = in_chunk.dtype
        return params

    @classmethod
    def tile(cls, op):
        inputs = op.inputs
        if has_unknown_shape(inputs):
            yield
        axis_to_nsplits = defaultdict(list)
        has_dataframe = any(output_type == OutputType.dataframe
                            for output_type in op.output_types)
        for ax in op.axes:
            if has_dataframe and ax == 1:
                # if DataFrame exists, for the columns axis,
                # we only allow 1 chunk to ensure the columns consistent
                axis_to_nsplits[ax].append((inputs[0].shape[ax],))
                continue
            for inp in inputs:
                if ax < inp.ndim:
                    axis_to_nsplits[ax].append(inp.nsplits[ax])
        ax_nsplit = {ax: decide_unify_split(*ns) for ax, ns in axis_to_nsplits.items()}
        rechunked_inputs = []
        for inp in inputs:
            inp = yield from cls._safe_rechunk(inp, ax_nsplit)
            rechunked_inputs.append(inp)
        inputs = rechunked_inputs

        mapper_seeds = [None] * len(op.axes)
        reducer_seeds = [None] * len(op.axes)
        for i, ax in enumerate(op.axes):
            rs = np.random.RandomState(op.seeds[i])
            size = len(ax_nsplit[ax])
            if size > 1:
                mapper_seeds[i] = gen_random_seeds(size, rs)
                reducer_seeds[i] = gen_random_seeds(size, rs)
            else:
                mapper_seeds[i] = reducer_seeds[i] = [op.seeds[i]] * size
        out_chunks = []
        out_nsplits = []
        for output_type, inp, oup in zip(op.output_types, inputs, op.outputs):
            inp_axes = tuple(ax for ax in op.axes if ax < inp.ndim)
            reduce_sizes = tuple(inp.chunk_shape[ax] for ax in inp_axes)
            output_types = [output_type]

            if len(inp_axes) == 0:
                continue

            nsplits = list(inp.nsplits)
            for ax in inp_axes:
                cs = len(nsplits[ax])
                if cs > 1:
                    nsplits[ax] = (np.nan,) * cs
            out_nsplits.append(tuple(nsplits))

            if all(reduce_size == 1 for reduce_size in reduce_sizes):
                # no need to do shuffle
                chunks = []
                for c in inp.chunks:
                    chunk_op = LearnShuffle(axes=inp_axes, seeds=op.seeds[:len(inp_axes)],
                                            output_types=output_types)
                    params = cls._calc_chunk_params(c, inp_axes, inp.chunk_shape,
                                                    oup, output_type, chunk_op, True)
                    out_chunk = chunk_op.new_chunk([c], kws=[params])
                    chunks.append(out_chunk)
                out_chunks.append(chunks)
                continue

            if inp.ndim > 1:
                left_chunk_shape = [s for ax, s in enumerate(inp.chunk_shape) if ax not in inp_axes]
                idx_iter = itertools.product(*[range(s) for s in left_chunk_shape])
            else:
                idx_iter = [()]
            reduce_chunks = []
            out_chunks.append(reduce_chunks)
            for idx in idx_iter:
                map_chunks = []
                for reducer_inds in itertools.product(*[range(s) for s in reduce_sizes]):
                    inp_index = list(idx)
                    for ax, reducer_ind in zip(inp_axes, reducer_inds):
                        inp_index.insert(ax, reducer_ind)
                    inp_index = tuple(inp_index)
                    in_chunk = inp.cix[inp_index]
                    params = in_chunk.params
                    map_chunk_op = LearnShuffle(
                        stage=OperandStage.map,
                        output_types=output_types, axes=inp_axes,
                        seeds=tuple(mapper_seeds[j][in_chunk.index[ax]]
                                    for j, ax in enumerate(inp_axes)),
                        reduce_sizes=reduce_sizes
                    )
                    map_chunk = map_chunk_op.new_chunk([in_chunk], **params)
                    map_chunks.append(map_chunk)

                map_chunk_kw = {}
                if output_type == OutputType.tensor:
                    map_chunk_kw = {'dtype': inp.dtype, 'shape': ()}
                proxy_chunk = LearnShuffleProxy(_tileable_keys=[inp.key], output_types=[output_type]) \
                    .new_chunk(map_chunks, **map_chunk_kw)

                reduce_axes = tuple(ax for j, ax in enumerate(inp_axes) if reduce_sizes[j] > 1)
                reduce_sizes_ = tuple(rs for rs in reduce_sizes if rs > 1)
                for c in map_chunks:
                    chunk_op = LearnShuffle(
                        stage=OperandStage.reduce,
                        output_types=output_types, axes=reduce_axes,
                        seeds=tuple(reducer_seeds[j][c.index[ax]] for j, ax in enumerate(inp_axes)
                                    if reduce_sizes[j] > 1),
                        reduce_sizes=reduce_sizes_)
                    params = cls._calc_chunk_params(c, inp_axes, inp.chunk_shape, oup,
                                                    output_type, chunk_op, False)
                    reduce_chunk = chunk_op.new_chunk([proxy_chunk], kws=[params])
                    reduce_chunks.append(reduce_chunk)

        new_op = op.copy()
        params = [out.params for out in op.outputs]
        if len(out_chunks) < len(op.outputs):
            # axes are all higher than its ndim
            for i, inp in enumerate(op.inputs):
                if all(ax >= inp.ndim for ax in op.axes):
                    out_chunks.insert(i, inp.chunks)
                    out_nsplits.insert(i, inp.nsplits)
            assert len(out_chunks) == len(op.outputs)
        for i, param, chunks, ns in zip(itertools.count(), params, out_chunks, out_nsplits):
            param['chunks'] = chunks
            param['nsplits'] = ns
            param['_position_'] = i
        return new_op.new_tileables(op.inputs, kws=params)

    @classmethod
    def execute_single(cls, ctx, op):
        x = ctx[op.inputs[0].key]
        conv = lambda x: x
        if op.output_types[0] == OutputType.tensor:
            xp = get_array_module(x)
            if xp is sparse:
                conv = lambda x: x
            else:
                conv = xp.ascontiguousarray \
                    if op.outputs[0].order.value == 'C' else xp.asfortranarray

        for axis, seed in zip(op.axes, op.seeds):
            size = x.shape[axis]
            ind = np.random.RandomState(seed).permutation(np.arange(size))
            slc = (slice(None),) * axis + (ind,)
            x = _safe_slice(x, slc, op.output_types[0])

        ctx[op.outputs[0].key] = conv(x)

    @classmethod
    def execute_map(cls, ctx, op):
        out = op.outputs[0]
        x = ctx[op.input.key]
        axes, seeds, reduce_sizes = op.axes, op.seeds, op.reduce_sizes
        if 1 in set(op.reduce_sizes):
            # if chunk size on shuffle axis == 0
            inds = [slice(None) for _ in range(x.ndim)]
            extra_axes, extra_seeds, extra_reduce_sizes = [], [], []
            for ax, seed, reduce_size in zip(axes, seeds, reduce_sizes):
                rs = np.random.RandomState(seed)
                if reduce_size == 1:
                    inds[ax] = rs.permutation(np.arange(x.shape[ax]))
                else:
                    extra_axes.append(ax)
                    extra_seeds.append(seed)
                    extra_reduce_sizes.append(reduce_size)
            # for the reduce == 1
            # do shuffle on the map phase
            x = _safe_slice(x, tuple(inds), op.output_types[0])
            axes, seeds, reduce_sizes = extra_axes, extra_seeds, extra_reduce_sizes

        to_hash_inds = []
        for ax, seed, reduce_size in zip(axes, seeds, reduce_sizes):
            rs = np.random.RandomState(seed)
            to_hash_inds.append(rs.randint(reduce_size, size=x.shape[ax]))

        for reduce_index in itertools.product(*(range(rs) for rs in reduce_sizes)):
            index = list(out.index)
            for ax, ind in zip(axes, reduce_index):
                index[ax] = ind
            selected = x
            for ax, to_hash_ind in zip(axes, to_hash_inds):
                slc = (slice(None),) * ax + (to_hash_ind == index[ax],)
                selected = _safe_slice(selected, slc, op.output_types[0])
            ctx[out.key, tuple(index)] = selected

    @classmethod
    def execute_reduce(cls, ctx, op: "LearnShuffle"):
        inputs_grid = np.empty(op.reduce_sizes, dtype=object)
        for input_index, inp in op.iter_mapper_data_with_index(ctx):
            reduce_index = tuple(input_index[ax] for ax in op.axes)
            inputs_grid[reduce_index] = inp
        ret = cls._concat_grid(inputs_grid, op.axes, op.output_types[0])
        for ax, seed in zip(op.axes, op.seeds):
            ind = np.random.RandomState(seed).permutation(np.arange(ret.shape[ax]))
            slc = (slice(None),) * ax + (ind,)
            ret = _safe_slice(ret, slc, op.output_types[0])
        ctx[op.outputs[0].key] = ret

    @classmethod
    def _concat_grid(cls, grid, axes, output_type):
        if output_type == OutputType.tensor:
            return cls._concat_tensor_grid(grid, axes)
        elif output_type == OutputType.dataframe:
            return cls._concat_dataframe_grid(grid, axes)
        else:
            assert output_type == OutputType.series
            return cls._concat_series_grid(grid, axes)

    @classmethod
    def _concat_dataframe_grid(cls, grid, axes):
        xdf = pd if isinstance(grid.ravel()[0], pd.DataFrame) else cudf
        # if 1 exists in axes, the shuffle would have been done in map phase
        assert len(axes) == 1
        return xdf.concat(grid, axis=axes[0])

    @classmethod
    def _concat_series_grid(cls, grid, axes):
        assert axes == (0,) and grid.ndim == 1

        return reduce(lambda a, b: a.append(b), grid)

    @classmethod
    def _concat_tensor_grid(cls, grid, axes):
        cur = grid
        xp = get_array_module(grid.ravel()[0])
        for ax, i in zip(axes[:0:-1], range(len(axes) - 1, 0, -1)):
            new_shape = grid.shape[:i]
            new_grid = np.empty(new_shape, dtype=object)
            for idx in itertools.product(*(range(s) for s in new_shape)):
                new_grid[idx] = xp.concatenate(cur[idx], axis=ax)
            cur = new_grid
        return xp.concatenate(cur, axis=axes[0])

    @classmethod
    def execute(cls, ctx, op):
        if op.stage == OperandStage.map:
            cls.execute_map(ctx, op)
        elif op.stage == OperandStage.reduce:
            cls.execute_reduce(ctx, op)
        else:
            cls.execute_single(ctx, op)


def shuffle(*arrays, **options):
    arrays = [convert_to_tensor_or_dataframe(ar) for ar in arrays]
    axes = options.pop('axes', (0,))
    if not isinstance(axes, Iterable):
        axes = (axes,)
    elif not isinstance(axes, tuple):
        axes = tuple(axes)
    random_state = check_random_state(
        options.pop('random_state', None)).to_numpy()
    if options:
        raise TypeError('shuffle() got an unexpected '
                        f'keyword argument {next(iter(options))}')

    max_ndim = max(ar.ndim for ar in arrays)
    axes = tuple(np.unique([validate_axis(max_ndim, ax) for ax in axes]).tolist())
    seeds = gen_random_seeds(len(axes), random_state)

    # verify shape
    for ax in axes:
        shapes = {ar.shape[ax] for ar in arrays if ax < ar.ndim}
        if len(shapes) > 1:
            raise ValueError(f'arrays do not have same shape on axis {ax}')

    op = LearnShuffle(axes=axes, seeds=seeds,
                      output_types=get_output_types(*arrays))
    shuffled_arrays = op(arrays)
    if len(arrays) == 1:
        return shuffled_arrays[0]
    else:
        return ExecutableTuple(shuffled_arrays)
