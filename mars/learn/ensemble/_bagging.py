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
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ... import opcodes
from ...core import OutputType, get_output_types, recursive_tile
from ...core.operand import OperandStage
from ...dataframe.core import DATAFRAME_TYPE
from ...dataframe.utils import parse_index
from ...serialization.serializables import AnyField, BoolField, \
    Int64Field, Float32Field, TupleField, ReferenceField, FieldTypes
from ...tensor.core import TENSOR_CHUNK_TYPE
from ...tensor.random import RandomStateField
from ...tensor.utils import gen_random_seeds
from ...typing import TileableType
from ...utils import has_unknown_shape
from ..operands import LearnOperand, LearnOperandMixin, LearnShuffleProxy
from ..utils.shuffle import LearnShuffle


def _extract_bagging_io(io_list: Iterable, op: LearnOperand,
                        output: bool = False):
    if not isinstance(io_list, Iterable):
        io_list = [io_list]
    input_iter = iter(io_list)
    out = [
        next(input_iter),
        next(input_iter) if op.with_labels else None,
        next(input_iter) if op.with_weights else None,
        next(input_iter) if output and op.with_feature_indices else None,
    ]
    return out


def _get_by_iloc(x, idx, axis=0):
    if hasattr(x, 'iloc'):
        item_getter = x.iloc
    else:
        item_getter = x
    if axis == 0:
        return item_getter[idx]
    else:
        return item_getter[:, idx]


def _concat_on_axis(data_list, axis=0, out_chunk=None):
    if isinstance(out_chunk, TENSOR_CHUNK_TYPE):
        return np.concatenate(data_list, axis=axis)
    else:
        return pd.concat(data_list, axis=axis)


def _concat_by_row(row, out_chunk=None):
    arr = np.empty((1,), dtype=object)
    arr[0] = _concat_on_axis(row.tolist(), axis=0, out_chunk=out_chunk)
    return arr


class BaggingSample(LearnShuffle, LearnOperandMixin):
    _op_type_ = opcodes.BAGGING_SHUFFLE_SAMPLE

    n_estimators: int = Int64Field('n_estimators')
    max_samples = AnyField('max_samples')
    max_features = AnyField('max_features')
    bootstrap: bool = BoolField('bootstrap')
    bootstrap_features: bool = BoolField('bootstrap_features')

    random_state = RandomStateField('random_state')
    sample_random_state = RandomStateField('sample_random_state')
    feature_random_state = RandomStateField('feature_random_state')

    reducer_ratio: float = Float32Field('reducer_ratio')
    n_reducers: int = Int64Field('n_reducers', default=None)
    column_offset: int = Int64Field('column_offset', default=None)

    chunk_shape: Tuple[int] = TupleField('chunk_shape', FieldTypes.int64)
    with_labels: bool = BoolField('with_labels')
    with_weights: bool = BoolField('with_weights')
    with_feature_indices: bool = BoolField('with_feature_indices')

    def __init__(self, max_samples: Union[int, float] = 1.0,
                 max_features: Union[int, float] = 1.0,
                 bootstrap: bool = True,
                 bootstrap_features: bool = False,
                 random_state: np.random.RandomState = None,
                 reducer_ratio: float = 1.0, **kw):
        super().__init__(
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            max_samples=max_samples,
            max_features=max_features,
            reducer_ratio=reducer_ratio,
            random_state=random_state,
            **kw
        )
        if self.random_state is None:
            self.random_state = np.random.RandomState()

    @property
    def output_limit(self) -> int:
        if self.with_feature_indices and self.stage != OperandStage.map:
            return 1 + self.with_labels + self.with_weights + self.with_feature_indices
        return 1

    def __call__(self, in_sample: TileableType,
                 in_labels: Optional[TileableType] = None,
                 in_weights: Optional[TileableType] = None):
        self._output_types = get_output_types(in_sample, in_labels, in_weights)

        self.with_labels = in_labels is not None
        self.with_weights = in_weights is not None
        axis_keep_shape = [
            isinstance(self.max_samples, float) and self.max_samples == 1.0,
            isinstance(self.max_features, float) and self.max_features == 1.0,
        ]
        self.with_feature_indices = not axis_keep_shape[1] or self.bootstrap_features
        if self.with_feature_indices:
            self._output_types += (OutputType.tensor,)

        new_shape = tuple(s if keep_shape else np.nan
                          for s, keep_shape in zip(in_sample.shape, axis_keep_shape))

        kws = []

        data_params = in_sample.params
        data_params['shape'] = new_shape
        kws.append(data_params)

        if in_labels is not None:
            labels_params = in_labels.params
            labels_params['shape'] = (new_shape[0],)
            kws.append(labels_params)

        if in_weights is not None:
            weights_params = in_weights.params
            weights_params['shape'] = (new_shape[0],)
            kws.append(weights_params)

        if self.with_feature_indices:
            feature_params = {
                'shape': (self.n_estimators, new_shape[1]),
                'dtype': np.dtype(int),
            }
            kws.append(feature_params)

        inputs = [in_sample]
        if in_labels is not None:
            inputs.append(in_labels)
        if in_weights is not None:
            inputs.append(in_weights)

        return self.new_tileables(inputs, kws=kws)

    @classmethod
    def _scatter_samples(cls, max_samples: Union[int, float], nsplits: Tuple[int],
                         random_state: np.random.RandomState, n_estimators: int) -> np.ndarray:
        nsp_array = np.array(nsplits)
        dim_size = nsp_array.sum()
        if isinstance(max_samples, int):
            expect_sample_count = max_samples
        else:
            expect_sample_count = int(max_samples * nsp_array.sum())

        if expect_sample_count == dim_size:
            return np.array([list(nsplits)] * n_estimators)

        split_probs = nsp_array / dim_size
        return random_state.multinomial(expect_sample_count, split_probs, size=n_estimators)

    @classmethod
    def tile(cls, op: 'BaggingSample'):
        in_sample, in_labels, in_weights, _ = _extract_bagging_io(op.inputs, op, output=False)
        out_data, out_labels, out_weights, out_feature_indices = _extract_bagging_io(
            op.outputs, op, output=True)

        # make sure all shapes are computed
        if has_unknown_shape(in_sample) or \
                (in_labels is not None and has_unknown_shape(in_labels)) or \
                (in_weights is not None and has_unknown_shape(in_weights)):
            yield

        to_tile = []
        if in_labels is not None:
            in_labels = in_labels.rechunk({0: in_sample.nsplits[0]})
            to_tile.append(in_labels)
        if in_weights is not None:
            in_weights = in_weights.rechunk({0: in_sample.nsplits[0]})
            to_tile.append(in_weights)

        # tile rechunks
        if to_tile:
            tiled = yield from recursive_tile(*to_tile)
            tiled_iter = iter(tiled)
            if in_labels is not None:
                in_labels = next(tiled_iter)
            if in_weights is not None:
                in_weights = next(tiled_iter)

        random_seeds = [gen_random_seeds(n, op.random_state)
                        for n in in_sample.chunk_shape]

        axis_keep_shape = [
            isinstance(op.max_samples, float) and op.max_samples == 1.0
            and not op.bootstrap,
            isinstance(op.max_features, float) and op.max_features == 1.0
            and not op.bootstrap_features,
        ]

        n_reducers = op.n_reducers if op.n_reducers is not None \
            else max(1, int(in_sample.chunk_shape[0] * op.reducer_ratio))

        # todo implement sampling without replacements
        map_chunks = []
        max_samples_splits = cls._scatter_samples(op.max_samples, in_sample.nsplits[0],
                                                  op.random_state, op.n_estimators)
        max_features_splits = cls._scatter_samples(op.max_features, in_sample.nsplits[1],
                                                   op.random_state, op.n_estimators)

        column_cum_offset = np.concatenate([[0], np.cumsum(in_sample.nsplits[1])])
        for chunk in in_sample.chunks:
            new_op = op.copy().reset_key()
            new_op.random_state = None
            new_op.sample_random_state = np.random.RandomState(random_seeds[0][chunk.index[0]])
            new_op.feature_random_state = np.random.RandomState(random_seeds[1][chunk.index[1]])
            new_op.stage = OperandStage.map
            new_op.max_samples = max_samples_splits[:, chunk.index[0]]
            new_op.max_features = max_features_splits[:, chunk.index[1]]
            new_op.n_reducers = n_reducers
            new_op.column_offset = int(column_cum_offset[chunk.index[1]])

            if chunk.index[0] != 0:
                new_op.with_feature_indices = False

            if chunk.index[1] != in_sample.chunk_shape[1] - 1:
                new_op.with_weights = False
                new_op.with_labels = False

            params = chunk.params
            params['shape'] = tuple(s if keep_shape else np.nan
                                    for s, keep_shape in zip(chunk.shape, axis_keep_shape))

            input_chunks = [chunk]
            if new_op.with_labels:
                input_chunks.append(in_labels.cix[chunk.index[0]])
            if new_op.with_weights:
                input_chunks.append(in_weights.cix[chunk.index[0]])
            map_chunks.append(new_op.new_chunk(input_chunks, **params))

        shuffle_op = LearnShuffleProxy(output_types=[OutputType.tensor]) \
            .new_chunk(map_chunks, dtype=np.dtype(int), shape=())

        remain_reducers = op.n_estimators % n_reducers
        reduce_data_chunks = []
        reduce_labels_chunks = []
        reduce_weights_chunks = []
        reduce_feature_chunks = []
        for idx in range(n_reducers):
            new_op = op.copy().reset_key()
            new_op.random_state = None
            new_op.stage = OperandStage.reduce
            new_op.chunk_shape = in_sample.chunk_shape
            new_op.n_estimators = op.n_estimators // n_reducers
            if remain_reducers:
                remain_reducers -= 1
                new_op.n_estimators += 1

            if new_op.n_estimators == 0:
                continue

            kws = []

            data_params = out_data.params
            data_params['index'] = (idx, 0)
            data_params['shape'] = (np.nan, out_data.shape[1])
            kws.append(data_params)

            if op.with_labels:
                labels_params = out_labels.params
                labels_params['index'] = (idx,)
                labels_params['shape'] = (np.nan,)
                kws.append(labels_params)

            if op.with_weights:
                weights_params = out_weights.params
                weights_params['index'] = (idx,)
                weights_params['shape'] = (np.nan,)
                kws.append(weights_params)

            if op.with_feature_indices:
                feature_params = {
                    'index': (idx, 0),
                    'shape': (new_op.n_estimators, out_feature_indices.shape[1]),
                    'dtype': np.dtype(int),
                }
                kws.append(feature_params)

            chunks = new_op.new_chunks([shuffle_op], kws=kws)
            data_chunk, labels_chunk, weights_chunk, feature_chunk \
                = _extract_bagging_io(chunks, op, output=True)

            reduce_data_chunks.append(data_chunk)
            if labels_chunk is not None:
                reduce_labels_chunks.append(labels_chunk)
            if weights_chunk is not None:
                reduce_weights_chunks.append(weights_chunk)
            if feature_chunk is not None:
                reduce_feature_chunks.append(feature_chunk)

        new_op = op.copy().reset_key()

        kws = [{
            'chunks': reduce_data_chunks,
            'nsplits': ((np.nan,) * len(reduce_data_chunks), (out_data.shape[1],)),
            **out_data.params
        }]
        if op.with_labels:
            kws.append({
                'chunks': reduce_labels_chunks,
                'nsplits': ((np.nan,) * len(reduce_data_chunks),),
                **out_labels.params
            })
        if op.with_weights:
            kws.append({
                'chunks': reduce_weights_chunks,
                'nsplits': ((np.nan,) * len(reduce_data_chunks),),
                **out_weights.params
            })
        if op.with_feature_indices:
            estimator_nsplit = tuple(c.op.n_estimators for c in reduce_data_chunks)
            kws.append({
                'chunks': reduce_feature_chunks,
                'nsplits': (estimator_nsplit, (out_feature_indices.shape[1],)),
                **out_feature_indices.params
            })
        return new_op.new_tileables(op.inputs, kws=kws)

    @classmethod
    def _gen_sample_indices(cls, max_range: int, size: int,
                            random_state: np.random.RandomState,
                            with_replacement: bool = False):
        if not with_replacement:
            result = random_state.choice(np.arange(max_range), size, False)
        else:
            result = random_state.randint(0, max_range - 1, size)
        result.sort()
        return result

    @classmethod
    def _execute_map(cls, ctx, op: 'BaggingSample'):
        in_sample, in_labels, in_weights, _ = _extract_bagging_io(op.inputs, op, output=False)
        in_sample_data = ctx[in_sample.key]
        in_labels_data = ctx[in_labels.key] if op.with_labels else None
        in_weights_data = ctx[in_weights.key] if op.with_weights else None
        out_samples = op.outputs[0]

        remains = op.n_estimators % op.n_reducers
        reducer_iters = [itertools.repeat(idx, 1 + op.n_estimators // op.n_reducers)
                         for idx in range(remains)]
        reducer_iters += [itertools.repeat(idx, op.n_estimators // op.n_reducers)
                          for idx in range(remains, op.n_reducers)]
        reducer_iter = itertools.chain(*reducer_iters)

        result_store = defaultdict(lambda: ([], [], [], []))
        for est_id in range(op.n_estimators):
            sampled_data = in_sample_data
            sampled_labels = in_labels_data
            sampled_weights = in_weights_data

            if op.max_samples[est_id] != in_sample_data.shape[0]:
                sample_indices = cls._gen_sample_indices(
                    in_sample_data.shape[0], op.max_samples[est_id],
                    op.sample_random_state, op.bootstrap)

                sampled_data = _get_by_iloc(sampled_data, sample_indices)
                if sampled_labels is not None:
                    sampled_labels = _get_by_iloc(sampled_labels, sample_indices)
                if sampled_weights is not None:
                    sampled_weights = _get_by_iloc(sampled_weights, sample_indices)

            if op.max_features[est_id] != in_sample_data.shape[1]:
                feature_indices = cls._gen_sample_indices(
                    in_sample_data.shape[1], op.max_features[est_id],
                    op.feature_random_state, op.bootstrap_features)

                sampled_data = _get_by_iloc(sampled_data, feature_indices, axis=1)
                if not op.with_feature_indices:
                    feature_indices = None
            else:
                feature_indices = None

            samples, labels, weights, feature_idx_array \
                = result_store[next(reducer_iter)]
            samples.append(sampled_data)
            if sampled_labels is not None:
                labels.append(sampled_labels)
            if sampled_weights is not None:
                weights.append(sampled_weights)
            if feature_indices is not None:
                feature_idx_array.append(feature_indices + op.column_offset)

        for reducer_id, (samples, labels, weights, feature_idx_array) \
                in result_store.items():
            ctx[out_samples.key, (reducer_id, 0)] = \
                tuple(samples + labels + weights + feature_idx_array)

    @classmethod
    def _execute_reduce(cls, ctx, op: 'BaggingSample'):
        out_data, out_labels, out_weights, out_feature_indices = _extract_bagging_io(
            op.outputs, op, output=True)

        input_keys = op.inputs[0].op.source_keys
        input_idxes = op.inputs[0].op.source_idxes

        sample_holder = [
            np.empty(op.chunk_shape, dtype=object)
            for _ in range(op.n_estimators)
        ]

        labels_holder = [
            np.empty(op.chunk_shape[0], dtype=object)
            for _ in range(op.n_estimators)
        ] if op.with_labels else None

        weights_holder = [
            np.empty(op.chunk_shape[0], dtype=object)
            for _ in range(op.n_estimators)
        ] if op.with_weights else None

        feature_indices_holder = [
            np.empty(op.chunk_shape[1], dtype=object)
            for _ in range(op.n_estimators)
        ] if op.with_feature_indices else None

        for input_key, input_idx in zip(input_keys, input_idxes):
            add_feature_index = input_idx[0] == 0
            add_label_weight = input_idx[1] == op.chunk_shape[1] - 1
            chunk_data = ctx[input_key, out_data.index]

            num_groups = 1
            if add_feature_index and op.with_feature_indices:  # contains feature indices
                num_groups += 1
            if add_label_weight:  # contains label or weight
                num_groups += int(op.with_weights) + int(op.with_labels)

            sample_count = len(chunk_data) // num_groups
            assert len(chunk_data) % num_groups == 0

            group_iter = (chunk_data[i * sample_count: (i + 1) * sample_count]
                          for i in range(num_groups))

            for data_idx, sample in enumerate(next(group_iter)):
                sample_holder[data_idx][input_idx] = sample

            if add_label_weight:
                if op.with_labels:
                    for data_idx, label in enumerate(next(group_iter)):
                        labels_holder[data_idx][input_idx[0]] = label
                if op.with_weights:
                    for data_idx, weight in enumerate(next(group_iter)):
                        weights_holder[data_idx][input_idx[0]] = weight

            if add_feature_index and op.with_feature_indices:
                for data_idx, feature_index in enumerate(next(group_iter)):
                    feature_indices_holder[data_idx][input_idx[1]] = feature_index

        data_results: List[Optional[np.ndarray]] = [None] * len(sample_holder)
        for est_idx, sample_mat in enumerate(sample_holder):
            row_chunks = np.apply_along_axis(_concat_by_row, axis=0, arr=sample_mat, out_chunk=out_data)
            data_results[est_idx] = _concat_on_axis(row_chunks[0].tolist(), axis=1, out_chunk=out_data)
        ctx[out_data.key] = tuple(data_results)

        for out, holder in zip(
            (out_labels, out_weights, out_feature_indices),
            (labels_holder, weights_holder, feature_indices_holder),
        ):
            if out is None:
                continue
            results: List[Optional[np.ndarray]] = [None] * len(holder)
            for est_idx, labels_vct in enumerate(holder):
                results[est_idx] = _concat_on_axis(labels_vct.tolist(), out_chunk=out)
            if holder is feature_indices_holder:
                ctx[out.key] = np.stack(results)
            else:
                ctx[out.key] = tuple(results)

    @classmethod
    def execute(cls, ctx, op: 'BaggingSample'):
        if op.stage == OperandStage.map:
            cls._execute_map(ctx, op)
        else:
            cls._execute_reduce(ctx, op)


class BaggingSampleReindex(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.BAGGING_SHUFFLE_REINDEX

    n_estimators: int = Int64Field('n_estimators')
    start_col_index: int = Int64Field('start_col_index', 0)
    feature_indices: TileableType = ReferenceField('feature_indices', default=None)

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if self.feature_indices is not None:
            self.feature_indices = inputs[-1]

    def __call__(self, data: TileableType, feature_indices: TileableType = None):
        self._output_types = get_output_types(data)
        inputs = [data]
        self.feature_indices = feature_indices
        params = data.params
        if feature_indices is not None:
            inputs.append(feature_indices)
            params['shape'] = (data.shape[0], np.nan)
        if isinstance(data, DATAFRAME_TYPE):
            params['index_value'] = parse_index(pd.Int64Index([]), data.key)
        return self.new_tileable(inputs, **params)

    @classmethod
    def tile(cls, op: "BaggingSampleReindex"):
        t_data = op.inputs[0]
        t_out = op.outputs[0]
        t_feature_idxes = op.feature_indices
        cum_nsplits = np.cumsum(np.concatenate([[0], t_data.nsplits[1]]))

        if t_feature_idxes is None:
            out = t_data
            if out.chunk_shape[1] > 1:
                out = yield from recursive_tile(out.rechunk({1: (out.shape[1],)}))
            return out

        # generate map chunks
        map_holder = np.empty(t_data.chunk_shape + (t_feature_idxes.chunk_shape[0],),
                              dtype=np.dtype(object))
        for chunk in t_data.chunks:
            for feature_idx_chunk in t_feature_idxes.chunks:
                new_op = op.copy().reset_key()
                new_op.stage = OperandStage.map
                new_op.start_col_index = int(cum_nsplits[chunk.index[1]])
                params = chunk.params
                new_index = params['index'] = chunk.index + (feature_idx_chunk.index[0],)
                if t_feature_idxes.chunk_shape[0] == 1:
                    new_index = new_index[:-1]
                map_holder[new_index] = new_op.new_chunk(
                    [chunk, feature_idx_chunk], **params
                )
        if op.feature_indices.chunk_shape[0] == 1:
            chunks = map_holder.reshape((t_data.chunk_shape[0],)).tolist()
        else:
            def _gen_combine_chunk(chunks):
                new_op = op.copy().reset_key()
                new_op.feature_indices = None
                new_op.stage = OperandStage.combine
                params = chunks[0].params
                params['shape'] = (chunks[0].shape[0], op.feature_indices.shape[1])
                params['index'] = (chunks[0].index[0], chunks[0].index[2])
                if isinstance(t_data, DATAFRAME_TYPE):
                    params['index_value'] = parse_index(pd.Int64Index([]), chunks[0].key)
                inputs = chunks.tolist()
                return new_op.new_chunk(inputs, **params)

            chunks_array = np.apply_along_axis(_gen_combine_chunk, 1, map_holder)
            chunks = chunks_array.reshape((chunks_array.size,)).tolist()

        new_op = op.copy().reset_key()
        new_nsplits = (t_data.nsplits[0], (op.feature_indices.shape[1],) * t_feature_idxes.chunk_shape[0])
        return new_op.new_tileables(op.inputs, chunks=chunks,
                                    nsplits=new_nsplits, **t_out.params)

    @classmethod
    def _execute_map(cls, ctx, op: "BaggingSampleReindex"):
        data = ctx[op.inputs[0].key]
        feature_idx = ctx[op.feature_indices.key] - op.start_col_index
        filtered = []
        for row in feature_idx:
            row = row[(row >= 0) & (row < data.shape[1])]
            filtered.append(_get_by_iloc(data, row, axis=1))
        ctx[op.outputs[0].key] = tuple(filtered)

    @classmethod
    def _execute_combine(cls, ctx, op: "BaggingSampleReindex"):
        data_inputs = [ctx[c.key] for c in op.inputs]
        concatenated = []
        for data_input in zip(*data_inputs):
            concatenated.append(_concat_on_axis(data_input, 1, op.inputs[0]))
        ctx[op.outputs[0].key] = tuple(concatenated)

    @classmethod
    def execute(cls, ctx, op: "BaggingSampleReindex"):
        if op.stage == OperandStage.combine:
            cls._execute_combine(ctx, op)
        else:
            cls._execute_map(ctx, op)
