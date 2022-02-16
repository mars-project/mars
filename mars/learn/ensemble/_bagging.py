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

import enum
import itertools
import warnings
from collections import defaultdict
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    clone as clone_estimator,
)
from sklearn.utils import check_random_state as sklearn_check_random_state

from ..utils import column_or_1d, convert_to_tensor_or_dataframe
from ..utils.multiclass import check_classification_targets
from ..utils.validation import check_is_fitted
from ... import opcodes, tensor as mt
from ...core import OutputType, get_output_types, recursive_tile
from ...core.context import Context
from ...core.operand import OperandStage
from ...dataframe.core import DATAFRAME_TYPE
from ...dataframe.utils import parse_index
from ...deploy.oscar.session import execute
from ...serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    Int8Field,
    Int64Field,
    Float32Field,
    TupleField,
    FunctionField,
    ReferenceField,
    FieldTypes,
)
from ...tensor.core import TENSOR_CHUNK_TYPE
from ...tensor.random import RandomStateField
from ...tensor.utils import gen_random_seeds
from ...typing import TileableType
from ...utils import has_unknown_shape
from ..operands import LearnOperand, LearnOperandMixin, LearnShuffleProxy
from ..utils.shuffle import LearnShuffle


def _extract_bagging_io(io_list: Iterable, op: LearnOperand, output: bool = False):
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
    if hasattr(x, "iloc"):
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


def _set_random_states(estimator, random_state=None):
    random_state = sklearn_check_random_state(random_state)
    to_set = {}
    for key in sorted(estimator.get_params(deep=True)):
        if key == "random_state" or key.endswith("__random_state"):
            to_set[key] = random_state.randint(np.iinfo(np.int32).max)

    if to_set:
        estimator.set_params(**to_set)


def _make_estimator(estimator, random_state=None):
    """Make and configure a copy of the `base_estimator_` attribute.

    Warning: This method should be used to properly instantiate new
    sub-estimators.
    """
    estimator = clone_estimator(estimator)
    if random_state is not None:
        _set_random_states(estimator, random_state)
    return estimator


class BaggingSample(LearnShuffle, LearnOperandMixin):
    _op_type_ = opcodes.BAGGING_SHUFFLE_SAMPLE

    n_estimators: int = Int64Field("n_estimators")
    max_samples = AnyField("max_samples")
    max_features = AnyField("max_features")
    bootstrap: bool = BoolField("bootstrap")
    bootstrap_features: bool = BoolField("bootstrap_features")

    random_state = RandomStateField("random_state")
    sample_random_state = RandomStateField("sample_random_state")
    feature_random_state = RandomStateField("feature_random_state")

    reducer_ratio: float = Float32Field("reducer_ratio")
    n_reducers: int = Int64Field("n_reducers", default=None)
    column_offset: int = Int64Field("column_offset", default=None)

    chunk_shape: Tuple[int] = TupleField("chunk_shape", FieldTypes.int64)
    with_labels: bool = BoolField("with_labels")
    with_weights: bool = BoolField("with_weights")
    with_feature_indices: bool = BoolField("with_feature_indices")

    def __init__(
        self,
        max_samples: Union[int, float] = 1.0,
        max_features: Union[int, float] = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        random_state: np.random.RandomState = None,
        reducer_ratio: float = 1.0,
        **kw,
    ):
        super().__init__(
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            max_samples=max_samples,
            max_features=max_features,
            reducer_ratio=reducer_ratio,
            random_state=random_state,
            **kw,
        )
        if self.random_state is None:
            self.random_state = np.random.RandomState()

    @property
    def output_limit(self) -> int:
        if self.stage != OperandStage.map:
            return 1 + self.with_labels + self.with_weights + self.with_feature_indices
        return 1

    def __call__(
        self,
        in_sample: TileableType,
        in_labels: Optional[TileableType] = None,
        in_weights: Optional[TileableType] = None,
    ):
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

        new_shape = tuple(
            s if keep_shape else np.nan
            for s, keep_shape in zip(in_sample.shape, axis_keep_shape)
        )

        kws = []

        data_params = in_sample.params
        data_params["shape"] = new_shape
        kws.append(data_params)

        if in_labels is not None:
            labels_params = in_labels.params
            labels_params["shape"] = (new_shape[0],)
            kws.append(labels_params)

        if in_weights is not None:
            weights_params = in_weights.params
            weights_params["shape"] = (new_shape[0],)
            kws.append(weights_params)

        if self.with_feature_indices:
            feature_params = {
                "shape": (self.n_estimators, new_shape[1]),
                "dtype": np.dtype(int),
            }
            kws.append(feature_params)

        inputs = [in_sample]
        if in_labels is not None:
            inputs.append(in_labels)
        if in_weights is not None:
            inputs.append(in_weights)

        return self.new_tileables(inputs, kws=kws)

    @classmethod
    def _scatter_samples(
        cls,
        max_samples: Union[int, float],
        nsplits: Tuple[int],
        random_state: np.random.RandomState,
        n_estimators: int,
    ) -> np.ndarray:
        nsp_array = np.array(nsplits)
        dim_size = nsp_array.sum()
        if isinstance(max_samples, int):
            expect_sample_count = max_samples
        else:
            expect_sample_count = int(max_samples * nsp_array.sum())

        if expect_sample_count == dim_size:
            return np.array([list(nsplits)] * n_estimators)

        split_probs = nsp_array / dim_size
        return random_state.multinomial(
            expect_sample_count, split_probs, size=n_estimators
        )

    @classmethod
    def tile(cls, op: "BaggingSample"):
        in_sample, in_labels, in_weights, _ = _extract_bagging_io(
            op.inputs, op, output=False
        )
        out_data, out_labels, out_weights, out_feature_indices = _extract_bagging_io(
            op.outputs, op, output=True
        )

        # make sure all shapes are computed
        if (
            has_unknown_shape(in_sample)
            or (in_labels is not None and has_unknown_shape(in_labels))
            or (in_weights is not None and has_unknown_shape(in_weights))
        ):
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
            tiled = yield from recursive_tile(to_tile)
            tiled_iter = iter(tiled)
            if in_labels is not None:
                in_labels = next(tiled_iter)
            if in_weights is not None:
                in_weights = next(tiled_iter)

        random_seeds = [
            gen_random_seeds(n, op.random_state) for n in in_sample.chunk_shape
        ]

        axis_keep_shape = [
            isinstance(op.max_samples, float)
            and op.max_samples == 1.0
            and not op.bootstrap,
            isinstance(op.max_features, float)
            and op.max_features == 1.0
            and not op.bootstrap_features,
        ]

        n_reducers = (
            op.n_reducers
            if op.n_reducers is not None
            else max(1, int(in_sample.chunk_shape[0] * op.reducer_ratio))
        )

        # todo implement sampling without replacements
        map_chunks = []
        max_samples_splits = cls._scatter_samples(
            op.max_samples, in_sample.nsplits[0], op.random_state, op.n_estimators
        )
        max_features_splits = cls._scatter_samples(
            op.max_features, in_sample.nsplits[1], op.random_state, op.n_estimators
        )

        column_cum_offset = np.concatenate([[0], np.cumsum(in_sample.nsplits[1])])
        for chunk in in_sample.chunks:
            new_op = op.copy().reset_key()
            new_op.random_state = None
            new_op.sample_random_state = np.random.RandomState(
                random_seeds[0][chunk.index[0]]
            )
            new_op.feature_random_state = np.random.RandomState(
                random_seeds[1][chunk.index[1]]
            )
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
            params["shape"] = tuple(
                s if keep_shape else np.nan
                for s, keep_shape in zip(chunk.shape, axis_keep_shape)
            )

            input_chunks = [chunk]
            if new_op.with_labels:
                input_chunks.append(in_labels.cix[chunk.index[0]])
            if new_op.with_weights:
                input_chunks.append(in_weights.cix[chunk.index[0]])
            map_chunks.append(new_op.new_chunk(input_chunks, **params))

        shuffle_op = LearnShuffleProxy(output_types=[OutputType.tensor]).new_chunk(
            map_chunks, dtype=np.dtype(int), shape=()
        )

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
            data_params["index"] = (idx, 0)
            data_params["shape"] = (np.nan, out_data.shape[1])
            kws.append(data_params)

            if op.with_labels:
                labels_params = out_labels.params
                labels_params["index"] = (idx,)
                labels_params["shape"] = (np.nan,)
                kws.append(labels_params)

            if op.with_weights:
                weights_params = out_weights.params
                weights_params["index"] = (idx,)
                weights_params["shape"] = (np.nan,)
                kws.append(weights_params)

            if op.with_feature_indices:
                feature_params = {
                    "index": (idx, 0),
                    "shape": (new_op.n_estimators, out_feature_indices.shape[1]),
                    "dtype": np.dtype(int),
                }
                kws.append(feature_params)

            chunks = new_op.new_chunks([shuffle_op], kws=kws)
            (
                data_chunk,
                labels_chunk,
                weights_chunk,
                feature_chunk,
            ) = _extract_bagging_io(chunks, op, output=True)

            reduce_data_chunks.append(data_chunk)
            if labels_chunk is not None:
                reduce_labels_chunks.append(labels_chunk)
            if weights_chunk is not None:
                reduce_weights_chunks.append(weights_chunk)
            if feature_chunk is not None:
                reduce_feature_chunks.append(feature_chunk)

        new_op = op.copy().reset_key()

        kws = [
            {
                "chunks": reduce_data_chunks,
                "nsplits": ((np.nan,) * len(reduce_data_chunks), (out_data.shape[1],)),
                **out_data.params,
            }
        ]
        if op.with_labels:
            kws.append(
                {
                    "chunks": reduce_labels_chunks,
                    "nsplits": ((np.nan,) * len(reduce_data_chunks),),
                    **out_labels.params,
                }
            )
        if op.with_weights:
            kws.append(
                {
                    "chunks": reduce_weights_chunks,
                    "nsplits": ((np.nan,) * len(reduce_data_chunks),),
                    **out_weights.params,
                }
            )
        if op.with_feature_indices:
            estimator_nsplit = tuple(c.op.n_estimators for c in reduce_data_chunks)
            kws.append(
                {
                    "chunks": reduce_feature_chunks,
                    "nsplits": (estimator_nsplit, (out_feature_indices.shape[1],)),
                    **out_feature_indices.params,
                }
            )
        return new_op.new_tileables(op.inputs, kws=kws)

    @classmethod
    def _gen_sample_indices(
        cls,
        max_range: int,
        size: int,
        random_state: np.random.RandomState,
        with_replacement: bool = False,
    ):
        if not with_replacement:
            result = random_state.choice(np.arange(max_range), size, False)
        else:
            result = random_state.randint(0, max_range - 1, size)
        result.sort()
        return result

    @classmethod
    def _execute_map(cls, ctx, op: "BaggingSample"):
        in_sample, in_labels, in_weights, _ = _extract_bagging_io(
            op.inputs, op, output=False
        )
        in_sample_data = ctx[in_sample.key]
        in_labels_data = ctx[in_labels.key] if op.with_labels else None
        in_weights_data = ctx[in_weights.key] if op.with_weights else None
        out_samples = op.outputs[0]

        remains = op.n_estimators % op.n_reducers
        reducer_iters = [
            itertools.repeat(idx, 1 + op.n_estimators // op.n_reducers)
            for idx in range(remains)
        ]
        reducer_iters += [
            itertools.repeat(idx, op.n_estimators // op.n_reducers)
            for idx in range(remains, op.n_reducers)
        ]
        reducer_iter = itertools.chain(*reducer_iters)

        result_store = defaultdict(lambda: ([], [], [], []))
        for est_id in range(op.n_estimators):
            sampled_data = in_sample_data
            sampled_labels = in_labels_data
            sampled_weights = in_weights_data

            if op.max_samples[est_id] != in_sample_data.shape[0]:
                sample_indices = cls._gen_sample_indices(
                    in_sample_data.shape[0],
                    op.max_samples[est_id],
                    op.sample_random_state,
                    op.bootstrap,
                )

                sampled_data = _get_by_iloc(sampled_data, sample_indices)
                if sampled_labels is not None:
                    sampled_labels = _get_by_iloc(sampled_labels, sample_indices)
                if sampled_weights is not None:
                    sampled_weights = _get_by_iloc(sampled_weights, sample_indices)

            if op.max_features[est_id] != in_sample_data.shape[1]:
                feature_indices = cls._gen_sample_indices(
                    in_sample_data.shape[1],
                    op.max_features[est_id],
                    op.feature_random_state,
                    op.bootstrap_features,
                )

                sampled_data = _get_by_iloc(sampled_data, feature_indices, axis=1)
                if not op.with_feature_indices:
                    feature_indices = None
            else:
                feature_indices = None

            samples, labels, weights, feature_idx_array = result_store[
                next(reducer_iter)
            ]
            samples.append(sampled_data)
            if sampled_labels is not None:
                labels.append(sampled_labels)
            if sampled_weights is not None:
                weights.append(sampled_weights)
            if feature_indices is not None:
                feature_idx_array.append(feature_indices + op.column_offset)

        for (
            reducer_id,
            (
                samples,
                labels,
                weights,
                feature_idx_array,
            ),
        ) in result_store.items():
            ctx[out_samples.key, (reducer_id, 0)] = tuple(
                samples + labels + weights + feature_idx_array
            )

    @classmethod
    def _execute_reduce(cls, ctx, op: "BaggingSample"):
        out_data, out_labels, out_weights, out_feature_indices = _extract_bagging_io(
            op.outputs, op, output=True
        )

        input_keys = op.inputs[0].op.source_keys
        input_idxes = op.inputs[0].op.source_idxes

        sample_holder = [
            np.empty(op.chunk_shape, dtype=object) for _ in range(op.n_estimators)
        ]

        labels_holder = (
            [np.empty(op.chunk_shape[0], dtype=object) for _ in range(op.n_estimators)]
            if op.with_labels
            else None
        )

        weights_holder = (
            [np.empty(op.chunk_shape[0], dtype=object) for _ in range(op.n_estimators)]
            if op.with_weights
            else None
        )

        feature_indices_holder = (
            [np.empty(op.chunk_shape[1], dtype=object) for _ in range(op.n_estimators)]
            if op.with_feature_indices
            else None
        )

        for input_key, input_idx in zip(input_keys, input_idxes):
            add_feature_index = input_idx[0] == 0
            add_label_weight = input_idx[1] == op.chunk_shape[1] - 1
            chunk_data = ctx[input_key, out_data.index]

            num_groups = 1
            if add_feature_index and op.with_feature_indices:
                # contains feature indices
                num_groups += 1
            if add_label_weight:  # contains label or weight
                num_groups += int(op.with_weights) + int(op.with_labels)

            sample_count = len(chunk_data) // num_groups
            assert len(chunk_data) % num_groups == 0

            group_iter = (
                chunk_data[i * sample_count : (i + 1) * sample_count]
                for i in range(num_groups)
            )

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
            row_chunks = np.apply_along_axis(
                _concat_by_row, axis=0, arr=sample_mat, out_chunk=out_data
            )
            data_results[est_idx] = _concat_on_axis(
                row_chunks[0].tolist(), axis=1, out_chunk=out_data
            )
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
    def execute(cls, ctx, op: "BaggingSample"):
        if op.stage == OperandStage.map:
            cls._execute_map(ctx, op)
        else:
            cls._execute_reduce(ctx, op)


class BaggingSampleReindex(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.BAGGING_SHUFFLE_REINDEX

    n_estimators: int = Int64Field("n_estimators")
    feature_indices: TileableType = ReferenceField("feature_indices", default=None)

    start_col_index: int = Int64Field("start_col_index", 0)

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
            params["shape"] = (data.shape[0], np.nan)
        if isinstance(data, DATAFRAME_TYPE):
            params["index_value"] = parse_index(pd.Index([], dtype=np.int64), data.key)
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
        map_holder = np.empty(
            t_data.chunk_shape + (t_feature_idxes.chunk_shape[0],),
            dtype=np.dtype(object),
        )
        for chunk in t_data.chunks:
            for feature_idx_chunk in t_feature_idxes.chunks:
                new_op = op.copy().reset_key()
                new_op.stage = OperandStage.map
                new_op.start_col_index = int(cum_nsplits[chunk.index[1]])
                params = chunk.params
                new_index = params["index"] = chunk.index + (
                    feature_idx_chunk.index[0],
                )
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
                params["shape"] = (chunks[0].shape[0], op.feature_indices.shape[1])
                params["index"] = (chunks[0].index[0], chunks[0].index[2])
                if isinstance(t_data, DATAFRAME_TYPE):
                    params["index_value"] = parse_index(
                        pd.Index([], dtype=np.int64), chunks[0].key
                    )
                inputs = chunks.tolist()
                return new_op.new_chunk(inputs, **params)

            chunks_array = np.apply_along_axis(_gen_combine_chunk, 1, map_holder)
            chunks = chunks_array.reshape((chunks_array.size,)).tolist()

        new_op = op.copy().reset_key()
        new_nsplits = (
            t_data.nsplits[0],
            (op.feature_indices.shape[1],) * t_feature_idxes.chunk_shape[0],
        )
        return new_op.new_tileables(
            op.inputs, chunks=chunks, nsplits=new_nsplits, **t_out.params
        )

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


class BaggingFitOperand(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.BAGGING_FIT

    base_estimator: BaseEstimator = AnyField("base_estimator")
    estimator_params: dict = DictField("estimator_params", default=None)
    n_estimators: int = Int64Field("n_estimators")
    max_samples = AnyField("max_samples", default=1.0)
    max_features = AnyField("max_features", default=1.0)
    bootstrap: bool = BoolField("bootstrap", default=False)
    bootstrap_features: bool = BoolField("bootstrap_features", default=True)
    random_state = RandomStateField("random_state", default=None)

    reducer_ratio: float = Float32Field("reducer_ratio")
    n_reducers: int = Int64Field("n_reducers")

    labels: TileableType = ReferenceField("labels", default=None)
    weights: TileableType = ReferenceField("weights", default=None)
    feature_indices: TileableType = ReferenceField("feature_indices", default=None)
    with_feature_indices: bool = BoolField("with_feature_indices", default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.random_state is None:
            self.random_state = np.random.RandomState()
        if self.with_feature_indices is None:
            full_features = (
                isinstance(self.max_features, float) and self.max_features == 1.0
            )
            self.with_feature_indices = not full_features or self.bootstrap_features

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)

        input_iter = iter(inputs)
        next(input_iter)
        if self.labels is not None:
            self.labels = next(input_iter)
        if self.weights is not None:
            self.weights = next(input_iter)
        if self.feature_indices is not None:
            self.feature_indices = next(input_iter)

    def _get_bagging_sample_tileables(self, samples=None):
        samples = samples or self.inputs[0]
        sample_op = BaggingSample(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            bootstrap_features=self.bootstrap_features,
            random_state=self.random_state,
            reducer_ratio=self.reducer_ratio,
            n_reducers=self.n_reducers,
            with_weights=self.weights is not None,
            with_labels=self.labels is not None,
            with_feature_indices=self.with_feature_indices,
        )
        return _extract_bagging_io(
            sample_op(samples, self.labels, self.weights), sample_op, output=True
        )

    @property
    def output_limit(self) -> int:
        if self.with_feature_indices:
            return 2
        return 1

    def __call__(
        self,
        in_data: TileableType,
        in_labels: Optional[TileableType] = None,
        in_weights: Optional[TileableType] = None,
        feature_indices: TileableType = None,
    ):
        self._output_types = [OutputType.tensor]
        inputs = [in_data]

        if in_labels is not None:
            self.labels = in_labels
            inputs.append(in_labels)
        if in_weights is not None:
            self.weights = in_weights
            inputs.append(in_weights)

        if feature_indices is not None:
            self.feature_indices = feature_indices
            inputs.append(feature_indices)

        kws = [dict(shape=(self.n_estimators,), dtype=np.dtype(object))]
        if self.with_feature_indices:
            self._output_types.append(OutputType.tensor)
            sample_tileables = self._get_bagging_sample_tileables(in_data)
            kws.append(sample_tileables[-1].params)

        return self.new_tileables(inputs, kws=kws)

    @classmethod
    def tile(cls, op: "BaggingFitOperand"):
        out = op.outputs[0]
        sample_tileables = op._get_bagging_sample_tileables()
        tiled_sample_iter = iter(
            (
                yield from recursive_tile(
                    tuple(t for t in sample_tileables if t is not None)
                )
            )
        )
        sampled, labels, weights, feature_indices = (
            t if t is None else next(tiled_sample_iter) for t in sample_tileables
        )

        estimator_nsplits = (tuple(c.op.n_estimators for c in sampled.chunks),)

        label_chunks = itertools.repeat(None) if labels is None else labels.chunks
        weight_chunks = itertools.repeat(None) if weights is None else weights.chunks

        out_chunks = []
        seeds = gen_random_seeds(len(sampled.chunks), op.random_state)
        for sample_chunk, label_chunk, weight_chunk, n_estimators in zip(
            sampled.chunks, label_chunks, weight_chunks, estimator_nsplits[0]
        ):
            chunk_op = BaggingFitOperand(
                base_estimator=op.base_estimator,
                estimator_params=op.estimator_params,
                labels=label_chunk,
                weights=weight_chunk,
                n_estimators=n_estimators,
                with_feature_indices=False,
                random_state=sklearn_check_random_state(seeds[sample_chunk.index[0]]),
            )
            chunk_op._output_types = op._output_types
            inputs = [
                c for c in [sample_chunk, label_chunk, weight_chunk] if c is not None
            ]
            out_chunks.append(
                chunk_op.new_chunk(
                    inputs,
                    index=(sample_chunk.index[0],),
                    shape=(n_estimators,),
                    dtype=out.dtype,
                )
            )

        out_op = op.copy().reset_key()
        kws = [
            dict(chunks=out_chunks, nsplits=estimator_nsplits, **out.params),
        ]
        if feature_indices is not None:
            kws.append(
                dict(
                    chunks=feature_indices.chunks,
                    nsplits=feature_indices.nsplits,
                    **feature_indices.params,
                )
            )
        return out_op.new_tileables(op.inputs, kws=kws, output_limit=len(kws))

    @classmethod
    def execute(cls, ctx: Union[dict, Context], op: "BaggingFitOperand"):
        sampled_data = ctx[op.inputs[0].key]
        labels_data = (
            ctx[op.labels.key] if op.labels is not None else itertools.repeat(None)
        )
        weights_data = (
            ctx[op.weights.key] if op.weights is not None else itertools.repeat(None)
        )

        for k, v in (op.estimator_params or dict()).items():
            setattr(op.base_estimator, k, v)

        new_estimators = []
        seeds = gen_random_seeds(len(sampled_data), op.random_state)
        for idx, (sampled, label, weights) in enumerate(
            zip(sampled_data, labels_data, weights_data)
        ):
            estimator = _make_estimator(op.base_estimator, seeds[idx])
            estimator.fit(sampled, y=label, sample_weight=weights)
            new_estimators.append(estimator)
        ctx[op.outputs[0].key] = np.array(new_estimators, dtype=np.dtype(object))


class PredictionType(enum.Enum):
    REGRESSION = 0
    PROBABILITY = 1
    LOG_PROBABILITY = 2
    DECISION_FUNCTION = 3


class BaggingPredictionOperand(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.BAGGING_PREDICTION

    estimators: TileableType = ReferenceField("estimators")
    feature_indices: TileableType = ReferenceField("feature_indices", default=None)
    n_classes: Optional[int] = Int64Field("n_classes", default=None)
    prediction_type: PredictionType = Int8Field(
        "prediction_type",
        on_serialize=lambda x: x.value,
        on_deserialize=PredictionType,
        default=PredictionType.PROBABILITY,
    )
    decision_function: Callable = FunctionField("decision_function", default=None)
    calc_means: bool = BoolField("calc_means", default=True)

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        input_iter = iter(inputs[1:])
        self.estimators = next(input_iter)
        if self.feature_indices is not None:
            self.feature_indices = next(input_iter)

    def __call__(
        self,
        instances: TileableType,
        estimators: TileableType,
        feature_indices: TileableType = None,
    ) -> TileableType:
        self._output_types = [OutputType.tensor]
        self.estimators = estimators
        self.feature_indices = feature_indices

        if self.n_classes is not None:
            shape = (instances.shape[0], estimators.shape[0], self.n_classes)
        else:
            shape = (instances.shape[0], estimators.shape[0])
        if self.calc_means:
            shape = (shape[0],) + shape[2:]

        params = {"dtype": np.dtype(float), "shape": shape}
        inputs = [instances, estimators]
        if feature_indices is not None:
            inputs.append(feature_indices)
        return self.new_tileable(inputs, **params)

    def _get_class_shape(self):
        if self.n_classes and self.n_classes > 2:
            return self.n_classes
        elif self.prediction_type == PredictionType.DECISION_FUNCTION:
            return None
        else:
            return self.n_classes

    @classmethod
    def _build_chunks_without_feature_indices(
        cls, op: "BaggingPredictionOperand", t_instances: TileableType
    ):
        class_shape = op._get_class_shape()
        chunks = []
        for c_instance in t_instances.chunks:
            for c_estimator in op.estimators.chunks:
                if class_shape is not None:
                    params = {
                        "dtype": np.dtype(float),
                        "shape": (
                            c_instance.shape[0],
                            class_shape,
                            c_estimator.shape[0],
                        ),
                        "index": (c_instance.index[0], 0, c_estimator.index[0]),
                    }
                else:
                    params = {
                        "dtype": np.dtype(float),
                        "shape": (c_instance.shape[0], c_estimator.shape[0]),
                        "index": (c_instance.index[0], c_estimator.index[0]),
                    }
                new_op = op.copy().reset_key()
                new_op.feature_indices = None
                chunks.append(new_op.new_chunk([c_instance, c_estimator], **params))
        return chunks

    @classmethod
    def _build_chunks_with_feature_indices(
        cls, op: "BaggingPredictionOperand", t_instances: TileableType
    ):
        class_shape = op._get_class_shape()
        chunks = []
        for c in t_instances.chunks:
            estimator_chunk = op.estimators.chunks[c.index[1]]

            if class_shape is not None:
                params = {
                    "dtype": np.dtype(float),
                    "shape": (c.shape[0], class_shape, estimator_chunk.shape[0]),
                    "index": (c.index[0], 0, c.index[1]),
                }
            else:
                params = {
                    "dtype": np.dtype(float),
                    "shape": (c.shape[0], estimator_chunk.shape[0]),
                    "index": c.index,
                }

            new_op = op.copy().reset_key()
            new_op.feature_indices = None
            chunks.append(new_op.new_chunk([c, estimator_chunk], **params))
        return chunks

    @classmethod
    def tile(cls, op: "BaggingPredictionOperand"):
        n_estimators = op.estimators.shape[0]
        reindex_op = BaggingSampleReindex(n_estimators=n_estimators)
        t_instances = yield from recursive_tile(
            reindex_op(op.inputs[0], op.feature_indices)
        )

        # for classifiers, form instance-class-estimator array
        # for regressors, form instance-estimator array
        # and then sum over estimator axis

        if op.feature_indices is None:
            chunks = cls._build_chunks_without_feature_indices(op, t_instances)
        else:
            chunks = cls._build_chunks_with_feature_indices(op, t_instances)

        new_op = op.copy().reset_key()
        class_shape = op._get_class_shape()
        if class_shape is not None:
            params = {
                "dtype": np.dtype(float),
                "shape": (t_instances.shape[0], class_shape, n_estimators),
            }
            nsplits = (t_instances.nsplits[0], (class_shape,), op.estimators.nsplits[0])
        else:
            params = {
                "dtype": np.dtype(float),
                "shape": (t_instances.shape[0], n_estimators),
            }
            nsplits = (t_instances.nsplits[0], op.estimators.nsplits[0])
        estimator_probas = new_op.new_tileable(
            op.inputs, chunks=chunks, nsplits=nsplits, **params
        )

        if not op.calc_means:
            return estimator_probas
        elif op.prediction_type != PredictionType.LOG_PROBABILITY:
            return [
                (
                    yield from recursive_tile(
                        mt.sum(estimator_probas, axis=-1) / n_estimators
                    )
                )
            ]
        else:
            return [
                (
                    yield from recursive_tile(
                        mt.log(mt.exp(estimator_probas).sum(axis=-1))
                        - np.log(n_estimators)
                    )
                )
            ]

    @classmethod
    def _predict_proba(cls, instance, estimator, n_classes):
        n_samples = instance.shape[0]
        proba = np.zeros((n_samples, n_classes))

        if hasattr(estimator, "predict_proba"):
            proba_estimator = estimator.predict_proba(instance)
            if n_classes == len(estimator.classes_):
                proba += proba_estimator

            else:
                proba[:, estimator.classes_] += proba_estimator[
                    :, range(len(estimator.classes_))
                ]
        else:
            # Resort to voting
            predictions = estimator.predict(instance)
            for i in range(n_samples):
                proba[i, predictions[i]] += 1
        return proba

    @classmethod
    def _predict_log_proba(cls, instance, estimator, n_classes):
        """Private function used to compute log probabilities within a job."""
        if not hasattr(estimator, "predict_log_proba"):
            return np.log(cls._predict_proba(instance, estimator, n_classes))

        n_samples = instance.shape[0]
        log_proba = np.empty((n_samples, n_classes))
        log_proba.fill(-np.inf)
        all_classes = np.arange(n_classes, dtype=int)

        log_proba_estimator = estimator.predict_log_proba(instance)

        if n_classes == len(estimator.classes_):
            log_proba = np.logaddexp(log_proba, log_proba_estimator)
        else:  # pragma: no cover
            log_proba[:, estimator.classes_] = np.logaddexp(
                log_proba[:, estimator.classes_],
                log_proba_estimator[:, range(len(estimator.classes_))],
            )
            missing = np.setdiff1d(all_classes, estimator.classes_)
            log_proba[:, missing] = np.logaddexp(log_proba[:, missing], -np.inf)
        return log_proba

    @classmethod
    def _decision_function(cls, instance, estimator, func=None):
        if func is not None:
            return func(instance, estimator)
        else:
            return estimator.decision_function(instance)

    @classmethod
    def execute(cls, ctx: Union[dict, Context], op: "BaggingPredictionOperand"):
        instances = ctx[op.inputs[0].key]
        estimators = ctx[op.estimators.key]
        if not isinstance(instances, tuple):
            instances = [instances] * len(estimators)

        estimate_results = []
        for instance, estimator in zip(instances, estimators):
            # classifier
            if op.prediction_type == PredictionType.PROBABILITY:
                estimate_results.append(
                    cls._predict_proba(instance, estimator, op.n_classes)
                )
            elif op.prediction_type == PredictionType.LOG_PROBABILITY:
                estimate_results.append(
                    cls._predict_log_proba(instance, estimator, op.n_classes)
                )
            elif op.prediction_type == PredictionType.DECISION_FUNCTION:
                estimate_results.append(
                    cls._decision_function(instance, estimator, op.decision_function)
                )
            else:
                estimate_results.append(estimator.predict(instance))

        out = op.outputs[0]
        ctx[out.key] = np.stack(estimate_results, axis=out.ndim - 1)


class BaseBagging:
    def __init__(
        self,
        base_estimator=None,
        n_estimators=10,
        *,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        reducers=1.0,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators

        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.random_state = (
            np.random.RandomState(random_state)
            if isinstance(random_state, int)
            else random_state
        )
        self.verbose = verbose
        self.reducers = reducers

        self.estimators_ = None
        self.estimator_features_ = None

    def _validate_y(self, y, session=None, run_kwargs=None):
        if len(y.shape) == 1 or y.shape[1] == 1:
            return column_or_1d(y, warn=True)
        else:
            return y

    def _fit(
        self,
        X,
        y=None,
        sample_weight=None,
        max_samples=None,
        estimator_params=None,
        session=None,
        run_kwargs=None,
    ):
        estimator_features, feature_indices = None, None
        n_more_estimators = self.n_estimators

        X = convert_to_tensor_or_dataframe(X)
        y = convert_to_tensor_or_dataframe(y) if y is not None else None
        sample_weight = (
            convert_to_tensor_or_dataframe(sample_weight)
            if sample_weight is not None
            else None
        )

        y = self._validate_y(y)

        if self.warm_start:
            feature_indices = self.estimator_features_
            if self.estimators_ is not None:
                exist_estimators = self.estimators_.shape[0]
                # move random states to skip duplicated results
                self.random_state.rand(exist_estimators)
                n_more_estimators = self.n_estimators - exist_estimators

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, self.estimators_.shape[0])
            )
        elif n_more_estimators == 0:
            warnings.warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
            return self

        fit_op = BaggingFitOperand(
            base_estimator=self.base_estimator,
            estimator_params=estimator_params,
            n_estimators=n_more_estimators,
            max_samples=max_samples or self.max_samples,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            bootstrap_features=self.bootstrap_features,
            random_state=self.random_state,
            reducer_ratio=self.reducers if isinstance(self.reducers, float) else None,
            n_reducers=self.reducers if isinstance(self.reducers, int) else None,
        )
        tileables = fit_op(X, y, sample_weight, feature_indices)
        ret = execute(*tileables, session=session, **(run_kwargs or dict()))

        if len(ret) == 2:
            estimators, estimator_features = ret
        else:
            estimators = ret

        if self.estimators_ is not None:
            estimators = mt.concatenate([self.estimators_, estimators])
        if self.estimator_features_ is not None:
            estimator_features = mt.concatenate(
                [self.estimator_features_, estimator_features]
            )

        self.estimators_, self.estimator_features_ = estimators, estimator_features
        return self

    def fit(self, X, y=None, sample_weight=None, session=None, run_kwargs=None):
        """
        Build a Bagging ensemble of estimators from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        return self._fit(
            X, y, sample_weight=sample_weight, session=session, run_kwargs=run_kwargs
        )


class BaggingClassifier(ClassifierMixin, BaseBagging):
    """
    A Bagging classifier.

    A Bagging classifier is an ensemble meta-estimator that fits base
    classifiers each on random subsets of the original dataset and then
    aggregate their individual predictions (either by voting or by averaging)
    to form a final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a black-box estimator (e.g., a decision
    tree), by introducing randomization into its construction procedure and
    then making an ensemble out of it.

    This algorithm encompasses several works from the literature. When random
    subsets of the dataset are drawn as random subsets of the samples, then
    this algorithm is known as Pasting [1]_. If samples are drawn with
    replacement, then the method is known as Bagging [2]_. When random subsets
    of the dataset are drawn as random subsets of the features, then the method
    is known as Random Subspaces [3]_. Finally, when base estimators are built
    on subsets of both samples and features, then the method is known as
    Random Patches [4]_.

    Read more in the :ref:`User Guide <bagging>`.

    Parameters
    ----------
    base_estimator : object, default=None
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a
        :class:`~sklearn.tree.DecisionTreeClassifier`.

    n_estimators : int, default=10
        The number of base estimators in the ensemble.

    max_samples : int or float, default=1.0
        The number of samples to draw from X to train each base estimator (with
        replacement by default, see `bootstrap` for more details).

        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator (
        without replacement by default, see `bootstrap_features` for more
        details).

        - If int, then draw `max_features` features.
        - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : bool, default=True
        Whether samples are drawn with replacement. If False, sampling
        without replacement is performed.

    bootstrap_features : bool, default=False
        Whether features are drawn with replacement.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble. See :term:`the Glossary <warm_start>`.

    random_state : int, RandomState instance or None, default=None
        Controls the random resampling of the original dataset
        (sample wise and feature wise).
        If the base estimator accepts a `random_state` attribute, a different
        seed is generated for each instance in the ensemble.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimators
        The collection of fitted base estimators.

    estimators_features_ : list of arrays
        The subset of drawn features for each base estimator.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int or list
        The number of classes.

    See Also
    --------
    BaggingRegressor : A Bagging regressor.

    References
    ----------

    .. [1] L. Breiman, "Pasting small votes for classification in large
           databases and on-line", Machine Learning, 36(1), 85-103, 1999.

    .. [2] L. Breiman, "Bagging predictors", Machine Learning, 24(2), 123-140,
           1996.

    .. [3] T. Ho, "The random subspace method for constructing decision
           forests", Pattern Analysis and Machine Intelligence, 20(8), 832-844,
           1998.

    .. [4] G. Louppe and P. Geurts, "Ensembles on Random Patches", Machine
           Learning and Knowledge Discovery in Databases, 346-361, 2012.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from mars.learn.ensemble import BaggingClassifier
    >>> from mars.learn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = BaggingClassifier(base_estimator=SVC(),
    ...                         n_estimators=10, random_state=0).fit(X, y)
    >>> clf.predict([[0, 0, 0, 0]])
    array([1])
    """

    def _validate_y(self, y, session=None, run_kwargs=None):
        to_run = [check_classification_targets(y)]
        y = column_or_1d(y, warn=True)
        to_run.extend(mt.unique(y, return_inverse=True))
        _, self.classes_, y = execute(
            *to_run, session=session, **(run_kwargs or dict())
        )
        self.n_classes_ = len(self.classes_)

        return y

    def _predict_proba(self, X):
        check_is_fitted(self)
        X = convert_to_tensor_or_dataframe(X)
        predict_op = BaggingPredictionOperand(
            n_classes=self.n_classes_,
            prediction_type=PredictionType.PROBABILITY,
        )
        return predict_op(X, self.estimators_, self.estimator_features_)

    def predict(self, X, session=None, run_kwargs=None):
        """
        Predict class for X.

        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability. If base estimators do not
        implement a ``predict_proba`` method, then it resorts to voting.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        probas = self._predict_proba(X)
        y = self.classes_.take(mt.argmax(probas, axis=1), axis=0)
        return execute(y, session=session, **(run_kwargs or dict()))

    def predict_proba(self, X, session=None, run_kwargs=None):
        """
        Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the base estimators in the
        ensemble. If base estimators do not implement a ``predict_proba``
        method, then it resorts to voting and the predicted class probabilities
        of an input sample represents the proportion of estimators predicting
        each class.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        probas = self._predict_proba(X)
        return execute(probas, session=session, **(run_kwargs or dict()))

    def predict_log_proba(self, X, session=None, run_kwargs=None):
        """
        Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the log of the mean predicted class probabilities of the base
        estimators in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        X = convert_to_tensor_or_dataframe(X)
        predict_op = BaggingPredictionOperand(
            n_classes=self.n_classes_,
            prediction_type=PredictionType.LOG_PROBABILITY,
        )
        probas = predict_op(X, self.estimators_, self.estimator_features_)
        return execute(probas, session=session, **(run_kwargs or dict()))

    def decision_function(self, X, session=None, run_kwargs=None):
        """
        Average of the decision functions of the base classifiers.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        score : ndarray of shape (n_samples, k)
            The decision function of the input samples. The columns correspond
            to the classes in sorted order, as they appear in the attribute
            ``classes_``. Regression and binary classification are special
            cases with ``k == 1``, otherwise ``k==n_classes``.
        """
        check_is_fitted(self)
        X = convert_to_tensor_or_dataframe(X)
        predict_op = BaggingPredictionOperand(
            n_classes=self.n_classes_,
            prediction_type=PredictionType.DECISION_FUNCTION,
        )
        result = predict_op(X, self.estimators_, self.estimator_features_)
        return execute(result, session=session, **(run_kwargs or dict()))


class BaggingRegressor(RegressorMixin, BaseBagging):
    """
    A Bagging regressor.

    A Bagging regressor is an ensemble meta-estimator that fits base
    regressors each on random subsets of the original dataset and then
    aggregate their individual predictions (either by voting or by averaging)
    to form a final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a black-box estimator (e.g., a decision
    tree), by introducing randomization into its construction procedure and
    then making an ensemble out of it.

    This algorithm encompasses several works from the literature. When random
    subsets of the dataset are drawn as random subsets of the samples, then
    this algorithm is known as Pasting [1]_. If samples are drawn with
    replacement, then the method is known as Bagging [2]_. When random subsets
    of the dataset are drawn as random subsets of the features, then the method
    is known as Random Subspaces [3]_. Finally, when base estimators are built
    on subsets of both samples and features, then the method is known as
    Random Patches [4]_.

    Read more in the :ref:`User Guide <bagging>`.

    Parameters
    ----------
    base_estimator : object, default=None
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a
        :class:`~sklearn.tree.DecisionTreeRegressor`.

    n_estimators : int, default=10
        The number of base estimators in the ensemble.

    max_samples : int or float, default=1.0
        The number of samples to draw from X to train each base estimator (with
        replacement by default, see `bootstrap` for more details).

        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator (
        without replacement by default, see `bootstrap_features` for more
        details).

        - If int, then draw `max_features` features.
        - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : bool, default=True
        Whether samples are drawn with replacement. If False, sampling
        without replacement is performed.

    bootstrap_features : bool, default=False
        Whether features are drawn with replacement.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble. See :term:`the Glossary <warm_start>`.

    random_state : int, RandomState instance or None, default=None
        Controls the random resampling of the original dataset
        (sample wise and feature wise).
        If the base estimator accepts a `random_state` attribute, a different
        seed is generated for each instance in the ensemble.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimators
        The collection of fitted sub-estimators.

    estimators_features_ : list of arrays
        The subset of drawn features for each base estimator.

    See Also
    --------
    BaggingClassifier : A Bagging classifier.

    References
    ----------

    .. [1] L. Breiman, "Pasting small votes for classification in large
           databases and on-line", Machine Learning, 36(1), 85-103, 1999.

    .. [2] L. Breiman, "Bagging predictors", Machine Learning, 24(2), 123-140,
           1996.

    .. [3] T. Ho, "The random subspace method for constructing decision
           forests", Pattern Analysis and Machine Intelligence, 20(8), 832-844,
           1998.

    .. [4] G. Louppe and P. Geurts, "Ensembles on Random Patches", Machine
           Learning and Knowledge Discovery in Databases, 346-361, 2012.

    Examples
    --------
    >>> from sklearn.svm import SVR
    >>> from mars.learn.ensemble import BaggingRegressor
    >>> from mars.learn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=4,
    ...                        n_informative=2, n_targets=1,
    ...                        random_state=0, shuffle=False)
    >>> regr = BaggingRegressor(base_estimator=SVR(),
    ...                         n_estimators=10, random_state=0).fit(X, y)
    >>> regr.predict([[0, 0, 0, 0]])
    array([-2.8720...])
    """

    def predict(self, X, session=None, run_kwargs=None):
        """
        Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the estimators in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self)
        X = convert_to_tensor_or_dataframe(X)
        predict_op = BaggingPredictionOperand(
            prediction_type=PredictionType.REGRESSION,
        )
        probas = predict_op(X, self.estimators_, self.estimator_features_)
        return execute(probas, session=session, **(run_kwargs or dict()))
