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

from ... import opcodes
from ... import tensor as mt
from ...serialize import KeyField
from ...tensor.array_utils import as_same_device, device, sparse
from ...tensor.core import TensorOrder
from ...tiles import TilesError
from ...utils import check_chunks_unknown_shape
from ..operands import LearnOperand, LearnOperandMixin, OutputType
from ._k_means_fast import _inertia_dense, _inertia_sparse, merge_update_chunks


class KMeansInertia(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.KMEANS_INERTIA

    _x = KeyField('x')
    _sample_weight = KeyField('sample_weight')
    _centers = KeyField('centers')
    _labels = KeyField('labels')

    def __init__(self, x=None, sample_weight=None, centers=None,
                 labels=None, output_types=None, **kw):
        super().__init__(_x=x, _sample_weight=sample_weight,
                         _centers=centers, _labels=labels,
                         _output_types=output_types, **kw)
        if self._output_types is None:
            self._output_types = [OutputType.tensor]

    @property
    def x(self):
        return self._x

    @property
    def sample_weight(self):
        return self._sample_weight

    @property
    def centers(self):
        return self._centers

    @property
    def labels(self):
        return self._labels

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs)
        for field in ('_x', '_sample_weight', '_centers', '_labels'):
            if getattr(self, field, None) is not None:
                setattr(self, field, next(inputs_iter))

    def __call__(self):
        params = {
            'shape': (),
            'dtype': np.dtype(float),
            'order': TensorOrder.C_ORDER
        }
        return self.new_tileable([self._x, self._sample_weight,
                                  self._centers, self._labels], kws=[params])

    @classmethod
    def tile(cls, op: "KMeansInertia"):
        check_chunks_unknown_shape(op.inputs, TilesError)
        x = op.x
        x = x.rechunk({1: x.shape[1]})._inplace_tile()
        sample_weight = op.sample_weight.rechunk({0: x.nsplits[0]})._inplace_tile()
        labels = op.labels.rechunk({0: x.nsplits[0]})._inplace_tile()
        centers = op.centers
        centers = centers.rechunk(centers.shape)._inplace_tile()

        out_chunks = []
        for x_chunk, sample_weight_chunk, labels_chunk \
                in zip(x.chunks, sample_weight.chunks, labels.chunks):
            chunk_op = op.copy().reset_key()
            chunk_params = {
                'shape': (1,),
                'dtype': np.dtype(float),
                'order': TensorOrder.C_ORDER,
                'index': x_chunk.index
            }
            out_chunk = chunk_op.new_chunk(
                [x_chunk, sample_weight_chunk, centers.chunks[0],
                 labels_chunk], kws=[chunk_params])
            out_chunks.append(out_chunk)

        new_op = op.copy()
        params = op.outputs[0].params
        params['shape'] = (x.chunk_shape[0],)
        params['chunks'] = out_chunks
        params['nsplits'] = ((1,) * x.chunk_shape[0],)
        out = new_op.new_tileable(op.inputs, kws=[params]).sum()
        out._inplace_tile()
        return [out]

    @classmethod
    def execute(cls, ctx, op):
        (x, sample_weight, centers, labels), device_id, xp = \
            as_same_device([ctx[inp.key] for inp in op.inputs], device=op.device,
                           ret_extra=True, copy_if_not_writeable=True)

        with device(device_id):
            if xp is np:
                method = _inertia_dense
            elif xp is sparse:
                method = _inertia_sparse
            else:  # pragma: no cover
                raise NotImplementedError('Cannot run inertial on GPU')

            result = method(x, sample_weight, centers, labels)
            ctx[op.outputs[0].key] = np.array([result])


def _inertia(X, sample_weight, centers, labels):
    op = KMeansInertia(x=X, sample_weight=sample_weight,
                       centers=centers, labels=labels)
    return op()


def _execute_merge_update(ctx, op):
    inputs, device_id, xp = as_same_device(
        [ctx[inp.key] for inp in op.inputs], op.device,
        ret_extra=True, copy_if_not_writeable=True)
    length = len(inputs) // 2
    assert len(inputs) % 2 == 0
    centers_new_chunks = inputs[:length]
    weight_in_cluster_chunks = inputs[length:]

    with device(device_id):
        weight_in_clusters = np.zeros(op.n_clusters,
                                      dtype=weight_in_cluster_chunks[0].dtype)
        centers_new = np.zeros_like(centers_new_chunks[0])
        n_clusters = op.n_clusters
        n_features = centers_new_chunks[0].shape[1]

        for weight_in_clusters_chunk, centers_new_chunk in \
                zip(weight_in_cluster_chunks, centers_new_chunks):
            merge_update_chunks(n_clusters, n_features,
                                weight_in_clusters, weight_in_clusters_chunk,
                                centers_new, centers_new_chunk)

        # centers new
        ctx[op.outputs[0].key] = centers_new
        # weight_in_clusters
        ctx[op.outputs[1].key] = weight_in_clusters


class KMeansRelocateEmptyClusters(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.KMEANS_RELOCASTE_EMPTY_CLUSTERS

    _empty_clusters = KeyField('empty_clusters')
    _far_x = KeyField('far_x')
    _far_labels = KeyField('far_labels')
    _far_sample_weights = KeyField('far_sample_weight')
    _centers_new = KeyField('centers_new')
    _weight_in_clusters = KeyField('weight_in_clusters')

    def __init__(self, empty_clusters=None, far_x=None, far_labels=None,
                 far_sample_weights=None, centers_new=None,
                 weight_in_clusters=None, output_types=None, **kw):
        super().__init__(_empty_clusters=empty_clusters, _far_x=far_x,
                         _far_labels=far_labels, _far_sample_weights=far_sample_weights,
                         _centers_new=centers_new, _weight_in_clusters=weight_in_clusters,
                         _output_types=output_types, **kw)
        if self._output_types is None:
            self._output_types = [OutputType.tensor] * self.output_limit

    @property
    def empty_clusters(self):
        return self._empty_clusters

    @property
    def far_x(self):
        return self._far_x

    @property
    def far_labels(self):
        return self._far_labels

    @property
    def far_sample_weights(self):
        return self._far_sample_weights

    @property
    def centers_new(self):
        return self._centers_new

    @property
    def weight_in_clusters(self):
        return self._weight_in_clusters

    @property
    def output_limit(self):
        return 2

    @property
    def _input_fields(self):
        return '_empty_clusters', '_far_x', '_far_labels', \
               '_far_sample_weights', '_centers_new', '_weight_in_clusters'

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs)
        for field in self._input_fields:
            ob = getattr(self, field)
            if ob is not None:
                setattr(self, field, next(inputs_iter))

    def __call__(self):
        kws = [
            # centers_new
            self._centers_new.params,
            # weight_in_clusters
            self._weight_in_clusters.params
        ]
        return self.new_tileables(
            [getattr(self, field) for field in self._input_fields], kws=kws)

    @classmethod
    def tile(cls, op: "KMeansRelocateEmptyClusters"):
        empty_clusters = op.empty_clusters.rechunk(
            op.empty_clusters.shape)._inplace_tile()
        far_x = op.far_x.rechunk(op.far_x.shape)._inplace_tile()
        far_labels = op.far_labels.rechunk(op.far_labels.shape)._inplace_tile()
        far_sample_weight = op.far_sample_weights.rechunk(
            op.far_sample_weights.shape)._inplace_tile()
        centers_new = op.centers_new.rechunk(op.centers_new.shape)._inplace_tile()
        weight_in_clusters = op.weight_in_clusters.rechunk(
            op.weight_in_clusters.shape)._inplace_tile()

        chunk_op = op.copy().reset_key()
        out_centers_new_chunk, out_weight_in_clusters_chunk = chunk_op.new_chunks(
            [empty_clusters.chunks[0], far_x.chunks[0], far_labels.chunks[0],
             far_sample_weight.chunks[0], centers_new.chunks[0],
             weight_in_clusters.chunks[0]], kws=[centers_new.chunks[0].params,
                                                 weight_in_clusters.chunks[0].params])

        out_centers_new_params = centers_new.params
        out_centers_new_params['nsplits'] = centers_new.nsplits
        out_centers_new_params['chunks'] = [out_centers_new_chunk]
        out_weight_in_clusters_params = weight_in_clusters.params
        out_weight_in_clusters_params['nsplits'] = weight_in_clusters.nsplits
        out_weight_in_clusters_params['chunks'] = [out_weight_in_clusters_chunk]
        new_op = op.copy()
        return new_op.new_tileables(op.inputs, kws=[out_centers_new_params,
                                                    out_weight_in_clusters_params])

    @classmethod
    def execute(cls, ctx, op):
        (empty_clusters, far_x, far_labels, far_sample_weight,
         center_new, weight_in_clusters), device_id, xp = as_same_device(
            [ctx[inp.key] for inp in op.inputs], op.device, ret_extra=True)

        out_centers_new = center_new.copy()
        out_weight_in_clusters = weight_in_clusters.copy()
        del center_new, weight_in_clusters

        n_empty = empty_clusters.shape[0]
        n_features = far_x.shape[1]

        for idx in range(n_empty):
            new_cluster_id = empty_clusters[idx]
            weight = far_sample_weight[idx]
            old_cluster_id = far_labels[idx]

            for k in range(n_features):
                out_centers_new[old_cluster_id, k] -= far_x[idx, k] * weight
                out_centers_new[new_cluster_id, k] = far_x[idx, k] * weight

            out_weight_in_clusters[new_cluster_id] = weight
            out_weight_in_clusters[old_cluster_id] -= weight

        ctx[op.outputs[0].key] = out_centers_new
        ctx[op.outputs[1].key] = out_weight_in_clusters


def _relocate_empty_clusters(X, sample_weight, centers_old, centers_new,
                             weight_in_clusters, labels, to_run=None,
                             session=None, run_kwargs=None):
    to_run = to_run or list()
    empty_clusters = mt.where(mt.equal(weight_in_clusters, 0))[0].astype(mt.int32)
    to_run.append(empty_clusters)

    mt.ExecutableTuple(to_run).execute(session=session, **(run_kwargs or dict()))

    n_empty = empty_clusters.shape[0]

    if n_empty == 0:
        return centers_new, weight_in_clusters

    distances = ((mt.asarray(X) - mt.asarray(centers_old)[labels]) ** 2).sum(axis=1)
    far_from_centers = \
        mt.argpartition(distances, -n_empty)[:-n_empty-1:-1].astype(np.int32)

    far_x = X[far_from_centers]
    far_labels = labels[far_from_centers]
    far_sample_weight = sample_weight[far_from_centers]

    op = KMeansRelocateEmptyClusters(
        empty_clusters=empty_clusters, far_x=far_x, far_labels=far_labels,
        far_sample_weights=far_sample_weight, centers_new=centers_new,
        weight_in_clusters=weight_in_clusters)
    return op()
