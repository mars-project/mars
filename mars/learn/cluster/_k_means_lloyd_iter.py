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
from sklearn.utils.extmath import row_norms as sklearn_row_norms

from ... import opcodes
from ...operands import OperandStage
from ...serialize import KeyField, BoolField, Int32Field
from ...tensor.array_utils import as_same_device, device, sparse
from ...tensor.core import TensorOrder
from ...tiles import TilesError
from ...utils import check_chunks_unknown_shape
from ..operands import LearnOperand, LearnOperandMixin, OutputType
from ._k_means_common import _execute_merge_update, _relocate_empty_clusters
from ._k_means_fast import update_center
from ._k_means_lloyd import update_chunk_dense, update_chunk_sparse


class KMeansLloydUpdate(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.KMEANS_LLOYD_UPDATE

    _x = KeyField('x')
    _sample_weight = KeyField('sample_weight')
    _x_squared_norms = KeyField('x_squared_norms')
    _centers_old = KeyField('centers_old')
    _labels = KeyField('labels')
    _update_centers = BoolField('update_centers')
    _n_clusters = Int32Field('n_clusters')

    def __init__(self, x=None, sample_weight=None, x_squared_norms=None,
                 centers_old=None, labels=None, update_centers=None,
                 n_clusters=None, output_types=None, stage=None, **kw):
        super().__init__(_x=x, _sample_weight=sample_weight,
                         _x_squared_norms=x_squared_norms,
                         _centers_old=centers_old, _labels=labels,
                         _update_centers=update_centers, _n_clusters=n_clusters,
                         _output_types=output_types, _stage=stage, **kw)
        if self._output_types is None:
            self._output_types = [OutputType.tensor] * self.output_limit

    @property
    def x(self):
        return self._x

    @property
    def sample_weight(self):
        return self._sample_weight

    @property
    def x_squared_norms(self):
        return self._x_squared_norms

    @property
    def centers_old(self):
        return self._centers_old

    @property
    def labels(self):
        return self._labels

    @property
    def update_centers(self):
        return self._update_centers

    @property
    def n_clusters(self):
        return self._n_clusters

    @property
    def output_limit(self):
        return 3 if self._stage != OperandStage.reduce else 2

    @property
    def _input_fields(self):
        return '_x', '_sample_weight', '_x_squared_norms', \
               '_centers_old', '_labels'

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs)
        for field in self._input_fields:
            if getattr(self, field, None) is not None:
                setattr(self, field, next(inputs_iter))

    def __call__(self):
        kws = [
            # labels
            self._labels.params,
            # centers_new
            {
                'shape': (self._n_clusters, self._x.shape[1]),
                'dtype': self._centers_old.dtype,
                'order': TensorOrder.C_ORDER
            },
            # weight_in_clusters
            {
                'shape': (self._n_clusters,),
                'dtype': self._centers_old.dtype,
                'order': TensorOrder.C_ORDER
            }
        ]
        return self.new_tileables(
            [getattr(self, field) for field in self._input_fields], kws=kws)

    @classmethod
    def tile(cls, op: "KMeansLloydUpdate"):
        check_chunks_unknown_shape(op.inputs, TilesError)
        x = op.x
        if x.chunk_shape[1] != 1:  # pragma: no cover
            x = x.rechunk({1: x.shape[1]})._inplace_tile()
        sample_weight = op.sample_weight.rechunk({0: x.nsplits[0]})._inplace_tile()
        x_squared_norms = op.x_squared_norms.rechunk({0: x.nsplits[0]})._inplace_tile()
        labels = op.labels.rechunk({0: x.nsplits[0]})._inplace_tile()
        assert len(op.centers_old.chunks) == 1

        labels_chunks, centers_new_chunks, weight_in_clusters_chunks = [], [], []
        for i in range(x.chunk_shape[0]):
            x_chunk = x.cix[i, 0]
            sample_weight_chunk = sample_weight.cix[i, ]
            x_squared_norms_chunk = x_squared_norms.cix[i, ]
            labels_chunk = labels.cix[i, ]
            chunk_op = op.copy().reset_key()
            chunk_op._stage = OperandStage.map
            chunk_kws = [
                labels_chunk.params,
                {'index': (0, 0),
                 'shape': (op.n_clusters, x_chunk.shape[1]),
                 'dtype': op.centers_old.dtype,
                 'order': TensorOrder.C_ORDER},
                {'index': (0,),
                 'shape': (op.n_clusters,),
                 'dtype': op.centers_old.dtype,
                 'order': TensorOrder.C_ORDER}
            ]
            labels_chunk, centers_new_chunk, weight_in_clusters_chunk = chunk_op.new_chunks(
                [x_chunk, sample_weight_chunk, x_squared_norms_chunk,
                 op.centers_old.chunks[0], labels_chunk], kws=chunk_kws)
            labels_chunks.append(labels_chunk)
            centers_new_chunks.append(centers_new_chunk)
            weight_in_clusters_chunks.append(weight_in_clusters_chunk)

        if op.update_centers:
            # merge centers_new and weight_in_clusters
            merge_op = KMeansLloydUpdate(stage=OperandStage.reduce,
                                         n_clusters=op.n_clusters)
            merge_chunk_kw = [
                centers_new_chunks[0].params,
                weight_in_clusters_chunks[0].params
            ]
            centers_new_chunk, weight_in_cluster_chunk = merge_op.new_chunks(
                centers_new_chunks + weight_in_clusters_chunks, kws=merge_chunk_kw)
        else:
            # the data is meaningless, just pick one
            centers_new_chunk = centers_new_chunks[0]
            weight_in_cluster_chunk = weight_in_clusters_chunks[0]

        out_params = [out.params for out in op.outputs]
        # labels
        out_params[0]['nsplits'] = labels.nsplits
        out_params[0]['chunks'] = labels_chunks
        # centers_new
        out_params[1]['nsplits'] = tuple((s,) for s in op.outputs[1].shape)
        out_params[1]['chunks'] = [centers_new_chunk]
        # weight_in_clusters
        out_params[2]['nsplits'] = tuple((s,) for s in op.outputs[2].shape)
        out_params[2]['chunks'] = [weight_in_cluster_chunk]
        new_op = op.copy()
        return new_op.new_tileables(op.inputs, kws=out_params)

    @classmethod
    def _execute_reduce(cls, ctx, op):
        return _execute_merge_update(ctx, op)

    @classmethod
    def execute(cls, ctx, op: "KMeansLloydUpdate"):
        if op.stage == OperandStage.reduce:
            return cls._execute_reduce(ctx, op)
        else:
            (x, sample_weight, x_squared_norms, centers_old, labels), device_id, xp = as_same_device(
                [ctx[inp.key] for inp in op.inputs], device=op.device,
                ret_extra=True, copy_if_not_writeable=True)

            with device(device_id):
                if not op.update_centers:
                    centers_new = centers_old.copy()
                else:
                    centers_new = np.zeros_like(centers_old)
                weight_in_clusters = np.zeros(op.n_clusters, dtype=x.dtype)
                centers_squared_norms = sklearn_row_norms(centers_old, squared=True)

                if xp is np:
                    method = update_chunk_dense
                elif xp is sparse:
                    method = update_chunk_sparse
                else:  # pragma: no cover
                    raise NotImplementedError('Does not support run on GPU')
                out_labels = labels.copy()
                method(x, sample_weight, x_squared_norms, centers_old,
                       centers_squared_norms, out_labels, centers_new,
                       weight_in_clusters, op.update_centers)

                # labels
                ctx[op.outputs[0].key] = out_labels
                # centers_new
                ctx[op.outputs[1].key] = centers_new
                # weight_in_cluster
                ctx[op.outputs[2].key] = weight_in_clusters


class KMeansLloydPostprocess(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.KMEANS_LLOYD_POSTPROCESS

    _centers_old = KeyField('centers_old')
    _centers_new = KeyField('centers_new')
    _center_shift = KeyField('center_shift')
    _weight_in_clusters = KeyField('weight_in_clusters')

    def __init__(self, centers_old=None, centers_new=None,
                 center_shift=None, weight_in_clusters=None,
                 output_types=None, stage=None, **kw):
        super().__init__(_centers_old=centers_old, _centers_new=centers_new,
                         _center_shift=center_shift,
                         _weight_in_clusters=weight_in_clusters,
                         _output_types=output_types, _stage=stage, **kw)
        if self._output_types is None:
            self._output_types = [OutputType.tensor] * self.output_limit

    @property
    def centers_old(self):
        return self._centers_old

    @property
    def centers_new(self):
        return self._centers_new

    @property
    def center_shift(self):
        return self._center_shift

    @property
    def weight_in_clusters(self):
        return self._weight_in_clusters

    @property
    def output_limit(self):
        return 2

    @property
    def _input_fields(self):
        return '_centers_old', '_centers_new', '_center_shift', \
               '_weight_in_clusters'

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
            # center_shift
            self._center_shift.params
        ]
        return self.new_tileables(
            [getattr(self, f) for f in self._input_fields], kws=kws)

    @classmethod
    def tile(cls, op: "KMeansLloydPostprocess"):
        assert len(op.centers_old.chunks) == 1
        assert len(op.centers_new.chunks) == 1
        assert len(op.center_shift.chunks) == 1
        assert len(op.weight_in_clusters.chunks) == 1

        centers_old_chunk = op.centers_old.chunks[0]
        centers_new_chunk = op.centers_new.chunks[0]
        center_shift_chunk = op.center_shift.chunks[0]
        weight_in_clusters_chunk = op.weight_in_clusters.chunks[0]
        centers_new_chunk, center_shift_chunk = KMeansLloydPostprocess(
            centers_old=centers_old_chunk,
            centers_new=centers_new_chunk,
            center_shift=center_shift_chunk,
            weight_in_clusters=weight_in_clusters_chunk
        ).new_chunks([centers_old_chunk, centers_new_chunk,
                      center_shift_chunk, weight_in_clusters_chunk],
                     kws=[centers_new_chunk.params, center_shift_chunk.params])

        centers_new_kw = op.centers_new.params
        centers_new_kw['chunks'] = [centers_new_chunk]
        centers_new_kw['nsplits'] = op.centers_new.nsplits
        center_shift_kw = op.center_shift.params
        center_shift_kw['chunks'] = [center_shift_chunk]
        center_shift_kw['nsplits'] = op.center_shift.nsplits
        new_op = op.copy()
        return new_op.new_tileables(
            op.inputs, kws=[centers_new_kw, center_shift_kw])

    @classmethod
    def execute(cls, ctx, op: "KMeansLloydPostprocess"):
        (centers_old, centers_new, center_shift, weight_in_clusters), device_id, xp = \
            as_same_device([ctx[inp.key] for inp in op.inputs], op.device,
                           ret_extra=True, copy_if_not_writeable=True)

        with device(device_id):
            out_center_shift = center_shift.copy()
            out_centers_new = centers_new.copy()
            update_center(centers_old, out_centers_new,
                          out_center_shift, weight_in_clusters)

            ctx[op.outputs[0].key] = out_centers_new
            ctx[op.outputs[1].key] = out_center_shift


def lloyd_iter(X, sample_weight, x_squared_norms, centers_old, labels,
               center_shift, update_centers=True, session=None, run_kwargs=None):
    update_op = KMeansLloydUpdate(x=X, sample_weight=sample_weight,
                                  x_squared_norms=x_squared_norms,
                                  centers_old=centers_old, labels=labels,
                                  update_centers=update_centers,
                                  n_clusters=centers_old.shape[0])
    to_run = []
    ret = update_op()
    to_run.extend(ret)
    labels, centers_new, weight_in_clusters = ret

    if update_centers:
        centers_new, weight_in_clusters = \
            _relocate_empty_clusters(X, sample_weight, centers_old, centers_new,
                                     weight_in_clusters, labels, to_run=to_run,
                                     session=session, run_kwargs=run_kwargs)
        postprocess = KMeansLloydPostprocess(
            centers_old=centers_old, centers_new=centers_new,
            center_shift=center_shift, weight_in_clusters=weight_in_clusters)
        centers_new, center_shift = postprocess()

    return centers_new, weight_in_clusters, labels, center_shift
