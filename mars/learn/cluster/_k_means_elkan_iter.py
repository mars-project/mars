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

from ... import opcodes
from ...core import OutputType, recursive_tile
from ...core.operand import OperandStage
from ...serialization.serializables import KeyField, Int32Field, BoolField
from ...tensor.array_utils import as_same_device, device, cp, sparse
from ...tensor.core import TensorOrder
from ...utils import has_unknown_shape
from ..operands import LearnOperand, LearnOperandMixin
from ._k_means_common import _execute_merge_update, _relocate_empty_clusters
from ._k_means_elkan import init_bounds_dense, init_bounds_sparse, \
    update_chunk_dense, update_chunk_sparse
from ._k_means_fast import update_center, update_upper_lower_bounds


class KMeansElkanInitBounds(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.KMEANS_ELKAN_INIT_BOUNDS

    _x = KeyField('x')
    _centers = KeyField('centers')
    _center_half_distances = KeyField('center_half_distances')
    _n_clusters = Int32Field('n_clusters')

    def __init__(self, x=None, centers=None, center_half_distances=None,
                 n_clusters=None, sparse=None, gpu=None,
                 output_types=None, **kw):
        super().__init__(_x=x, _centers=centers,
                         _center_half_distances=center_half_distances,
                         _n_clusters=n_clusters, _sparse=sparse, _gpu=gpu,
                         _output_types=output_types, **kw)
        if self._output_types is None:
            self._output_types = [OutputType.tensor] * self.output_limit

    @property
    def x(self):
        return self._x

    @property
    def centers(self):
        return self._centers

    @property
    def center_half_distances(self):
        return self._center_half_distances

    @property
    def n_clusters(self):
        return self._n_clusters

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._x = self._inputs[0]
        self._centers = self._inputs[1]
        self._center_half_distances = self._inputs[2]

    @property
    def output_limit(self):
        return 3

    def __call__(self):
        params = []
        # labels
        params.append({
            'shape': (self._x.shape[0],),
            'dtype': np.dtype(np.int32),
            'order': TensorOrder.C_ORDER
        })
        # upper bounds
        params.append({
            'shape': (self._x.shape[0],),
            'dtype': self._x.dtype,
            'order': TensorOrder.C_ORDER
        })
        # lower bounds
        params.append({
            'shape': (self._x.shape[0], self._n_clusters),
            'dtype': self._x.dtype,
            'order': TensorOrder.C_ORDER
        })
        return self.new_tileables(
            [self._x, self._centers, self._center_half_distances], kws=params)

    @classmethod
    def tile(cls, op: "KMeansElkanInitBounds"):
        # unify chunks on axis 0
        if has_unknown_shape(op.centers, op.center_half_distances):
            yield
        x = op.x
        centers = yield from recursive_tile(
            op.centers.rechunk(op.centers.shape))
        center_half_distances = yield from recursive_tile(
            op.center_half_distances.rechunk(op.center_half_distances.shape))

        out_chunks = [list() for _ in range(op.output_limit)]
        for c in x.chunks:
            chunk_op = op.copy().reset_key()
            chunk_params = []
            # labels chunk
            chunk_params.append({
                'shape': (c.shape[0],),
                'index': (c.index[0],),
                'dtype': np.dtype(np.int32),
                'order': TensorOrder.C_ORDER
            })
            # upper bounds
            chunk_params.append({
                'shape': (c.shape[0],),
                'index': (c.index[0],),
                'dtype': c.dtype,
                'order': TensorOrder.C_ORDER
            })
            # lower bounds
            chunk_params.append({
                'shape': (c.shape[0], op.n_clusters),
                'index': (c.index[0], 0),
                'dtype': c.dtype,
                'order': TensorOrder.C_ORDER
            })
            chunks = chunk_op.new_chunks(
                [c, centers.chunks[0], center_half_distances.chunks[0]],
                kws=chunk_params)
            for i, out_chunk in enumerate(chunks):
                out_chunks[i].append(out_chunk)

        out_nsplits = [
            (x.nsplits[0],),
            (x.nsplits[0],),
            (x.nsplits[0], (op.n_clusters,))]
        out_params = [out.params for out in op.outputs]
        for i, chunks in enumerate(out_chunks):
            out_params[i]['chunks'] = chunks
        for i, nsplits in enumerate(out_nsplits):
            out_params[i]['nsplits'] = nsplits
        new_op = op.copy()
        return new_op.new_tileables(op.inputs, kws=out_params)

    @classmethod
    def execute(cls, ctx, op):
        (x, centers, center_half_distances), device_id, xp = \
            as_same_device([ctx[inp.key] for inp in op.inputs], device=op.device,
                           ret_extra=True, copy_if_not_writeable=True)

        with device(device_id):
            if xp is cp:  # pragma: no cover
                raise NotImplementedError('cannot support init_bounds '
                                          'for kmeans elkan')

            n_samples = x.shape[0]
            n_clusters = op.n_clusters

            labels = np.full(n_samples, -1, dtype=np.int32)
            upper_bounds = np.zeros(n_samples, dtype=x.dtype)
            lower_bounds = np.zeros((n_samples, n_clusters), dtype=x.dtype)

            if xp is np:
                init_bounds = init_bounds_dense
            else:
                assert xp is sparse
                init_bounds = init_bounds_sparse

            init_bounds(x, centers, center_half_distances,
                        labels, upper_bounds, lower_bounds)

            ctx[op.outputs[0].key] = labels
            ctx[op.outputs[1].key] = upper_bounds
            ctx[op.outputs[2].key] = lower_bounds


def init_bounds(X, centers, center_half_distances, n_clusters):
    op = KMeansElkanInitBounds(x=X, centers=centers,
                               center_half_distances=center_half_distances,
                               n_clusters=n_clusters,
                               sparse=False, gpu=X.op.gpu)
    return op()


class KMeansElkanUpdate(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.KMEANS_ELKAN_UPDATE

    _x = KeyField('x')
    _sample_weight = KeyField('sample_weight')
    _centers_old = KeyField('centers_old')
    _center_half_distances = KeyField('center_half_distances')
    _distance_next_center = KeyField('distance_next_center')
    _labels = KeyField('labels')
    _upper_bounds = KeyField('upper_bounds')
    _lower_bounds = KeyField('lower_bounds')
    _update_centers = BoolField('update_centers')
    _n_clusters = Int32Field('n_clusters')

    def __init__(self, x=None, sample_weight=None, centers_old=None,
                 center_half_distances=None, distance_next_center=None,
                 labels=None, upper_bounds=None, lower_bounds=None, update_centers=None,
                 n_clusters=None, output_types=None, **kw):
        super().__init__(_x=x, _sample_weight=sample_weight, _centers_old=centers_old,
                         _center_half_distances=center_half_distances,
                         _distance_next_center=distance_next_center,
                         _labels=labels, _upper_bounds=upper_bounds,
                         _lower_bounds=lower_bounds, _update_centers=update_centers,
                         _n_clusters=n_clusters, _output_types=output_types, **kw)
        if self._output_types is None:
            self._output_types = [OutputType.tensor] * self.output_limit

    @property
    def x(self):
        return self._x

    @property
    def sample_weight(self):
        return self._sample_weight

    @property
    def centers_old(self):
        return self._centers_old

    @property
    def center_half_distances(self):
        return self._center_half_distances

    @property
    def distance_next_center(self):
        return self._distance_next_center

    @property
    def labels(self):
        return self._labels

    @property
    def upper_bounds(self):
        return self._upper_bounds

    @property
    def lower_bounds(self):
        return self._lower_bounds

    @property
    def update_centers(self):
        return self._update_centers

    @property
    def n_clusters(self):
        return self._n_clusters

    @property
    def output_limit(self):
        return 5 if self.stage != OperandStage.reduce else 2

    @property
    def _input_fields(self):
        return '_x', '_sample_weight', '_centers_old', \
               '_center_half_distances', '_distance_next_center', \
               '_labels', '_upper_bounds', '_lower_bounds'

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if self.stage != OperandStage.reduce:
            input_fields = self._input_fields
            assert len(input_fields) == len(self._inputs)
            inputs_iter = iter(inputs)
            for field in input_fields:
                setattr(self, field, next(inputs_iter))

    def __call__(self):
        kws = list((
            # labels
            self._labels.params,
            # upper_bounds
            self._upper_bounds.params,
            # lower_bounds
            self._lower_bounds.params,
        ))
        # centers_new
        kws.append({
            'shape': (self._n_clusters, self._x.shape[1]),
            'dtype': self._centers_old.dtype,
            'order': TensorOrder.C_ORDER
        })
        # weight_in_clusters
        kws.append({
            'shape': (self._n_clusters,),
            'dtype': self._centers_old.dtype,
            'order': TensorOrder.C_ORDER
        })
        return self.new_tileables(
            [getattr(self, f) for f in self._input_fields], kws=kws)

    @classmethod
    def tile(cls, op: "KMeansElkanUpdate"):
        if has_unknown_shape(*op.inputs):
            yield
        x = op.x
        if x.chunk_shape[1] != 1:  # pragma: no cover
            x = yield from recursive_tile(x.rechunk({1: x.shape[1]}))
        sample_weight = yield from recursive_tile(op.sample_weight.rechunk({0: x.nsplits[0]}))
        labels = yield from recursive_tile(op.labels.rechunk({0: x.nsplits[0]}))
        upper_bounds = yield from recursive_tile(op.upper_bounds.rechunk({0: x.nsplits[0]}))
        lower_bounds = yield from recursive_tile(
            op.lower_bounds.rechunk({0: x.nsplits[0],
                                     1: op.lower_bounds.shape[1]}))
        centers_old = yield from recursive_tile(op.centers_old.rechunk(op.centers_old.shape))
        center_half_distances = yield from recursive_tile(op.center_half_distances.rechunk(
            op.center_half_distances.shape))
        distance_next_center = yield from recursive_tile(op.distance_next_center.rechunk(
            op.distance_next_center.shape))

        out_chunks = [list() for _ in range(op.output_limit)]
        for i in range(x.chunk_shape[0]):
            x_chunk = x.cix[i, 0]
            sample_weight_chunk = sample_weight.cix[i, ]
            labels_chunk = labels.cix[i, ]
            upper_bounds_chunk = upper_bounds.cix[i, ]
            lower_bounds_chunk = lower_bounds.cix[i, 0]
            chunk_op = op.copy().reset_key()
            chunk_op.stage = OperandStage.map
            chunk_kws = list((
                # labels
                labels_chunk.params,
                # upper_bounds
                upper_bounds_chunk.params,
                # lower_boudns
                lower_bounds_chunk.params
            ))
            # centers_new
            chunk_kws.append({
                'index': (0, 0),
                'shape': (op.n_clusters, x_chunk.shape[1]),
                'dtype': centers_old.dtype,
                'order': TensorOrder.C_ORDER
            })
            # weight_in_clusters
            chunk_kws.append({
                'index': (0,),
                'shape': (op.n_clusters,),
                'dtype': centers_old.dtype,
                'order': TensorOrder.C_ORDER
            })
            chunks = chunk_op.new_chunks(
                [x_chunk, sample_weight_chunk, centers_old.chunks[0],
                 center_half_distances.chunks[0], distance_next_center.chunks[0],
                 labels_chunk, upper_bounds_chunk, lower_bounds_chunk], kws=chunk_kws)
            assert len(chunks) == len(out_chunks)
            for oc, c in zip(out_chunks, chunks):
                oc.append(c)

        label_chunks, upper_bounds_chunks, lower_bounds_chunks = out_chunks[:3]
        centers_new_chunks, weight_in_cluster_chunks = out_chunks[3:]

        if op.update_centers:
            # merge centers_new and weight_in_clusters
            merge_op = KMeansElkanUpdate(stage=OperandStage.reduce,
                                         n_clusters=op.n_clusters)
            merge_chunk_kw = [
                centers_new_chunks[0].params,
                weight_in_cluster_chunks[0].params
            ]
            centers_new_chunk, weight_in_cluster_chunk = merge_op.new_chunks(
                centers_new_chunks + weight_in_cluster_chunks, kws=merge_chunk_kw)
        else:
            # the data is meaningless, just pick one
            centers_new_chunk = centers_new_chunks[0]
            weight_in_cluster_chunk = weight_in_cluster_chunks[0]

        out_params = [out.params for out in op.outputs]
        # labels
        out_params[0]['nsplits'] = labels.nsplits
        out_params[0]['chunks'] = label_chunks
        # upper_bounds
        out_params[1]['nsplits'] = upper_bounds.nsplits
        out_params[1]['chunks'] = upper_bounds_chunks
        # lower_bounds
        out_params[2]['nsplits'] = lower_bounds.nsplits
        out_params[2]['chunks'] = lower_bounds_chunks
        # centers_new
        out_params[3]['nsplits'] = tuple((s,) for s in op.outputs[3].shape)
        out_params[3]['chunks'] = [centers_new_chunk]
        # weight_in_clusters
        out_params[4]['nsplits'] = tuple((s,) for s in op.outputs[4].shape)
        out_params[4]['chunks'] = [weight_in_cluster_chunk]
        new_op = op.copy()
        return new_op.new_tileables(op.inputs, kws=out_params)

    @classmethod
    def _execute_reduce(cls, ctx, op):
        return _execute_merge_update(ctx, op)

    @classmethod
    def execute(cls, ctx, op: "KMeansElkanUpdate"):
        if op.stage == OperandStage.reduce:
            return cls._execute_reduce(ctx, op)
        else:
            (x, sample_weight, centers_old, center_half_distances, distance_next_center,
             labels, upper_bounds, lower_bounds), device_id, xp = as_same_device(
                [ctx[inp.key] for inp in op.inputs], device=op.device,
                ret_extra=True, copy_if_not_writeable=True)

            with device(device_id):
                if not op.update_centers:
                    centers_new = centers_old.copy()
                else:
                    centers_new = np.zeros_like(centers_old)
                weight_in_clusters = np.zeros(op.n_clusters, dtype=x.dtype)

                if xp is np:
                    method = update_chunk_dense
                elif xp is sparse:
                    method = update_chunk_sparse
                else:  # pragma: no cover
                    raise NotImplementedError('Does not support run on GPU')

                out_labels, out_upper_bounds, out_lower_bounds = \
                    labels.copy(), upper_bounds.copy(), lower_bounds.copy()
                method(x, sample_weight, centers_old, center_half_distances, distance_next_center,
                       out_labels, out_upper_bounds, out_lower_bounds,
                       centers_new, weight_in_clusters, op.update_centers)

                # labels
                ctx[op.outputs[0].key] = out_labels
                # upper_bounds
                ctx[op.outputs[1].key] = out_upper_bounds
                # lower_bounds
                ctx[op.outputs[2].key] = out_lower_bounds
                # centers_new
                ctx[op.outputs[3].key] = centers_new
                # weight_in_cluster
                ctx[op.outputs[4].key] = weight_in_clusters


class KMeansElkanPostprocess(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.KMEANS_ELKAN_POSTPROCESS

    _centers_old = KeyField('centers_old')
    _centers_new = KeyField('centers_new')
    _center_shift = KeyField('center_shift')
    _lower_bounds = KeyField('lower_bounds')
    _upper_bounds = KeyField('upper_bounds')
    _labels = KeyField('labels')
    _weight_in_clusters = KeyField('weight_in_clusters')

    def __init__(self, centers_old=None, centers_new=None,
                 center_shift=None, lower_bounds=None, upper_bounds=None,
                 labels=None, weight_in_clusters=None, output_types=None, **kw):
        super().__init__(_centers_old=centers_old, _centers_new=centers_new,
                         _center_shift=center_shift, _lower_bounds=lower_bounds,
                         _upper_bounds=upper_bounds, _labels=labels,
                         _weight_in_clusters=weight_in_clusters,
                         _output_types=output_types, **kw)
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
    def lower_bounds(self):
        return self._lower_bounds

    @property
    def upper_bounds(self):
        return self._upper_bounds

    @property
    def labels(self):
        return self._labels

    @property
    def weight_in_clusters(self):
        return self._weight_in_clusters

    @property
    def output_limit(self):
        if self.stage is None:
            # for tileable
            return 4
        elif self.stage == OperandStage.combine:
            return 2
        else:
            assert self.stage == OperandStage.reduce
            return 2

    @property
    def _input_fields(self):
        return '_centers_old', '_centers_new', '_center_shift', \
               '_lower_bounds', '_upper_bounds', '_labels', \
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
            self._center_shift.params,
            # upper_bounds
            self._upper_bounds.params,
            # lower_bounds
            self._lower_bounds.params
        ]
        return self.new_tileables(
            [getattr(self, f) for f in self._input_fields], kws=kws)

    @classmethod
    def tile(cls, op: "KMeansElkanPostprocess"):
        assert len(op.centers_old.chunks) == 1
        assert len(op.centers_new.chunks) == 1
        assert len(op.center_shift.chunks) == 1
        assert len(op._weight_in_clusters.chunks) == 1
        assert op.lower_bounds.chunk_shape[1] == 1

        centers_old_chunk = op.centers_old.chunks[0]
        centers_new_chunk = op.centers_new.chunks[0]
        center_shift_chunk = op.center_shift.chunks[0]
        weight_in_clusters_chunk = op.weight_in_clusters.chunks[0]

        # calculate center shift first
        centers_new_chunk, center_shift_chunk = KMeansElkanPostprocess(
            centers_old=centers_old_chunk,
            centers_new=centers_new_chunk,
            center_shift=center_shift_chunk,
            weight_in_clusters=weight_in_clusters_chunk,
            stage=OperandStage.combine
        ).new_chunks([centers_old_chunk, centers_new_chunk,
                      center_shift_chunk, weight_in_clusters_chunk],
                     kws=[centers_new_chunk.params, center_shift_chunk.params])

        upper_bounds_chunks, lower_bounds_chunks = [], []
        for upper_bound_chunk, lower_bound_chunk, labels_chunk in \
                zip(op.upper_bounds.chunks, op.lower_bounds.chunks, op.labels.chunks):
            chunk_kws = [upper_bound_chunk.params, lower_bound_chunk.params]
            upper_bound_chk, lower_bound_chk = KMeansElkanPostprocess(
                center_shift=center_shift_chunk,
                lower_bounds=lower_bound_chunk,
                upper_bounds=upper_bound_chunk,
                labels=labels_chunk,
                stage=OperandStage.reduce
            ).new_chunks([center_shift_chunk, lower_bound_chunk,
                          upper_bound_chunk, labels_chunk],
                         kws=chunk_kws)
            upper_bounds_chunks.append(upper_bound_chk)
            lower_bounds_chunks.append(lower_bound_chk)

        centers_new_kw = op.centers_new.params
        centers_new_kw['chunks'] = [centers_new_chunk]
        centers_new_kw['nsplits'] = op.centers_new.nsplits
        center_shift_kw = op.center_shift.params
        center_shift_kw['chunks'] = [center_shift_chunk]
        center_shift_kw['nsplits'] = op.center_shift.nsplits
        upper_bounds_kw = op.upper_bounds.params
        upper_bounds_kw['chunks'] = upper_bounds_chunks
        upper_bounds_kw['nsplits'] = op.upper_bounds.nsplits
        lower_bounds_kw = op.lower_bounds.params
        lower_bounds_kw['chunks'] = lower_bounds_chunks
        lower_bounds_kw['nsplits'] = op.lower_bounds.nsplits
        new_op = op.copy()
        return new_op.new_tileables(
            op.inputs, kws=[centers_new_kw, center_shift_kw,
                            upper_bounds_kw, lower_bounds_kw])

    @classmethod
    def _execute_combine(cls, ctx, op):
        (centers_old, centers_new, center_shift, weight_in_clusters), device_id, xp = \
            as_same_device([ctx[inp.key] for inp in op.inputs], op.device,
                           ret_extra=True, copy_if_not_writeable=True)

        with device(device_id):
            out_centers_new = centers_new.copy()
            out_center_shift = center_shift.copy()
            update_center(centers_old, out_centers_new,
                          out_center_shift, weight_in_clusters)

            ctx[op.outputs[0].key] = out_centers_new
            ctx[op.outputs[1].key] = out_center_shift

    @classmethod
    def _execute_reduce(cls, ctx, op):
        (center_shift, lower_bounds, upper_bounds, labels), device_id, xp = \
            as_same_device([ctx[inp.key] for inp in op.inputs], op.device,
                           ret_extra=True, copy_if_not_writeable=True)

        with device(device_id):
            out_upper_bounds = upper_bounds.copy()
            out_lower_bounds = lower_bounds.copy()
            update_upper_lower_bounds(out_upper_bounds, out_lower_bounds,
                                      labels, center_shift)
            ctx[op.outputs[0].key] = out_upper_bounds
            ctx[op.outputs[1].key] = out_lower_bounds

    @classmethod
    def execute(cls, ctx, op):
        if op.stage == OperandStage.combine:
            return cls._execute_combine(ctx, op)
        else:
            assert op.stage == OperandStage.reduce
            return cls._execute_reduce(ctx, op)


def elkan_iter(X, sample_weight, centers_old, center_half_distances,
               distance_next_center, upper_bounds, lower_bounds, labels,
               center_shift, update_centers=True, session=None, run_kwargs=None):
    update_op = KMeansElkanUpdate(x=X, sample_weight=sample_weight,
                                  centers_old=centers_old,
                                  center_half_distances=center_half_distances,
                                  distance_next_center=distance_next_center,
                                  labels=labels, upper_bounds=upper_bounds,
                                  lower_bounds=lower_bounds,
                                  update_centers=update_centers,
                                  n_clusters=centers_old.shape[0])
    to_run = []
    ret = update_op()
    to_run.extend(ret)
    labels, upper_bounds, lower_bounds, centers_new, weight_in_clusters = ret

    if update_centers:
        centers_new, weight_in_clusters = \
            _relocate_empty_clusters(X, sample_weight, centers_old, centers_new,
                                     weight_in_clusters, labels, to_run=to_run,
                                     session=session, run_kwargs=run_kwargs)
        postprocess = KMeansElkanPostprocess(
            centers_old=centers_old, centers_new=centers_new,
            center_shift=center_shift, lower_bounds=lower_bounds,
            upper_bounds=upper_bounds, labels=labels,
            weight_in_clusters=weight_in_clusters)
        centers_new, center_shift, upper_bounds, lower_bounds = postprocess()

    return centers_new, weight_in_clusters, upper_bounds, lower_bounds, \
           labels, center_shift
