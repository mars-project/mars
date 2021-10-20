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
from ... import tensor as mt
from ...core import OutputType, recursive_tile
from ...serialization.serializables import KeyField, Int32Field
from ...serialization.serializables import Float32Field, BoolField
from ...tensor.array_utils import as_same_device, device, sparse
from ...tensor.core import TensorOrder
from ...tensor.random import RandomStateField
from ...utils import has_unknown_shape
from ..operands import LearnOperand, LearnOperandMixin
from ._kmeans import _labels_inertia


def _mini_batch_update(X, sample_weight, x_squared_norms, weight_sums, centers,
                       old_center_buffer, nearest_center, compute_squared_diff):

    # dense variant in mostly numpy (not as memory efficient though)
    k = centers.shape[0]
    squared_diff = 0.0
    for center_idx in range(k):
        # find points from minibatch that are assigned to this center
        center_mask = nearest_center == center_idx
        wsum = sample_weight[center_mask].sum()

        if wsum > 0:
            # print("\033[35m 更新center\033[0m")
            if compute_squared_diff:
                old_center_buffer[:] = centers[center_idx]

            centers[center_idx] *= weight_sums[center_idx]
            centers[center_idx] += np.sum(X[center_mask] *
                                          sample_weight[center_mask, np.newaxis], axis=0)

            weight_sums[center_idx] += wsum

            # inplace rescale to compute mean of all points (old and new)
            # Note: numpy >= 1.10 does not support '/=' for the following
            # expression for a mixture of int and float (see numpy issue #6464)
            centers[center_idx] = centers[center_idx] / weight_sums[center_idx]

            # update the squared diff if necessary
            if compute_squared_diff:
                diff = centers[center_idx].ravel() - old_center_buffer.ravel()
                squared_diff += np.dot(diff, diff)

    return squared_diff


class MiniBatchUpdate(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.MINI_BATCH_UPDATE

    _x = KeyField('x')
    _sample_weight = KeyField('sample_weight')
    _x_squared_norms = KeyField('x_squared_norms')
    _weight_sums = KeyField('weight_sums')
    _centers_old = KeyField('centers_old')
    _nearest_center = KeyField('nearest_center')
    _compute_squared_diff = BoolField('compute_suqared_diff')
    _n_clusters = Int32Field('n_clusters')

    def __init__(self, x=None, sample_weight=None, x_squared_norms=None,
                 weight_sums=None, centers_old=None, nearest_center=None,
                 compute_squared_diff=None, n_clusters=None, output_types=None, **kw):
        super().__init__(_x=x, _sample_weight=sample_weight,
                         _x_squared_norms=x_squared_norms,
                         _weight_sums=weight_sums, _centers_old=centers_old,
                         _nearest_center=nearest_center,
                         _compute_squared_diff=compute_squared_diff,
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
    def x_squared_norms(self):
        return self._x_squared_norms

    @property
    def centers_old(self):
        return self._centers_old

    @property
    def weight_sums(self):
        return self._weight_sums

    @property
    def nearest_center(self):
        return self._nearest_center

    @property
    def compute_squared_diff(self):
        return self._compute_squared_diff

    @property
    def n_clusters(self):
        return self._n_clusters

    @property
    def output_limit(self):
        return 2

    @property
    def _input_fields(self):
        return '_x', '_sample_weight', '_x_squared_norms', \
               '_centers_old', '_weight_sums', '_nearest_center'

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(inputs)
        for field in self._input_fields:
            if getattr(self, field, None) is not None:
                setattr(self, field, next(inputs_iter))

    def __call__(self):
        kws = [
            # centers_new
            {
                'shape': (self._n_clusters, self._x.shape[1]),
                'dtype': self._centers_old.dtype,
                'order': TensorOrder.C_ORDER
            },
            # squared_diff
            {
                'shape': (),
                'dtype': np.dtype(float),
                'order': TensorOrder.C_ORDER
            }
        ]
        return self.new_tileables(
            [getattr(self, f) for f in self._input_fields], kws=kws
        )

    @classmethod
    def tile(cls, op: "MiniBatchUpdate"):
        if has_unknown_shape(*op.inputs):
            yield

        x = op.x
        # if x.chunk_shape[1] != 1:
        x = yield from recursive_tile(x.rechunk({1: x.shape[1]}))
        sample_weight = yield from recursive_tile(
            op.sample_weight.rechunk({0: x.nsplits[0]}))
        x_squared_norms = yield from recursive_tile(
            op.x_squared_norms.rechunk({0: x.nsplits[0]}))
        nearest_center = yield from recursive_tile(
            op.nearest_center.rechunk({0: x.nsplits[0]}))
        assert len(op.centers_old.chunks) == 1

        centers_new_chunks, squared_diff_chunks = [], []
        for i in range(x.chunk_shape[0]):
            x_chunk = x.cix[i, 0]
            sample_weight_chunk = sample_weight.cix[i, ]
            x_squared_norms_chunk = x_squared_norms.cix[i, ]
            nearest_center_chunk = nearest_center.cix[i, ]

            chunk_op = op.copy().reset_key()
            chunk_kws = [
                {
                    'index': (0, 0),
                    'shape': (op.n_clusters, x_chunk.shape[1]),
                    'dtype': op.centers_old.dtype,
                    'order': TensorOrder.C_ORDER,
                },
                {
                    'index': (0,),
                    'shape': (1,),
                    'dtype': np.dtype(float),
                    'order': TensorOrder.C_ORDER,
                }
            ]
            centers_new_chunk, squared_diff_chunk = chunk_op.new_chunks(
                [x_chunk, sample_weight_chunk, x_squared_norms_chunk,
                    op.weight_sums.chunks[0], op.centers_old.chunks[0],
                    nearest_center_chunk], kws=chunk_kws)
            centers_new_chunks.append(centers_new_chunk)
            # weight_sums_chunks.append(weight_sums_chunk)
            squared_diff_chunks.append(squared_diff_chunk)

        out_params = [out.params for out in op.outputs]
        # centers_new
        out_params[0]['nsplits'] = tuple((s,) for s in op.outputs[0].shape)
        out_params[0]['chunks'] = [centers_new_chunk]
        # squared_diff
        out_params[1]['nsplits'] = ((1,) * x.chunk_shape[0],)
        out_params[1]['chunks'] = [squared_diff_chunk]
        out_params[1]['shape'] = (x.chunk_shape[0],)
        new_op = op.copy()

        return new_op.new_tileables(op.inputs, kws=out_params)

    @classmethod
    def execute(cls, ctx, op: "MiniBatchUpdate"):
        (x, sample_weight, x_squared_norms, weight_sums, centers_old, nearest_center), \
            device_id, xp = as_same_device(
                [ctx[inp.key] for inp in op.inputs], device=op.device,
                ret_extra=True, copy_if_not_writeable=True)

        with device(device_id):
            if xp is np:
                method = _mini_batch_update
            elif xp is sparse:
                raise NotImplementedError('Does not support for sparse')
            else:
                raise NotImplementedError('Does not support run on GPU')

            centers_new = centers_old.copy()
            squared_diff = method(
                x, sample_weight, x_squared_norms, weight_sums, centers_new,
                centers_old, nearest_center, op.compute_squared_diff)

            # centers_new
            ctx[op.outputs[0].key] = centers_new
            # squared_diff
            ctx[op.outputs[1].key] = np.array([squared_diff])


def _reassign_cluster(X, weight_sums, centers, reassignment_ratio, random_state):
    # Reassign clusters that have very low weight
    to_reassign = weight_sums < reassignment_ratio * weight_sums.max()
    # pick at most .5 * batch_size samples as new centers
    if to_reassign.sum() > .5 * X.shape[0]:
        indices_dont_reassign = \
            np.argsort(weight_sums)[int(.5 * X.shape[0]):]
        to_reassign[indices_dont_reassign] = False
    n_reassigns = to_reassign.sum()

    if n_reassigns:
        # Pick new clusters amongst observations with uniform probability
        new_centers = random_state.choice(X.shape[0], replace=False,
                                          size=n_reassigns)
        # TODO(mimku): Add support for sparse mode
        centers[to_reassign] = X[new_centers]

    # reset counts of reassigned centers, but don't reset them too small
    # to avoid instant reassignment. This is a pretty dirty hack as it
    # also modifies the learning rates.
    weight_sums[to_reassign] = np.min(weight_sums[~to_reassign])
    return n_reassigns


class MiniBatchReassignCluster(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.MINI_BATCH_REASSIGN_CLUSTER

    _x = KeyField('x')
    _weight_sums = KeyField('weight_sums')
    _centers = KeyField('centers')
    _reassignment_ratio = Float32Field('reassignment_ratio')
    _state = RandomStateField('state')

    def __init__(self, x=None, weight_sums=None, centers=None,
                 reassignment_ratio=None, state=None,
                 output_types=None, **kw):
        super().__init__(_x=x, _weight_sums=weight_sums, _centers=centers,
                         _reassignment_ratio=reassignment_ratio, _state=state,
                         _output_types=output_types, **kw)
        if self._output_types is None:
            self._output_types = [OutputType.tensor]

    @property
    def x(self):
        return self._x

    @property
    def weight_sums(self):
        return self._weight_sums

    @property
    def centers(self):
        return self._centers

    @property
    def reassignment_ratio(self):
        return self._reassignment_ratio

    @property
    def state(self):
        return self._state

    @property
    def _input_fields(self):
        return '_x', '_weight_sums', '_centers'

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(inputs)
        for field in self._input_fields:
            if getattr(self, field, None) is not None:
                setattr(self, field, next(inputs_iter))

    def __call__(self):
        kws = [
            # n_reassigns
            {
                'shape': (),
                'dtype': np.dtype(int),
                'order': TensorOrder.C_ORDER
            }
        ]
        return self.new_tileable(
            [getattr(self, f) for f in self._input_fields], kws=kws
        )

    @classmethod
    def tile(cls, op: "MiniBatchReassignCluster"):
        if has_unknown_shape(*op.inputs):
            yield
        x = op.x
        if x.chunk_shape[1] != 1:
            x = yield from recursive_tile(x.rechunk({1: x.shape[1]}))
        assert len(op.centers.chunks) == 1

        n_reassign_chunks = []
        for i in range(x.chunk_shape[0]):
            x_chunk = x.cix[i, 0]

            chunk_op = op.copy().reset_key()
            chunk_kws = [
                {
                    'index': (0,),
                    'shape': (1,),
                    'dtype': np.dtype(float),
                    'order': TensorOrder.C_ORDER,
                }
            ]
            n_reassign_chunk = chunk_op.new_chunks(
                [x_chunk, op.weight_sums.chunks[0], op.centers.chunks[0]],
                kws=chunk_kws
            )

        n_reassign_chunks.append(n_reassign_chunk)
        out_params = [out.params for out in op.outputs]
        # n_reassign
        out_params[0]['nsplits'] = ((1,) * x.chunk_shape[0],)
        out_params[0]['chunks'] = n_reassign_chunk
        out_params[0]['shape'] = (x.chunk_shape[0],)
        new_op = op.copy()

        return new_op.new_tileables(op.inputs, kws=out_params)

    @classmethod
    def execute(cls, ctx, op: "MiniBatchReassignCluster"):

        (x, weight_sums, centers), device_id, xp = as_same_device(
            [ctx[inp.key] for inp in op.inputs], device=op.device,
            ret_extra=True, copy_if_not_writeable=True
        )
        with device(device_id):
            if xp is np:
                method = _reassign_cluster
            elif xp is sparse:
                raise NotImplementedError('Does not support run on sparse')
            else:
                raise NotImplementedError('Does not support run on GPU')

            n_reassigns = method(x, weight_sums, centers, op.reassignment_ratio, op.state)

            ctx[op.outputs[0].key] = np.array([n_reassigns], dtype=int)


def _mini_batch_step(X, sample_weight, x_squared_norms, centers, n_clusters,
                     compute_squared_diff, weight_sums, random_reassign=False,
                     random_state=None, reassignment_ratio=.01, verbose=False,
                     session=None, run_kwargs=None):
    """Incremental update of the centers for the Minibatch K-Means algorithm.

    Parameters
    ----------

    X : Tensor, shape (n_samples, n_features)
        The observation to cluster. It must be noted that the data will be
        converted to C ordering.

    sample_weight : array-like of shape (n_samples, ), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

    x_squared_norms : Tensor, shape (n_samples, )
        Squared euclidean norm of each data point.

    centers : Tensor, shape (k, n_features)
        The cluster centers. This array is MODIFIED IN PLACE

    n_clusters : int
        The number of clusters to form.

    compute_squared_diff : boolean
        If set to False, the squared diff computation is skipped.

    weight_sums : Tensor, shape (k, )
        The weight of each cluster center.

    random_reassign : boolean, default=False
        If True, centers with very low counts are randomly reassigned to
        observations.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization and to
        pick new clusters amongst observations with uniform probability. Use
        an int to make the randomness deterministic.

    reassignment_ratio : float, default=.01
        Control the fraction of the maximum number of counts for a
        center to be reassigned. A higher value means that low count
        centers are more likely to be reassigned, which means that the
        model will take longer to converge, but should converge in a
        better clustering.

    verbose : bool, default=False
        Controls the verbosity.

    Returns
    -------
    centers_new : Tensor, shape (k, n_features)
        Cluster center after an iteration.

    inertia : float
        Sum of squared distances of samples to their closest cluster center.

    squared_diff : float
        Squared distances between previous and updated cluster centers.
    """
    nearest_center, inertia = _labels_inertia(X, sample_weight, x_squared_norms,
                                              centers, session=session, run_kwargs=run_kwargs)

    if random_reassign and reassignment_ratio > 0:
        reassign_op = MiniBatchReassignCluster(x=X, weight_sums=weight_sums, centers=centers,
                                               reassignment_ratio=reassignment_ratio,
                                               state=random_state)
        n_reassign = reassign_op()
        mt.ExecutableTuple([n_reassign]).execute(session=session,
                                                 **(run_kwargs or dict()))

        if verbose:
            print(f"Reassigning {n_reassign} cluster centers.")

    update_op = MiniBatchUpdate(x=X, sample_weight=sample_weight,
                                x_squared_norms=x_squared_norms,
                                weight_sums=weight_sums, centers_old=centers,
                                nearest_center=nearest_center,
                                compute_squared_diff=compute_squared_diff,
                                n_clusters=n_clusters)
    ret = update_op()
    centers_new, squared_diff = ret

    # Execute for checking convergence later
    mt.ExecutableTuple([inertia, squared_diff]).execute(session=session,
                                                        **(run_kwargs or dict()))

    return centers_new, inertia, squared_diff
