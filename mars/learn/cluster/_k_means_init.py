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
from ...operands import OperandStage
from ...serialize import KeyField, Int32Field, TupleField
from ...tensor.array_utils import as_same_device, device
from ...tensor.core import TensorOrder
from ...tensor.random.core import _on_serialize_random_state, \
    _on_deserialize_random_state
from ...tiles import TilesError
from ...utils import recursive_tile, check_chunks_unknown_shape
from ..metrics import euclidean_distances
from ..operands import LearnOperand, LearnOperandMixin, OutputType


class KMeansPlusPlusInit(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.KMEANS_PLUS_PLUS_INIT

    _x = KeyField('x')
    _n_clusters = Int32Field('n_clusters')
    _x_squared_norms = KeyField('x_squared_norms')
    _state = TupleField('state', on_serialize=_on_serialize_random_state,
                        on_deserialize=_on_deserialize_random_state)
    _n_local_trials = Int32Field('n_local_trials')

    def __init__(self, x=None, n_clusters=None, x_squared_norms=None,
                 state=None, n_local_trials=None, output_types=None, **kw):
        super().__init__(_x=x, _n_clusters=n_clusters, _x_squared_norms=x_squared_norms,
                         _state=state, _n_local_trials=n_local_trials,
                         _output_types=output_types, **kw)
        if self._output_types is None:
            self._output_types = [OutputType.tensor]

    @property
    def x(self):
        return self._x

    @property
    def n_clusters(self):
        return self._n_clusters

    @property
    def x_squared_norms(self):
        return self._x_squared_norms

    @property
    def state(self):
        return self._state

    @property
    def n_local_trials(self):
        return self._n_local_trials

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._x = self._inputs[0]
        self._x_squared_norms = self._inputs[-1]

    def __call__(self):
        inputs = [self._x, self._x_squared_norms]
        kw = {
            'shape': (self._n_clusters, self._x.shape[1]),
            'dtype': self._x.dtype,
            'order': TensorOrder.C_ORDER
        }
        return self.new_tileable(inputs, kws=[kw])

    @classmethod
    def _tile_one_chunk(cls, op: "KMeansPlusPlusInit"):
        out = op.outputs[0]

        chunk_op = op.copy().reset_key()
        chunk_kw = out.params.copy()
        chunk_kw['index'] = (0, 0)
        chunk_inputs = [op.x.chunks[0], op.x_squared_norms.chunks[0]]
        chunk = chunk_op.new_chunk(chunk_inputs, kws=[chunk_kw])

        kw = out.params
        kw['chunks'] = [chunk]
        kw['nsplits'] = tuple((s,) for s in out.shape)
        new_op = op.copy()
        return new_op.new_tileables(op.inputs, kws=[kw])

    @classmethod
    def tile(cls, op: "KMeansPlusPlusInit"):
        if len(op.x.chunks) == 1:
            assert len(op.x_squared_norms.chunks) == 1
            return cls._tile_one_chunk(op)
        else:
            return cls._tile_k_init(op)

    @classmethod
    def _tile_k_init(cls, op: "KMeansPlusPlusInit"):
        X = op.x
        n_clusters = op.n_clusters
        x_squared_norms = op.x_squared_norms
        random_state = op.state
        n_local_trials = op.n_local_trials

        n_samples, n_features = X.shape

        centers = mt.empty((n_clusters, n_features), dtype=X.dtype)

        assert x_squared_norms is not None, 'x_squared_norms None in _k_init'

        # Set the number of local seeding trials if none is given
        if n_local_trials is None:
            # This is what Arthur/Vassilvitskii tried, but did not report
            # specific results for other than mentioning in the conclusion
            # that it helped.
            n_local_trials = 2 + int(np.log(n_clusters))

        # Pick first center randomly
        center_id = random_state.randint(n_samples)
        if X.issparse():
            centers[0] = X[center_id].todense()
        else:
            centers[0] = X[center_id]

        # Initialize list of closest distances and calculate current potential
        closest_dist_sq = euclidean_distances(
            centers[0, mt.newaxis], X, Y_norm_squared=x_squared_norms,
            squared=True)
        current_pot = closest_dist_sq.sum()

        # Pick the remaining n_clusters-1 points
        for c in range(1, n_clusters):
            # Choose center candidates by sampling with probability proportional
            # to the squared distance to the closest existing center
            rand_vals = random_state.random_sample(n_local_trials) * current_pot
            candidate_ids = mt.searchsorted(closest_dist_sq.cumsum(),
                                            rand_vals)
            # XXX: numerical imprecision can result in a candidate_id out of range
            candidate_ids = mt.clip(candidate_ids, None, closest_dist_sq.size - 1)

            # Compute distances to center candidates
            distance_to_candidates = euclidean_distances(
                X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

            # update closest distances squared and potential for each candidate
            distance_to_candidates = mt.minimum(closest_dist_sq, distance_to_candidates)

            candidates_pot = distance_to_candidates.sum(axis=1)

            # Decide which candidate is the best
            best_candidate = mt.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate]
            closest_dist_sq = distance_to_candidates[best_candidate]
            best_candidate = candidate_ids[best_candidate]

            # Permanently add best center candidate found in local tries
            if X.issparse():
                centers[c] = X[best_candidate].todense()
            else:
                centers[c] = X[best_candidate]

        return [recursive_tile(centers)]

    @classmethod
    def execute(cls, ctx, op: "KMeansPlusPlusInit"):
        try:
            from sklearn.cluster._kmeans import _k_init
        except ImportError:  # pragma: no cover
            from sklearn.cluster.k_means_ import _k_init

        (x, x_squared_norms), device_id, _ = as_same_device(
            [ctx[inp.key] for inp in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            ctx[op.outputs[0].key] = _k_init(x, op.n_clusters, x_squared_norms,
                                             op.state, op.n_local_trials)


###############################################################################
# Initialization heuristic


def _k_init(X, n_clusters, x_squared_norms, random_state, n_local_trials=None):
    """Init n_clusters seeds according to k-means++

    Parameters
    ----------
    X : array or sparse matrix, shape (n_samples, n_features)
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=np.float64).

    n_clusters : integer
        The number of seeds to choose

    x_squared_norms : array, shape (n_samples,)
        Squared Euclidean norm of each data point.

    random_state : int, RandomState instance
        The generator used to initialize the centers. Use an int to make the
        randomness deterministic.
        See :term:`Glossary <random_state>`.

    n_local_trials : integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007

    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.
    """
    op = KMeansPlusPlusInit(x=X, n_clusters=n_clusters, x_squared_norms=x_squared_norms,
                            state=random_state, n_local_trials=n_local_trials)
    return op()


class KMeansScalablePlusPlusInit(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.KMEANS_SCALABLE_PLUS_PLUS_INIT

    _x = KeyField('x')
    _n_clusters = Int32Field('n_clusters')
    _x_squared_norms = KeyField('x_squared_norms')
    _state = TupleField('state', on_serialize=_on_serialize_random_state,
                        on_deserialize=_on_deserialize_random_state)
    _init_iter = Int32Field('init_iter')
    _oversampling_factor = Int32Field('oversampling_factor')

    def __init__(self, x=None, n_clusters=None, x_squared_norms=None,
                 state=None, init_iter=None, oversampling_factor=None,
                 output_types=None, stage=None, **kw):
        super().__init__(_x=x, _n_clusters=n_clusters, _x_squared_norms=x_squared_norms,
                         _state=state, _init_iter=init_iter,
                         _oversampling_factor=oversampling_factor,
                         _stage=stage, _output_types=output_types, **kw)
        if self._output_types is None:
            self._output_types = [OutputType.tensor]

    @property
    def x(self):
        return self._x

    @property
    def n_clusters(self):
        return self._n_clusters

    @property
    def x_squared_norms(self):
        return self._x_squared_norms

    @property
    def state(self):
        return self._state

    @property
    def init_iter(self):
        return self._init_iter

    @property
    def oversampling_factor(self):
        return self._oversampling_factor

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if self._x is not None:
            self._x = self._inputs[0]
        if self._x_squared_norms is not None:
            self._x_squared_norms = self._inputs[-1]

    def __call__(self):
        inputs = [self._x, self._x_squared_norms]
        kw = {
            'shape': (self._n_clusters, self._x.shape[1]),
            'dtype': self._x.dtype,
            'order': TensorOrder.C_ORDER
        }
        return self.new_tileable(inputs, kws=[kw])

    @classmethod
    def tile(cls, op: "KMeansScalablePlusPlusInit"):
        check_chunks_unknown_shape(op.inputs, TilesError)

        x = mt.tensor(op.x)
        x_squared_norms = mt.atleast_2d(op.x_squared_norms)
        out = op.outputs[0]

        random_state = op.state
        rs = mt.random.RandomState.from_numpy(random_state)

        n_samples, n_features = x.shape
        n_clusters = op.n_clusters

        # step 1, sample a centroid
        centers = x[random_state.randint(n_samples, size=1)]

        for _ in range(op.init_iter):
            distances = euclidean_distances(
                x, centers, X_norm_squared=x_squared_norms, squared=True)

            # calculate the cost of data with respect to current centers
            cost = mt.sum(mt.min(distances, axis=1))

            # calculate the distribution to sample new centers
            distribution = mt.min(distances, axis=1) / cost

            # pick new centers
            new_centers_size = op.oversampling_factor * n_clusters
            new_centers = x[rs.choice(n_samples, new_centers_size, p=distribution)]

            centers = mt.concatenate([centers, new_centers])

        # rechunk centers into one chunk
        centers = recursive_tile(centers).rechunk(centers.shape)

        distances = recursive_tile(euclidean_distances(
            x, centers, X_norm_squared=x_squared_norms, squared=True))
        map_chunks = []
        # calculate weight for each chunk
        for c in distances.chunks:
            map_chunk_op = KMeansScalablePlusPlusInit(stage=OperandStage.map)
            map_chunk_kw = {
                'shape': (len(centers),),
                'dtype': np.dtype(np.int64),
                'order': TensorOrder.C_ORDER,
                'index': (c.index[0],)
            }
            map_chunk = map_chunk_op.new_chunk([c], kws=[map_chunk_kw])
            map_chunks.append(map_chunk)

        reduce_chunk_op = KMeansScalablePlusPlusInit(n_clusters=op.n_clusters,
                                                     state=random_state,
                                                     stage=OperandStage.reduce)
        reduce_chunk_kw = out.params
        reduce_chunk_kw['index'] = (0, 0)
        reduce_chunk = reduce_chunk_op.new_chunk([centers.chunks[0]] + map_chunks,
                                                 kws=[reduce_chunk_kw])

        new_op = op.copy()
        kw = out.params
        kw['chunks'] = [reduce_chunk]
        kw['nsplits'] = tuple((s,) for s in out.shape)
        return new_op.new_tileables(op.inputs, kws=[kw])

    @classmethod
    def _execute_map(cls, ctx, op: "KMeansScalablePlusPlusInit"):
        distances = ctx[op.inputs[0].key]
        min_distance_ids = np.argmin(distances, axis=1)
        result = np.zeros(distances.shape[1], dtype=np.int64)
        count = np.bincount(min_distance_ids)
        result[:len(count)] = count
        ctx[op.outputs[0].key] = result

    @classmethod
    def _execute_reduce(cls, ctx, op: "KMeansScalablePlusPlusInit"):
        from sklearn.cluster import KMeans

        inputs = [ctx[inp.key] for inp in op.inputs]

        count = np.zeros(inputs[1].shape[0], dtype=np.int64)
        for inp in inputs[1:]:
            count += inp
        weight = count / count.sum()

        centers = inputs[0]

        kmeans = KMeans(n_clusters=op.n_clusters, n_init=1,
                        random_state=op.state)
        kmeans.fit(centers, sample_weight=weight)
        ctx[op.outputs[0].key] = kmeans.cluster_centers_

    @classmethod
    def execute(cls, ctx, op: "KMeansScalablePlusPlusInit"):
        if op.stage == OperandStage.map:
            return cls._execute_map(ctx, op)
        else:
            return cls._execute_reduce(ctx, op)


def _scalable_k_init(X, n_clusters, x_squared_norms, random_state,
                     oversampling_factor=2, init_iter=5):
    op = KMeansScalablePlusPlusInit(x=X, n_clusters=n_clusters,
                                    x_squared_norms=x_squared_norms,
                                    state=random_state, init_iter=init_iter,
                                    oversampling_factor=oversampling_factor)
    return op()
