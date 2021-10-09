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

# from copy import deepcopy ==

import warnings
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
try:
    from sklearn.cluster import _hierarchical_fast as _hierarchical
except:
    raise ImportError
from heapq import heapify, heappop, heappush, heappushpop


from ..operands import LearnOperand, LearnOperandMixin
from ... import opcodes
from ... import tensor as mt
from ..metrics.pairwise import pairwise_distances
from ...tensor.array_utils import as_same_device, device
from ...core import OutputType, recursive_tile
from ...tensor.core import TensorOrder
from ...utils import has_unknown_shape
from ...serialization.serializables import Int32Field, ListField
from ...serialization.serializables import KeyField, StringField


def ward_tree(X, *, connectivity=None, n_clusters=None, return_distance=False,
              session=None, run_kwargs=None):
    X = mt.asarray(X)
    if X.ndim == 1:
        X = mt.reshape(X, (-1, 1))
    n_samples, n_features = X.shape

    if connectivity is None:
        from scipy.cluster import hierarchy
        if n_clusters is not None:
            warnings.warn('Partial build of the tree is implemented '
                        'only for structured clustering (i.e. with '
                        'explicit connectivity). The algorithm '
                        'will build the full tree and only '
                        'retain the lower branches required '
                        'for the specified number of clusters',
                        stacklevel=2)
        out = hierarchy.ward(X)

        children_ = out[:, :2].astype(mt.intp)
        children_ = mt.tensor(children_)

        if return_distance:
            distances = out[:, 2]
            distance = mt.tensor(distances)
            return children_, 1, n_samples, None, distances
        else:
            return children_, 1, n_samples, None

    fix_op = FixConnectivity(x=X, connectivity=connectivity, affinity='euclidean')
    ret = fix_op()
    connectivity, n_connected_components = ret
    mt.ExecutableTuple([connectivity, n_connected_components]).execute(
        session=session, **(run_kwargs or dict()))

    if not sparse.isspmatrix_lil(connectivity):
        if not sparse.isspmatrix(connectivity):
            connectivity = sparse.lil_matrix(connectivity)
        else:
            connectivity = connectivity.tolil()

    if n_clusters is None:
        n_nodes = 2 * n_samples - 1
    else:
        if n_clusters > n_samples:
            raise ValueError(f'Cannot provide more clusters than samples. {n_clusters}'
                             f' n_clusters was asked, and there are {n_samples}')
        n_nodes = 2 * n_samples - n_clusters

    coord_row = []
    coord_col = []
    A = []
    for ind, row in enumerate(connectivity.rows):
        A.append(row)
        # We keep only the upper triangular for the moments
        # Generator expressions are faster than arrays on the following
        row = [i for i in row if i < ind]
        coord_row.extend(len(row) * [ind, ])
        coord_col.extend(row)

    coord_row = mt.array(coord_row, dtype=mt.intp, order='C')
    coord_col = mt.array(coord_col, dtype=mt.intp, order='C')

    # build moments as a list
    moments_1 = mt.zeros(n_nodes, order='C')
    moments_1[:n_samples] = 1
    moments_2 = mt.zeros((n_nodes, n_features), order='C')
    moments_2[:n_samples] = X
    inertia = mt.empty(len(coord_row), dtype=mt.float64, order='C')
    moments_1.execute()
    moments_2.execute()
    mt.ExecutableTuple([coord_row, coord_col, inertia]).execute(session=session,
                                                        **(run_kwargs or dict()))

    inertia = _hierarchical.compute_ward_dist(moments_1.to_numpy(),
                                              moments_2.to_numpy(),
                                              coord_row.to_numpy(),
                                              coord_col.to_numpy(),
                                              inertia.to_numpy())

    inertia = list(zip(inertia, coord_row.to_numpy(), coord_col.to_numpy()))

    heapify(inertia)

    build_op = BuildWardTree(moments_1=moments_1, moments_2=moments_2,
                             inertia=inertia, n_nodes=n_nodes,
                             n_samples=n_samples, a=A)
    parent, children, distance, n_leaves = build_op()
    mt.ExecutableTuple([parent, children, distance, n_leaves]).execute(
        session=session, **(run_kwargs or dict()))

    n_leaves = int(n_leaves.fetch().squeeze())

    if return_distance:
        # 2 is scaling factor to compare w/ unstructured version
        distances = mt.sqrt(2. * distance)
        distances.execute()
        return children, n_connected_components, n_leaves, parent, distances
    else:
        return children, n_connected_components, n_leaves, parent


def _ward_tree_build(n_nodes, n_samples, A, moments_1, moments_2, inertia):

    parent = np.arange(n_nodes, dtype=np.intp)
    used_node = np.ones(n_nodes, dtype=bool)
    children = []
    distances = np.empty(n_nodes - n_samples)
    not_visited = np.empty(n_nodes, dtype=np.int8, order='C')

    # recursive merge loop
    for k in range(n_samples, n_nodes):
        # identify the merge
        while True:
            inert, i, j = heappop(inertia)
            if used_node[i] and used_node[j]:
                break
        parent[i], parent[j] = k, k
        children.append((i, j))
        used_node[i] = used_node[j] = False

        distances[k - n_samples] = inert    # distance

        # update the moments
        moments_1[k] = moments_1[i] + moments_1[j]
        moments_2[k] = moments_2[i] + moments_2[j]

        # update the structure matrix A and the inertia matrix
        coord_col = []
        not_visited.fill(1)
        not_visited[k] = 0
        _hierarchical._get_parents(A[i], coord_col, parent, not_visited)
        _hierarchical._get_parents(A[j], coord_col, parent, not_visited)
        # List comprehension is faster than a for loop
        [A[col].append(k) for col in coord_col]
        A.append(coord_col)
        coord_col = np.array(coord_col, dtype=np.intp, order='C')
        coord_row = np.empty(coord_col.shape, dtype=np.intp, order='C')
        coord_row.fill(k)
        n_additions = len(coord_row)
        ini = np.empty(n_additions, dtype=np.float64, order='C')

        _hierarchical.compute_ward_dist(moments_1, moments_2,
                                        coord_row, coord_col, ini)

        # List comprehension is faster than a for loop
        [heappush(inertia, (ini[idx], k, coord_col[idx]))
            for idx in range(n_additions)]

    # Separate leaves in children (empty lists up to now)
    n_leaves = n_samples
    # sort children to get consistent output with unstructured version
    children = [c[::-1] for c in children]
    children = np.array(children)  # return numpy array for efficient caching

    return parent, children, distances, n_leaves


def _fix_connectivity(X, connectivity, affinity):
    n_samples = X.shape[0]

    if (connectivity.shape[0] != n_samples or
            connectivity.shape[1] != n_samples):
        raise ValueError(f'Wrong shape for connectivity matrix: '
                         f'{connectivity.shape} when X is {X.shape}')

    connectivity = connectivity + connectivity.T
    # Convert connectivity matrix to LIL
    if not sparse.isspmatrix_lil(connectivity):
        if not sparse.isspmatrix(connectivity):
            connectivity = sparse.lil_matrix(connectivity)
        else:
            connectivity = connectivity.tolil()

    # Compute the number of nodes
    n_connected_components, labels = connected_components(connectivity)

    if n_connected_components > 1:
        warnings.warn(f"the number of connected components of the "
                      f"connectivity matrix is {n_connected_components} > 1. "
                      f"Completing it to avoid stopping the tree early.",
                      stacklevel=2)
        for i in range(n_connected_components):
            idx_i = np.where(labels == i)[0]
            Xi = X[idx_i]
            for j in range(i):
                idx_j = np.where(labels == j)[0]
                Xj = X[idx_j]
                D = pairwise_distances(Xi, Xj, metric=affinity)
                ii, jj = np.where(D == np.min(D))
                ii = ii[0]
                jj = jj[0]
                connectivity[idx_i[ii], idx_j[jj]] = True
                connectivity[idx_j[jj], idx_i[ii]] = True

    connectivity = sparse.csr_matrix(connectivity).A
    return connectivity, n_connected_components


class FixConnectivity(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.FIX_CONNECTIVITY

    _x = KeyField('x')
    _connectivity = KeyField('connectivity')
    _affinity = StringField('affinity')

    def __init__(self, x=None, connectivity=None, affinity=None,
                 output_types=None, **kw):
        super().__init__(_x=x, _connectivity=connectivity, _affinity=affinity,
                         _output_types=output_types, **kw)
        if self._output_types is None:
            self._output_types = [OutputType.tensor] * self.output_limit

    @property
    def x(self):
        return self._x

    @property
    def connectivity(self):
        return self._connectivity

    @property
    def affinity(self):
        return self._affinity

    @property
    def output_limit(self):
        return 2

    @property
    def _input_fields(self):
        return '_x', '_connectivity'

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(inputs)
        for field in self._input_fields:
            if getattr(self, field, None) is not None:
                setattr(self, field, next(inputs_iter))

    def __call__(self):
        kws = [
            # connectivity
            {
                'shape': (self._x.shape[0], self._x.shape[0]),
                'dtype': self._connectivity.dtype,
                'order': TensorOrder.C_ORDER
            },
            # n_connected_components
            {
                'shape': (),
                'dtype': np.dtype(int),
                'order': TensorOrder.C_ORDER
            }
        ]
        return self.new_tileables(
            [getattr(self, f) for f in self._input_fields], kws=kws
        )

    @classmethod
    def tile(cls, op: "FixConnectivity"):
        if has_unknown_shape(*op.inputs):
            yield

        x = op.x
        x = yield from recursive_tile(x.rechunk({1: x.shape[1]}))
        connectivity = yield from recursive_tile(
            op.connectivity.rechunk({0: x.nsplits[0]}))

        connectivity_return_chunks, n_connected_components_chunks = [], []
        for i in range(x.chunk_shape[0]):
            x_chunk = x.cix[i, 0]
            connectivity_chunk = connectivity.cix[i, 0]

            chunk_op = op.copy().reset_key()
            chunk_kws = [
                {
                    'index': (0, 0),
                    'shape': (x_chunk.shape[0], x_chunk.shape[0]),
                    'dtype': op.connectivity.dtype,
                    'order': TensorOrder.C_ORDER,
                },
                {
                    'index': (0,),
                    'shape': (1,),
                    'dtype': np.dtype(int),
                    'order': TensorOrder.C_ORDER,
                }
            ]
            connectivity_return_chunk, n_connected_components_chunk = \
                chunk_op.new_chunks([x_chunk, connectivity_chunk], kws=chunk_kws)
            connectivity_return_chunks.append(connectivity_return_chunk)
            n_connected_components_chunks.append(n_connected_components_chunk)

        out_params = [out.params for out in op.outputs]
        # connectiviity
        out_params[0]['nsplits'] = tuple((s,) for s in op.outputs[0].shape)
        out_params[0]['chunks'] = [connectivity_return_chunk]
        # n_connected_components
        out_params[1]['nsplits'] = ((1,) * x.chunk_shape[0],)
        out_params[1]['chunks'] = [n_connected_components_chunk]
        out_params[1]['shape'] = (x.chunk_shape[0],)
        new_op = op.copy()

        return new_op.new_tileables(op.inputs, kws=out_params)

    @classmethod
    def execute(cls, ctx, op: "FixConnectivity"):
        (x, connectivity), device_id, xp = as_same_device(
                [ctx[inp.key] for inp in op.inputs], device=op.device,
                ret_extra=True, copy_if_not_writeable=True)

        with device(device_id):
            method = _fix_connectivity
            connectivity, n_connected_components = method(x, connectivity, op.affinity)

            # connectivity
            ctx[op.outputs[0].key] = connectivity
            # n_connected_components
            ctx[op.outputs[1].key] = np.array([n_connected_components])


class BuildWardTree(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.BUILD_WARD_TREE

    _moments_1 = KeyField('moments_1')
    _moments_2 = KeyField('moments_2')
    _inertia = ListField('inertia')
    _n_nodes = Int32Field('n_nodes')
    _n_samples = Int32Field('n_samples')
    _a = ListField('a')

    def __init__(self, moments_1=None, moments_2=None, inertia=None, n_nodes=None,
                 n_samples=None, a=None, output_types=None, **kw):

        super().__init__(_moments_1=moments_1, _moments_2=moments_2,
                         _inertia=inertia, _n_nodes=n_nodes, _n_samples=n_samples,
                         _a=a, _output_types=output_types, **kw)
        if self._output_types is None:
            self._output_types = [OutputType.tensor] * self.output_limit

    @property
    def moments_1(self):
        return self._moments_1

    @property
    def moments_2(self):
        return self._moments_2

    @property
    def inertia(self):
        return self._inertia

    @property
    def n_nodes(self):
        return self._n_nodes

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def a(self):
        return self._a

    @property
    def output_limit(self):
        return 4

    @property
    def _input_fields(self):
        return '_moments_1', '_moments_2'

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(inputs)
        for field in self._input_fields:
            if getattr(self, field, None) is not None:
                setattr(self, field, next(inputs_iter))

    def __call__(self):

        kws = [
            # parent
            {
                'shape': (self.n_nodes, ),
                'dtype': self._moments_1.dtype,
                'order': TensorOrder.C_ORDER
            },
            # children
            {
                'shape': (self.n_nodes-self.n_samples, 2),
                'dtype': self._moments_1.dtype,
                'order': TensorOrder.C_ORDER
            },
            # distance
            {
                'shape': (self.n_nodes-self.n_samples, ),
                'dtype': self._moments_1.dtype,
                'order': TensorOrder.C_ORDER
            },
            # n_leaves
            {
                'shape': (),
                'dtype': np.dtype(int),
                'order': TensorOrder.C_ORDER
            }
        ]
        out = self.new_tileables(
            [getattr(self, f) for f in self._input_fields], kws=kws
        )
        return out

    @classmethod
    def tile(cls, op: "BuildWardTree"):
        if has_unknown_shape(*op.inputs):
            yield

        moments_2 = yield from recursive_tile(op._moments_2.rechunk({1: op._moments_2.shape[1]}))
        moments_1 = yield from recursive_tile(
            op.moments_1.rechunk({0: op._moments_2.nsplits[0]}))

        parent_chunks, children_chunks, distance_chunks, n_leaves_chunks = [], [], [], []
        for i in range(op.moments_2.chunk_shape[0]):
            moments_1_chunk = moments_1.cix[i, ]
            moments_2_chunk = moments_2.cix[i, 0]

            chunk_op = op.copy().reset_key()
            chunk_kws = [
                {
                    'index': (0,),
                    'shape': (op.n_nodes, ),
                    'dtype': op.moments_1.dtype,
                    'order': TensorOrder.C_ORDER,
                },
                {
                    'index': (0, 0),
                    'shape': (op.n_nodes-op.n_samples, 2),
                    'dtype': op.moments_1.dtype,
                    'order': TensorOrder.C_ORDER,
                },
                {
                    'index': (0,),
                    'shape': (op.n_nodes-op.n_samples, ),
                    'dtype': op.moments_1.dtype,
                    'order': TensorOrder.C_ORDER,
                },
                {
                    'index': (0,),
                    'shape': (1,),
                    'dtype': np.dtype(int),
                    'order': TensorOrder.C_ORDER,
                }
            ]
            parent_chunk, children_chunk, distance_chunk, n_leaves_chunk = \
                chunk_op.new_chunks([moments_1_chunk, moments_2_chunk], kws=chunk_kws)
            parent_chunks.append(parent_chunk)
            children_chunks.append(children_chunk)
            distance_chunks.append(distance_chunk)
            n_leaves_chunks.append(n_leaves_chunk)

        out_params = [out.params for out in op.outputs]
        # parent
        out_params[0]['nsplits'] = tuple((s,) for s in op.outputs[0].shape)
        out_params[0]['chunks'] = [parent_chunk]
        # children
        out_params[1]['nsplits'] = tuple((s,) for s in op.outputs[1].shape)
        out_params[1]['chunks'] = [children_chunk]
        # distance
        out_params[2]['nsplits'] = tuple((s,) for s in op.outputs[2].shape)
        out_params[2]['chunks'] = [distance_chunk]
        # n_leaves
        out_params[3]['nsplits'] = ((1,) * op.moments_2.chunk_shape[0],)
        out_params[3]['chunks'] = [n_leaves_chunk]
        out_params[3]['shape'] = (op.moments_2.chunk_shape[0], )

        new_op = op.copy()
        return new_op.new_tileables(op.inputs, kws=out_params)

    @classmethod
    def execute(cls, ctx, op: "BuildWardTree"):
        (moments_1, moments_2), device_id, xp = as_same_device(
            [ctx[inp.key] for inp in op.inputs], device=op.device,
                ret_extra=True, copy_if_not_writeable=True
        )
        with device(device_id):
            method = _ward_tree_build
            parent, children, distance, n_leaves = method(op.n_nodes,
                    op.n_samples, op.a, moments_1, moments_2, op.inertia)

            # parent
            ctx[op.outputs[0].key] = parent
            # children
            ctx[op.outputs[1].key] = children
            # distance
            ctx[op.outputs[2].key] = distance
            # n_leaves
            ctx[op.outputs[3].key] = np.array([n_leaves])


def _cut_tree(n_clusters, children, n_leaves):
    if n_clusters > n_leaves:
        raise ValueError('Cannot extract more clusters than samples: '
                         '%s clusters where given for a tree with %s leaves.'
                         % (n_clusters, n_leaves))
    children = children[0]
    nodes = [-(max(children[-1]) + 1)]
    for _ in range(n_clusters - 1):
        # As we have a heap, nodes[0] is the smallest element
        these_children = children[-nodes[0] - n_leaves]
        # Insert the 2 children and remove the largest node
        heappush(nodes, -these_children[0])
        heappushpop(nodes, -these_children[1])
    label = np.zeros(n_leaves, dtype=np.intp)
    for i, node in enumerate(nodes):
        label[_hierarchical._hc_get_descendent(-node, children, n_leaves)] = i
    return label


class CutTree(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.CUT_TREE

    _children = KeyField('children')
    _n_clusters = Int32Field('n_clusters')
    _n_leaves = Int32Field('n_leaves')
    _n_samples = Int32Field('n_samples')

    def __init__(self, children=None, n_clusters=None, n_leaves=None,
                 n_samples=None, output_types=None, **kw):
        super().__init__(_children=children, _n_clusters=n_clusters,
                         _n_leaves=n_leaves, _n_samples=n_samples,
                         _output_types=output_types, **kw)
        if self._output_types is None:
            self._output_types = [OutputType.tensor]

    @property
    def children(self):
        return self._children

    @property
    def n_clusters(self):
        return self._n_clusters

    @property
    def n_leaves(self):
        return self._n_leaves

    @property
    def n_samples(self):
        return self._n_samples

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(inputs)
        for field in ('_children', ):
            if getattr(self, field, None) is not None:
                setattr(self, field, next(inputs_iter))

    def __call__(self):
        params = {
            'shape': (self.n_samples, ),
            'dtype': np.dtype(int),
            'order': TensorOrder.C_ORDER
        }
        return self.new_tileable([self._children], kws=[params])

    @classmethod
    def tile(cls, op: "CutTree"):
        if has_unknown_shape(*op.inputs):
            yield

        children = yield from recursive_tile(op.children.rechunk({1: op.children.shape[1]}))

        label_chunks =  []
        for i in range(op.children.chunk_shape[0]):
            children_chunk = children.cix[i, 0]
            chunk_op = op.copy().reset_key()
            chunk_params = {
                'index': (0,),
                'shape': (op.n_samples,),
                'dtype': np.dtype(int),
                'order': TensorOrder.C_ORDER,
            }
            label_chunk = chunk_op.new_chunk(
                [children_chunk], kws=[chunk_params]
            )
            label_chunks.append(label_chunk)

        new_op = op.copy()
        params = op.outputs[0].params
        params['shape'] = (op.n_samples, )
        params['chunks'] = label_chunks
        params['nsplits'] = tuple((s,) for s in op.outputs[0].shape)

        out = new_op.new_tileable(op.inputs, kws=[params])
        ret = yield from recursive_tile(out)
        return ret

    @classmethod
    def execute(cls, ctx, op: "CutTree"):

        (children), device_id, xp = as_same_device(
                [ctx[inp.key] for inp in op.inputs], device=op.device,
                ret_extra=True, copy_if_not_writeable=True)

        with device(device_id):
            method = _cut_tree
            label = method(op.n_clusters, children, op.n_leaves)
            # label
            ctx[op.outputs[0].key] = label
