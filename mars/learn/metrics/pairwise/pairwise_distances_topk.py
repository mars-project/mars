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

import warnings
from functools import partial

import numpy as np
try:
    from sklearn import get_config as sklearn_get_config
except ImportError:  # pragma: no cover
    sklearn_get_config = None

from .... import opcodes
from .... import options
from ....core import recursive_tile
from ....core.operand import OperandStage
from ....serialization.serializables import KeyField, BoolField, DictField, Int64Field, AnyField
from ....tensor.core import TensorOrder
from ....tensor.merge import TensorConcatenate
from ....tensor.array_utils import as_same_device, device, get_array_module
from ....utils import has_unknown_shape, parse_readable_size, ensure_own_data
from ...utils import gen_batches
from ...utils.validation import _num_samples
from .core import PairwiseDistances


def get_chunk_n_rows(row_bytes, max_n_rows=None,
                     working_memory=None):
    """Calculates how many rows can be processed within working_memory

    Parameters
    ----------
    row_bytes : int
        The expected number of bytes of memory that will be consumed
        during the processing of each row.
    max_n_rows : int, optional
        The maximum return value.
    working_memory : int or float, optional
        The number of rows to fit inside this number of MiB will be returned.
        When None (default), the value of
        ``sklearn.get_config()['working_memory']`` is used.

    Returns
    -------
    int or the value of n_samples

    Warns
    -----
    Issues a UserWarning if ``row_bytes`` exceeds ``working_memory`` MiB.
    """

    if working_memory is None:  # pragma: no cover
        working_memory = options.learn.working_memory
        if working_memory is None and sklearn_get_config is not None:
            working_memory = sklearn_get_config()['working_memory']
        elif working_memory is None:
            working_memory = 1024

    if isinstance(working_memory, int):
        working_memory *= 2 ** 20
    else:
        working_memory = parse_readable_size(working_memory)[0]

    chunk_n_rows = int(working_memory // row_bytes)
    if max_n_rows is not None:
        chunk_n_rows = min(chunk_n_rows, max_n_rows)
    if chunk_n_rows < 1:  # pragma: no cover
        warnings.warn('Could not adhere to working_memory config. '
                      'Currently %.0fMiB, %.0fMiB required.' %
                      (working_memory, np.ceil(row_bytes * 2 ** -20)))
        chunk_n_rows = 1
    return chunk_n_rows


def _precompute_metric_params(X, Y, xp, metric=None, **kwds):  # pragma: no cover
    """Precompute data-derived metric parameters if not provided
    """
    if metric == "seuclidean" and 'V' not in kwds:
        if X is Y:
            V = xp.var(X, axis=0, ddof=1)
        else:
            V = xp.var(xp.vstack([X, Y]), axis=0, ddof=1)
        return {'V': V}
    if metric == "mahalanobis" and 'VI' not in kwds:
        if X is Y:
            VI = xp.linalg.inv(xp.cov(X.T)).T
        else:
            VI = xp.linalg.inv(xp.cov(xp.vstack([X, Y]).T)).T
        return {'VI': VI}
    return {}


def _check_chunk_size(reduced, chunk_size):  # pragma: no cover
    """Checks chunk is a sequence of expected size or a tuple of same
    """
    if reduced is None:
        return
    is_tuple = isinstance(reduced, tuple)
    if not is_tuple:
        reduced = (reduced,)
    if any(isinstance(r, tuple) or not hasattr(r, '__iter__')
           for r in reduced):
        raise TypeError('reduce_func returned %r. '
                        'Expected sequence(s) of length %d.' %
                        (reduced if is_tuple else reduced[0], chunk_size))
    if any(_num_samples(r) != chunk_size for r in reduced):
        actual_size = tuple(_num_samples(r) for r in reduced)
        raise ValueError('reduce_func returned object of length %s. '
                         'Expected same length as input: %d.' %
                         (actual_size if is_tuple else actual_size[0],
                          chunk_size))


def _pariwise_distance_chunked(X, Y, reduce_func=None, metric='euclidean',
                               working_memory=None, xp=None, **kwds):
    if xp is np:
        from sklearn.metrics import pairwise_distances
    else:  # pragma: no cover
        from cuml.metrics import pairwise_distances

    n_samples_X = _num_samples(X)
    if metric == 'precomputed':  # pragma: no cover
        slices = (slice(0, n_samples_X),)
    else:
        # We get as many rows as possible within our working_memory budget to
        # store len(Y) distances in each row of output.
        #
        # Note:
        #  - this will get at least 1 row, even if 1 row of distances will
        #    exceed working_memory.
        #  - this does not account for any temporary memory usage while
        #    calculating distances (e.g. difference of vectors in manhattan
        #    distance.
        chunk_n_rows = get_chunk_n_rows(row_bytes=8 * _num_samples(Y),
                                        max_n_rows=n_samples_X,
                                        working_memory=working_memory)
        slices = gen_batches(n_samples_X, chunk_n_rows)

    # precompute data-derived metric params
    params = _precompute_metric_params(X, Y, xp, metric=metric, **kwds)
    kwds.update(**params)

    for sl in slices:
        if sl.start == 0 and sl.stop == n_samples_X:
            X_chunk = X  # enable optimised paths for X is Y
        else:
            X_chunk = X[sl]
        # call pairwise op's execute method to get the result
        D_chunk = pairwise_distances(
            ensure_own_data(X_chunk), ensure_own_data(Y),
            metric=metric, **kwds)
        if ((X is Y or Y is None) and metric == 'euclidean'):
            # zeroing diagonal, taking care of aliases of "euclidean",
            # i.e. "l2"
            D_chunk.flat[sl.start::_num_samples(X) + 1] = 0  # pylint: disable=invalid-slice-index
        if reduce_func is not None:
            chunk_size = D_chunk.shape[0]
            D_chunk = reduce_func(D_chunk, sl.start)
            _check_chunk_size(D_chunk, chunk_size)
        yield D_chunk


class PairwiseDistancesTopk(PairwiseDistances):
    _op_type_ = opcodes.PAIRWISE_DISTANCES_TOPK

    _x = KeyField('x')
    _y = KeyField('y')
    _k = Int64Field('k')
    _metric = AnyField('metric')
    _metric_kwargs = DictField('metric_kwargs')
    _return_index = BoolField('return_index')
    _working_memory = AnyField('working_memory')
    # for chunks
    _y_offset = Int64Field('y_offset')

    def __init__(self, x=None, y=None, k=None, metric=None,
                 metric_kwargs=None, return_index=None, working_memory=None, **kw):
        super().__init__(_x=x, _y=y, _k=k, _metric=metric,
                         _metric_kwargs=metric_kwargs, _return_index=return_index,
                         _working_memory=working_memory, **kw)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def k(self):
        return self._k

    @property
    def metric(self):
        return self._metric

    @property
    def metric_kwargs(self):
        return self._metric_kwargs

    @property
    def return_index(self):
        return self._return_index

    @property
    def working_memory(self):
        return self._working_memory

    @property
    def y_offset(self):
        return self._y_offset

    @property
    def output_limit(self):
        return 1 if not self._return_index or \
                    self.stage == OperandStage.map else 2

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if self.stage != OperandStage.agg:
            self._x = self._inputs[0]
            self._y = self._inputs[1]
        else:
            self._x = self._y = None

    def __call__(self, X, Y):
        from .pairwise import pairwise_distances

        # leverage pairwise_distances for checks
        d = pairwise_distances(X, Y, metric=self._metric,
                               **self._metric_kwargs)

        if self._k > Y.shape[0]:
            self._k = Y.shape[0]

        X, Y = d.op.inputs

        shape_list = [X.shape[0]]
        shape_list.append(min(Y.shape[0], self._k))
        shape = tuple(shape_list)
        kws = [
            {'shape': shape,
             'order': TensorOrder.C_ORDER,
             'dtype': np.dtype(np.float64),
             '_type_': 'distance'},
        ]
        if self._return_index:
            kws.append({'shape': shape,
                        'order': TensorOrder.C_ORDER,
                        'dtype': np.dtype(np.int64),
                        '_type_': 'index'})
            return self.new_tensors([X, Y], kws=kws)
        else:
            return self.new_tensors([X, Y], kws=kws)[0]

    @classmethod
    def _gen_out_chunks(cls, x_chunk, y_chunk, chunk_op):
        k = chunk_op.k
        i, j = x_chunk.index[0], y_chunk.index[0]

        distance_chunk_params = {
            'shape': (x_chunk.shape[0], k),
            'order': TensorOrder.C_ORDER,
            'dtype': np.dtype(np.float64),
            'index': (i, j),
            '_type_': 'distance',
        }
        if chunk_op.return_index:
            index_chunk_params = {
                'shape': (x_chunk.shape[0], k),
                'order': TensorOrder.C_ORDER,
                'dtype': np.dtype(np.int64),
                'index': (i, j),
                '_type_': 'index',
            }
            distance_chunk, index_chunk = chunk_op.new_chunks(
                [x_chunk, y_chunk], kws=[distance_chunk_params,
                                         index_chunk_params])
            return distance_chunk, index_chunk
        else:
            return chunk_op.new_chunks([x_chunk, y_chunk],
                                       kws=[distance_chunk_params])[0]

    @classmethod
    def tile(cls, op: "PairwiseDistancesTopk"):
        X, Y = op.x, op.y
        k = op.k

        if X.chunk_shape[1] > 1:
            X = yield from recursive_tile(X.rechunk({1: X.shape[1]}))

        if has_unknown_shape(Y):
            yield
        if Y.chunk_shape[1] > 1:
            Y = yield from recursive_tile(Y.rechunk({1: Y.shape[1]}))

        out_distance_chunks, out_index_chunks = [], []
        y_acc_chunk_shapes = [0] + np.cumsum(Y.nsplits[0]).tolist()
        for i in range(len(range(X.chunk_shape[0]))):
            x_chunk = X.cix[i, 0]
            y_chunk_shape = Y.chunk_shape[0]

            if y_chunk_shape == 1:
                chunk_op = op.copy().reset_key()
                y_chunk = Y.chunks[0]
                o = cls._gen_out_chunks(x_chunk, y_chunk, chunk_op)
                if chunk_op.return_index:
                    out_distance_chunks.append(o[0])
                    out_index_chunks.append(o[1])
                else:
                    out_distance_chunks.append(o)
            else:
                to_concat_chunks = []
                for j in range(y_chunk_shape):
                    y_chunk = Y.cix[j, 0]
                    chunk_op = op.copy().reset_key()
                    chunk_op._y_offset = y_acc_chunk_shapes[j]
                    chunk_op.stage = OperandStage.map
                    o = chunk_op.new_chunk([x_chunk, y_chunk],
                                           shape=(x_chunk.shape[0], k),
                                           order=TensorOrder.C_ORDER,
                                           index=(i, j))
                    to_concat_chunks.append(o)

                concat_op = TensorConcatenate(axis=1, dtype=to_concat_chunks[0].dtype)
                concat = concat_op.new_chunk(to_concat_chunks,
                                             shape=(x_chunk.shape[0],
                                                    k * y_chunk_shape),
                                             order=TensorOrder.C_ORDER,
                                             index=(i, 0))

                chunk_op = op.copy().reset_key()
                chunk_op.stage = OperandStage.agg
                distance_params = {
                    'shape': (x_chunk.shape[0], k),
                    'order': TensorOrder.C_ORDER,
                    'dtype': np.dtype(np.float64),
                    'index': (i, 0),
                    '_type_': 'distance',
                }
                if op.return_index:
                    index_params = {
                        'shape': (x_chunk.shape[0], k),
                        'order': TensorOrder.C_ORDER,
                        'dtype': np.dtype(np.int64),
                        'index': (i, 0),
                        '_type': 'index',
                    }
                    distance_chunk, index_chunk = chunk_op.new_chunks(
                        [concat], kws=[distance_params, index_params])
                    out_distance_chunks.append(distance_chunk)
                    out_index_chunks.append(index_chunk)
                else:
                    out_distance_chunks.append(chunk_op.new_chunk(
                        [concat], kws=[distance_params]))

        new_op = op.copy()
        nsplits = (tuple(c.shape[0] for c in out_distance_chunks), (k,))
        params = [o.params for o in op.outputs]
        params[0]['chunks'] = out_distance_chunks
        params[0]['nsplits'] = nsplits
        if op.return_index:
            params[1]['chunks'] = out_index_chunks
            params[1]['nsplits'] = nsplits
        return new_op.new_tensors(op.inputs, kws=params)

    @classmethod
    def _topk_reduce_func(cls, dist, start, topk, xp, metric):
        """Reduce a chunk of distances to topk

        Parameters
        ----------
        dist : array of shape (n_samples_chunk, n_samples)
        start : int
            The index in X which the first row of dist corresponds to.
        topk : int

        Returns
        -------
        dist : array of shape (n_samples_chunk, n_neighbors)
        neigh : array of shape (n_samples_chunk, n_neighbors)
        """
        sample_range = xp.arange(dist.shape[0])[:, None]
        if topk - 1 >= dist.shape[1]:
            neigh_ind = xp.repeat(
                xp.arange(dist.shape[1]).reshape(1, -1), dist.shape[0], axis=0)
        else:
            neigh_ind = xp.argpartition(dist, topk - 1, axis=1)
            neigh_ind = neigh_ind[:, :topk]
        # argpartition doesn't guarantee sorted order, so we sort again
        neigh_ind = neigh_ind[
            sample_range, xp.argsort(dist[sample_range, neigh_ind])]
        return dist[sample_range, neigh_ind], neigh_ind

    @classmethod
    def _calcuate_topk_distances(cls, x, y, op, xp):
        metric = op.metric
        reduce_func = partial(cls._topk_reduce_func, topk=op.k,
                              xp=xp, metric=op.metric)
        kwds = op.metric_kwargs or dict()
        need_sqrt = False
        if metric == 'euclidean' and not kwds.get('squared', False):
            need_sqrt = True
            kwds['squared'] = True
        chunked_results = _pariwise_distance_chunked(
            x, y, reduce_func=reduce_func,
            metric=op.metric, working_memory=op.working_memory,
            xp=xp, **kwds)
        neigh_dist, neigh_ind = zip(*chunked_results)
        dist, ind = np.vstack(neigh_dist), np.vstack(neigh_ind)
        if metric == 'euclidean' and need_sqrt:
            dist = xp.sqrt(dist)
        if getattr(op, 'y_offset', None) is not None:
            ind += op.y_offset
        return dist, ind

    @classmethod
    def _execute_map(cls, ctx, op: "PairwiseDistancesTopk"):
        (x, y), device_id, xp = as_same_device(
            [ctx[inp.key] for inp in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            xp = get_array_module(x, nosparse=True)
            ctx[op.outputs[0].key] = cls._calcuate_topk_distances(x, y, op, xp)

    @classmethod
    def _execute_agg(cls, ctx, op: "PairwiseDistancesTopk"):
        inputs, device_id, xp = as_same_device(
            [ctx[inp.key] for inp in op.inputs], device=op.device, ret_extra=True)
        inputs = inputs[0]
        distances = inputs[0]

        with device(device_id):
            dist, ind = cls._topk_reduce_func(distances, 0, op.k,
                                              xp, op.metric)
            ctx[op.outputs[0].key] = dist
            if op.return_index:
                inds = inputs[1]
                ind_result = xp.empty_like(ind)
                for i in range(len(ind_result)):  # pylint: disable=consider-using-enumerate
                    ind_result[i] = inds[i][ind[i]]
                ctx[op.outputs[1].key] = ind_result

    @classmethod
    def _execute(cls, ctx, op: "PairwiseDistancesTopk"):
        (x, y), device_id, xp = as_same_device(
            [ctx[inp.key] for inp in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            xp = get_array_module(x, nosparse=True)
            dist, ind = cls._calcuate_topk_distances(x, y, op, xp)
            dist, ind_on_ind = cls._topk_reduce_func(dist, 0, op.k,
                                                     xp, op.metric)
            ctx[op.outputs[0].key] = dist
            if op.return_index:
                ind_result = xp.empty_like(ind_on_ind)
                for i in range(len(ind_on_ind)):  # pylint: disable=consider-using-enumerate
                    ind_result[i] = ind[i][ind_on_ind[i]]
                ctx[op.outputs[1].key] = ind_result

    @classmethod
    def execute(cls, ctx, op):
        if op.stage == OperandStage.map:
            return cls._execute_map(ctx, op)
        elif op.stage == OperandStage.agg:
            return cls._execute_agg(ctx, op)
        else:
            return cls._execute(ctx, op)


def pairwise_distances_topk(X, Y=None, k=None, metric="euclidean",
                            return_index=True, axis=1, working_memory=None, **kwds):
    if k is None:  # pragma: no cover
        raise ValueError('`k` has to be specified')

    if Y is None:
        Y = X
    if axis == 0:
        X, Y = Y, X
    if working_memory is None:
        working_memory = options.learn.working_memory
    op = PairwiseDistancesTopk(x=X, y=Y, k=k, metric=metric, metric_kwargs=kwds,
                               return_index=return_index, working_memory=working_memory)
    return op(X, Y)
