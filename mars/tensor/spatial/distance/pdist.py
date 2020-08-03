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

from typing import List, Tuple

import numpy as np

from .... import opcodes as OperandDef
from ....operands import OperandStage
from ....serialize import ValueType, KeyField, AnyField, Float16Field, Int32Field, TupleField
from ....tiles import TilesError
from ....utils import check_chunks_unknown_shape, get_shuffle_input_keys_idxes, \
    require_module
from ....config import options
from ...operands import TensorMapReduceOperand, TensorOperandMixin, TensorShuffleProxy
from ...array_utils import as_same_device, device, cp
from ...core import TensorOrder
from ...datasource.array import tensor as astensor


class TensorPdist(TensorMapReduceOperand, TensorOperandMixin):
    _op_type_ = OperandDef.PDIST

    _input = KeyField('input')
    _metric = AnyField('metric')
    _p = Float16Field('p')
    _w = KeyField('w')
    _v = KeyField('V')
    _vi = KeyField('VI')
    _aggregate_size = Int32Field('aggregate_size')

    _a = KeyField('a')
    _a_offset = Int32Field('a_offset')
    _b = KeyField('b')
    _b_offset = Int32Field('b_offset')
    _out_sizes = TupleField('out_sizes', ValueType.int32)
    _n = Int32Field('n')

    def __init__(self, metric=None, p=None, w=None, v=None, vi=None,
                 a=None, a_offset=None, b=None, b_offset=None, out_sizes=None, n=None,
                 aggregate_size=None, stage=None, shuffle_key=None, dtype=None, **kw):
        super().__init__(_metric=metric, _p=p, _w=w, _v=v, _vi=vi,
                         _a=a, _a_offset=a_offset, _b=b, _b_offset=b_offset, _out_sizes=out_sizes,
                         _n=n, _dtype=dtype, _aggregate_size=aggregate_size, _stage=stage,
                         _shuffle_key=shuffle_key, **kw)

    def _set_inputs(self, inputs: List) -> None:
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs)

        if self.stage == OperandStage.map:
            self._a = next(inputs_iter)
            if self._b is not None:
                self._b = next(inputs_iter)
        else:
            self._input = next(inputs_iter)

        if self._w is not None:
            self._w = next(inputs_iter)
        if self._v is not None:
            self._v = next(inputs_iter)
        if self._vi is not None:
            self._vi = next(inputs_iter)

    @property
    def input(self):
        return self._input

    @property
    def metric(self):
        return self._metric

    @property
    def p(self):
        return self._p

    @property
    def w(self):
        return self._w

    @property
    def v(self):
        return self._v

    @property
    def vi(self):
        return self._vi

    @property
    def aggregate_size(self):
        return self._aggregate_size

    @property
    def a(self):
        return self._a

    @property
    def a_offset(self):
        return self._a_offset

    @property
    def b(self):
        return self._b

    @property
    def b_offset(self):
        return self._b_offset

    @property
    def out_sizes(self):
        return self._out_sizes

    @property
    def n(self):
        return self._n

    def __call__(self, x, shape: Tuple):
        inputs = [x]
        for val in [self._w, self._v, self._vi]:
            if val is not None:
                inputs.append(val)
        return self.new_tensor(inputs, shape=shape, order=TensorOrder.C_ORDER)

    @classmethod
    def _tile_one_chunk(cls, op, in_tensor, w, v, vi):
        out_tensor = op.outputs[0]
        chunk_op = op.copy().reset_key()
        chunk_inputs = [in_tensor.chunks[0]]
        for val in [w, v, vi]:
            if val is not None:
                chunk_inputs.append(val.chunks[0])
        chunk = chunk_op.new_chunk(chunk_inputs, shape=out_tensor.shape,
                                   order=out_tensor.order, index=(0,) * out_tensor.ndim)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, shape=out_tensor.shape,
                                  order=out_tensor.order,
                                  nsplits=tuple((s,) for s in out_tensor.shape),
                                  chunks=[chunk])

    @classmethod
    def _tile_chunks(cls, op, in_tensor, w, v, vi):
        out_tensor = op.outputs[0]
        extra_inputs = []
        for val in [w, v, vi]:
            if val is not None:
                extra_inputs.append(val.chunks[0])

        n = in_tensor.shape[0]
        aggregate_size = op.aggregate_size
        if aggregate_size is None:
            aggregate_size = np.ceil(out_tensor.size * out_tensor.dtype.itemsize /
                                     options.chunk_store_limit).astype(int).item()
        out_sizes = [out_tensor.size // aggregate_size for _ in range(aggregate_size)]
        for i in range(out_tensor.size % aggregate_size):
            out_sizes[i] += 1

        chunk_size = in_tensor.chunk_shape[0]
        map_chunks = []
        axis_0_cum_size = np.cumsum(in_tensor.nsplits[0]).tolist()
        for i in range(chunk_size):
            for j in range(i, chunk_size):
                kw = {
                    'stage': OperandStage.map,
                    'a': in_tensor.cix[i, 0],
                    'a_offset': axis_0_cum_size[i - 1] if i > 0 else 0,
                    'out_sizes': tuple(out_sizes),
                    'n': n,
                    'metric': op.metric,
                    'p': op.p,
                    'w': w.chunks[0] if w is not None else None,
                    'v': v.chunks[0] if v is not None else None,
                    'vi': vi.chunks[0] if vi is not None else None,
                    'dtype': out_tensor.dtype
                }
                if i != j:
                    kw['b'] = in_tensor.cix[j, 0]
                    kw['b_offset'] = axis_0_cum_size[j - 1] if j > 0 else 0
                map_op = TensorPdist(**kw)
                map_chunk_inputs = [kw['a']]
                if 'b' in kw:
                    map_chunk_inputs.append(kw['b'])
                if kw['w'] is not None:
                    map_chunk_inputs.append(kw['w'])
                if kw['v'] is not None:
                    map_chunk_inputs.append(kw['v'])
                if kw['vi'] is not None:
                    map_chunk_inputs.append(kw['vi'])
                # calc chunk shape
                if i == j:
                    a_axis_0_size = kw['a'].shape[0]
                    chunk_shape = (a_axis_0_size * (a_axis_0_size - 1) // 2,)
                else:
                    chunk_shape = (kw['a'].shape[0] * kw['b'].shape[0],)
                map_chunk = map_op.new_chunk(map_chunk_inputs, shape=chunk_shape,
                                             order=out_tensor.order,
                                             index=(i * chunk_size + j,))
                map_chunks.append(map_chunk)

        proxy_chunk = TensorShuffleProxy(dtype=out_tensor.dtype).new_chunk(
            map_chunks, shape=())

        reduce_chunks = []
        for p in range(aggregate_size):
            reduce_chunk_op = TensorPdist(
                stage=OperandStage.reduce, shuffle_key=str(p), dtype=out_tensor.dtype)
            reduce_chunk = reduce_chunk_op.new_chunk(
                [proxy_chunk], shape=(out_sizes[p],), order=out_tensor.order,
                index=(p,))
            reduce_chunks.append(reduce_chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, shape=out_tensor.shape,
                                  order=out_tensor.order,
                                  nsplits=(tuple(out_sizes),),
                                  chunks=reduce_chunks)

    @classmethod
    def tile(cls, op):
        # make sure every inputs have known shape
        check_chunks_unknown_shape(op.inputs, TilesError)

        in_tensor = op.input.rechunk({1: op.input.shape[1]})._inplace_tile()
        # rechunk w, v, vi into one chunk if any of them has value
        extra_inputs = [None] * 3
        for i, ei in enumerate([op.w, op.v, op.vi]):
            if ei is None:
                continue
            new_ei = ei.rechunk(ei.shape)._inplace_tile()
            extra_inputs[i] = new_ei
        w, v, vi = extra_inputs

        if len(in_tensor.chunks) == 1:
            # only 1 chunk
            return cls._tile_one_chunk(op, in_tensor, w, v, vi)
        else:
            return cls._tile_chunks(op, in_tensor, w, v, vi)

    @classmethod
    def _execute_map(cls, ctx, op):
        from scipy.spatial.distance import pdist, cdist

        inputs, device_id, xp = as_same_device(
            [ctx[inp.key] for inp in op.inputs], device=op.device, ret_extra=True)

        if xp is cp:  # pragma: no cover
            raise NotImplementedError('`pdist` does not support running on GPU yet')

        with device(device_id):
            inputs_iter = iter(inputs)
            a = next(inputs_iter)
            if op.b is not None:
                b = next(inputs_iter)
            else:
                b = None
            kw = dict()
            if op.p is not None:
                kw['p'] = op.p
            if op.w is not None:
                kw['w'] = next(inputs_iter)
            if op.v is not None:
                kw['V'] = next(inputs_iter)
            if op.vi is not None:
                kw['VI'] = next(inputs_iter)
            metric = op.metric

            if b is None:
                # one input, pdist on same chunk
                dists = pdist(a, metric=metric, **kw)
                i_indices, j_indices = xp.triu_indices(a.shape[0], k=1)
                i_indices += op.a_offset
                j_indices += op.a_offset
            else:
                # two inputs, pdist on different chunks
                dists = cdist(a, b, metric=metric, **kw).ravel()
                mgrid = \
                    xp.mgrid[op.a_offset: op.a_offset + a.shape[0],
                    op.b_offset: op.b_offset + b.shape[0]]
                i_indices, j_indices = mgrid[0].ravel(), mgrid[1].ravel()

            out_row_sizes = xp.arange(op.n - 1, -1, -1)
            out_row_cum_sizes = xp.empty((op.n + 1,), dtype=int)
            out_row_cum_sizes[0] = 0
            xp.cumsum(out_row_sizes, out=out_row_cum_sizes[1:])
            indices = out_row_cum_sizes[i_indices] + j_indices - \
                      (op.n - out_row_sizes[i_indices])

            # save as much memory as possible
            del i_indices, j_indices, out_row_sizes, out_row_cum_sizes

            out_cum_size = xp.cumsum(op.out_sizes)
            out = op.outputs[0]
            for i in range(len(op.out_sizes)):
                start_index = out_cum_size[i - 1] if i > 0 else 0
                end_index = out_cum_size[i]
                to_filter = (indices >= start_index) & (indices < end_index)
                downside_indices = indices[to_filter] - start_index
                downside_dists = dists[to_filter]
                ctx[out.key, str(i)] = (downside_indices, downside_dists)

    @classmethod
    def _execute_single(cls, ctx, op):
        from scipy.spatial.distance import pdist

        inputs, device_id, xp = as_same_device(
            [ctx[inp.key] for inp in op.inputs], device=op.device, ret_extra=True)

        if xp is cp:  # pragma: no cover
            raise NotImplementedError('`pdist` does not support running on GPU yet')

        with device(device_id):
            inputs_iter = iter(inputs)
            x = next(inputs_iter)
            kw = dict()
            if op.p is not None:
                kw['p'] = op.p
            if op.w is not None:
                kw['w'] = next(inputs_iter)
            if op.v is not None:
                kw['V'] = next(inputs_iter)
            if op.vi is not None:
                kw['VI'] = next(inputs_iter)

        ctx[op.outputs[0].key] = pdist(x, metric=op.metric, **kw)

    @classmethod
    def _execute_reduce(cls, ctx, op):
        input_keys, _ = get_shuffle_input_keys_idxes(op.inputs[0])
        raw_inputs = [ctx[(input_key, op.shuffle_key)] for input_key in input_keys]
        raw_indices = [inp[0] for inp in raw_inputs]
        raw_dists = [inp[1] for inp in raw_inputs]
        inputs, device_id, xp = as_same_device(
            raw_indices + raw_dists, op.device, ret_extra=True)
        raw_indices = inputs[:len(raw_indices)]
        raw_dists = inputs[len(raw_indices):]
        output = op.outputs[0]

        with device(device_id):
            indices = xp.concatenate(raw_indices)
            dists = xp.concatenate(raw_dists)
            out_dists = xp.empty(output.shape, dtype=float)
            out_dists[indices] = dists
            ctx[output.key] = out_dists

    @classmethod
    def execute(cls, ctx, op):
        if op.stage == OperandStage.map:
            cls._execute_map(ctx, op)
        elif op.stage == OperandStage.reduce:
            cls._execute_reduce(ctx, op)
        else:
            cls._execute_single(ctx, op)


@require_module('scipy.spatial.distance')
def pdist(X, metric='euclidean', **kwargs):
    """
    Pairwise distances between observations in n-dimensional space.

    See Notes for common calling conventions.

    Parameters
    ----------
    X : Tensor
        An m by n tensor of m original observations in an
        n-dimensional space.
    metric : str or function, optional
        The distance metric to use. The distance function can
        be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
        'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
        'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching',
        'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
        'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.
    **kwargs : dict, optional
        Extra arguments to `metric`: refer to each metric documentation for a
        list of all possible arguments.

        Some possible arguments:

        p : scalar
        The p-norm to apply for Minkowski, weighted and unweighted.
        Default: 2.

        w : Tensor
        The weight vector for metrics that support weights (e.g., Minkowski).

        V : Tensor
        The variance vector for standardized Euclidean.
        Default: var(X, axis=0, ddof=1)

        VI : Tensor
        The inverse of the covariance matrix for Mahalanobis.
        Default: inv(cov(X.T)).T

        out : Tensor.
        The output tensor
        If not None, condensed distance matrix Y is stored in this tensor.
        Note: metric independent, it will become a regular keyword arg in a
        future scipy version

    Returns
    -------
    Y : Tensor
        Returns a condensed distance matrix Y.  For
        each :math:`i` and :math:`j` (where :math:`i<j<m`),where m is the number
        of original observations. The metric ``dist(u=X[i], v=X[j])``
        is computed and stored in entry ``ij``.

    See Also
    --------
    squareform : converts between condensed distance matrices and
                 square distance matrices.

    Notes
    -----
    See ``squareform`` for information on how to calculate the index of
    this entry or to convert the condensed distance matrix to a
    redundant square matrix.

    The following are common calling conventions.

    1. ``Y = pdist(X, 'euclidean')``

       Computes the distance between m points using Euclidean distance
       (2-norm) as the distance metric between the points. The points
       are arranged as m n-dimensional row vectors in the matrix X.

    2. ``Y = pdist(X, 'minkowski', p=2.)``

       Computes the distances using the Minkowski distance
       :math:`||u-v||_p` (p-norm) where :math:`p \\geq 1`.

    3. ``Y = pdist(X, 'cityblock')``

       Computes the city block or Manhattan distance between the
       points.

    4. ``Y = pdist(X, 'seuclidean', V=None)``

       Computes the standardized Euclidean distance. The standardized
       Euclidean distance between two n-vectors ``u`` and ``v`` is

       .. math::

          \\sqrt{\\sum {(u_i-v_i)^2 / V[x_i]}}


       V is the variance vector; V[i] is the variance computed over all
       the i'th components of the points.  If not passed, it is
       automatically computed.

    5. ``Y = pdist(X, 'sqeuclidean')``

       Computes the squared Euclidean distance :math:`||u-v||_2^2` between
       the vectors.

    6. ``Y = pdist(X, 'cosine')``

       Computes the cosine distance between vectors u and v,

       .. math::

          1 - \\frac{u \\cdot v}
                   {{||u||}_2 {||v||}_2}

       where :math:`||*||_2` is the 2-norm of its argument ``*``, and
       :math:`u \\cdot v` is the dot product of ``u`` and ``v``.

    7. ``Y = pdist(X, 'correlation')``

       Computes the correlation distance between vectors u and v. This is

       .. math::

          1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}
                   {{||(u - \\bar{u})||}_2 {||(v - \\bar{v})||}_2}

       where :math:`\\bar{v}` is the mean of the elements of vector v,
       and :math:`x \\cdot y` is the dot product of :math:`x` and :math:`y`.

    8. ``Y = pdist(X, 'hamming')``

       Computes the normalized Hamming distance, or the proportion of
       those vector elements between two n-vectors ``u`` and ``v``
       which disagree. To save memory, the matrix ``X`` can be of type
       boolean.

    9. ``Y = pdist(X, 'jaccard')``

       Computes the Jaccard distance between the points. Given two
       vectors, ``u`` and ``v``, the Jaccard distance is the
       proportion of those elements ``u[i]`` and ``v[i]`` that
       disagree.

    10. ``Y = pdist(X, 'chebyshev')``

       Computes the Chebyshev distance between the points. The
       Chebyshev distance between two n-vectors ``u`` and ``v`` is the
       maximum norm-1 distance between their respective elements. More
       precisely, the distance is given by

       .. math::

          d(u,v) = \\max_i {|u_i-v_i|}

    11. ``Y = pdist(X, 'canberra')``

       Computes the Canberra distance between the points. The
       Canberra distance between two points ``u`` and ``v`` is

       .. math::

         d(u,v) = \\sum_i \\frac{|u_i-v_i|}
                              {|u_i|+|v_i|}


    12. ``Y = pdist(X, 'braycurtis')``

       Computes the Bray-Curtis distance between the points. The
       Bray-Curtis distance between two points ``u`` and ``v`` is


       .. math::

            d(u,v) = \\frac{\\sum_i {|u_i-v_i|}}
                           {\\sum_i {|u_i+v_i|}}

    13. ``Y = pdist(X, 'mahalanobis', VI=None)``

       Computes the Mahalanobis distance between the points. The
       Mahalanobis distance between two points ``u`` and ``v`` is
       :math:`\\sqrt{(u-v)(1/V)(u-v)^T}` where :math:`(1/V)` (the ``VI``
       variable) is the inverse covariance. If ``VI`` is not None,
       ``VI`` will be used as the inverse covariance matrix.

    14. ``Y = pdist(X, 'yule')``

       Computes the Yule distance between each pair of boolean
       vectors. (see yule function documentation)

    15. ``Y = pdist(X, 'matching')``

       Synonym for 'hamming'.

    16. ``Y = pdist(X, 'dice')``

       Computes the Dice distance between each pair of boolean
       vectors. (see dice function documentation)

    17. ``Y = pdist(X, 'kulsinski')``

       Computes the Kulsinski distance between each pair of
       boolean vectors. (see kulsinski function documentation)

    18. ``Y = pdist(X, 'rogerstanimoto')``

       Computes the Rogers-Tanimoto distance between each pair of
       boolean vectors. (see rogerstanimoto function documentation)

    19. ``Y = pdist(X, 'russellrao')``

       Computes the Russell-Rao distance between each pair of
       boolean vectors. (see russellrao function documentation)

    20. ``Y = pdist(X, 'sokalmichener')``

       Computes the Sokal-Michener distance between each pair of
       boolean vectors. (see sokalmichener function documentation)

    21. ``Y = pdist(X, 'sokalsneath')``

       Computes the Sokal-Sneath distance between each pair of
       boolean vectors. (see sokalsneath function documentation)

    22. ``Y = pdist(X, 'wminkowski', p=2, w=w)``

       Computes the weighted Minkowski distance between each pair of
       vectors. (see wminkowski function documentation)

    23. ``Y = pdist(X, f)``

       Computes the distance between all pairs of vectors in X
       using the user supplied 2-arity function f. For example,
       Euclidean distance between the vectors could be computed
       as follows::

         dm = pdist(X, lambda u, v: np.sqrt(((u-v)**2).sum()))

       Note that you should avoid passing a reference to one of
       the distance functions defined in this library. For example,::

         dm = pdist(X, sokalsneath)

       would calculate the pair-wise distances between the vectors in
       X using the Python function sokalsneath. This would result in
       sokalsneath being called :math:`{n \\choose 2}` times, which
       is inefficient. Instead, the optimized C version is more
       efficient, and we call it using the following syntax.::

         dm = pdist(X, 'sokalsneath')

    """

    X = astensor(X, order='C')

    if X.issparse():
        raise ValueError('Sparse tensors are not supported by this function.')

    s = X.shape
    if len(s) != 2:
        raise ValueError('A 2-dimensional tensor must be passed.')

    m = s[0]
    out = kwargs.pop("out", None)
    if out is not None:
        if not hasattr(out, 'shape'):
            raise TypeError('return arrays must be a tensor')
        if out.shape != (m * (m - 1) // 2,):
            raise ValueError("output tensor has incorrect shape.")
        if out.dtype != np.double:
            raise ValueError("Output tensor must be double type.")

    if not callable(metric) and not isinstance(metric, str):
        raise TypeError('2nd argument metric must be a string identifier '
                        'or a function.')

    p = kwargs.pop('p', None)
    w = kwargs.pop('w', None)
    if w is not None:
        w = astensor(w)
    v = kwargs.pop('V', None)
    if v is not None:
        v = astensor(v)
    vi = kwargs.pop('VI', None)
    if vi is not None:
        vi = astensor(vi)
    aggregate_size = kwargs.pop('aggregate_size', None)

    if len(kwargs) > 0:
        raise TypeError('`pdist` got an unexpected keyword argument \'{}\''.format(
            next(n for n in kwargs)))

    op = TensorPdist(metric=metric,
                     p=p, w=w, v=v, vi=vi, aggregate_size=aggregate_size,
                     dtype=np.dtype(float))
    shape = (m * (m - 1) // 2,)
    ret = op(X, shape)

    if out is None:
        return ret
    else:
        out.data = ret.data
        return out
