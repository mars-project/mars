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
from typing import Tuple

import numpy as np

from .... import opcodes as OperandDef
from ....core import recursive_tile
from ....serialization.serializables import KeyField, AnyField, Float16Field
from ....utils import has_unknown_shape, require_module, ensure_own_data
from ...operands import TensorOperand, TensorOperandMixin
from ...core import TensorOrder
from ...datasource import tensor as astensor
from ...array_utils import as_same_device, cp, device


class TensorCdist(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.CDIST

    _xa = KeyField('XA')
    _xb = KeyField('XB')
    _metric = AnyField('metric')
    _p = Float16Field('p', on_serialize=lambda x: float(x) if x is not None else x)
    _w = KeyField('w')
    _v = KeyField('V')
    _vi = KeyField('VI')

    def __init__(self, metric=None, p=None, w=None,
                 v=None, vi=None, **kw):
        super().__init__(_metric=metric, _p=p,
                         _w=w, _v=v, _vi=vi, **kw)

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs)
        self._xa = next(inputs_iter)
        self._xb = next(inputs_iter)
        if self._w is not None:
            self._w = next(inputs_iter)
        if self._v is not None:
            self._v = next(inputs_iter)
        if self._vi is not None:
            self._vi = next(inputs_iter)

    @property
    def xa(self):
        return self._xa

    @property
    def xb(self):
        return self._xb

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

    def __call__(self, xa, xb, shape: Tuple):
        inputs = [xa, xb]
        for val in [self._w, self._v, self._vi]:
            if val is not None:
                inputs.append(val)
        return self.new_tensor(inputs, shape=shape, order=TensorOrder.C_ORDER)

    @classmethod
    def _tile_one_chunk(cls, op, xa, xb, w, v, vi):
        out_tensor = op.outputs[0]
        chunk_op = op.copy().reset_key()
        chunk_inputs = [xa.chunks[0], xb.chunks[0]]
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
    def _tile_chunks(cls, op, xa, xb, w, v, vi):
        out_tensor = op.outputs[0]
        acs, bcs = xa.chunk_shape[0], xb.chunk_shape[0]

        out_chunks = []
        for idx in itertools.product(range(acs), range(bcs)):
            ixa, ixb = idx
            chunk_op = op.copy().reset_key()

            chunk_inputs = []
            xa_chunk = xa.cix[ixa, 0]
            xb_chunk = xb.cix[ixb, 0]
            chunk_inputs.extend([xa_chunk, xb_chunk])
            if w is not None:
                w_chunk = chunk_op._w = w.chunks[0]
                chunk_inputs.append(w_chunk)
            if v is not None:
                v_chunk = chunk_op._v = v.chunks[0]
                chunk_inputs.append(v_chunk)
            if vi is not None:
                vi_chunk = chunk_op._vi = vi.chunks[0]
                chunk_inputs.append(vi_chunk)
            chunk = chunk_op.new_chunk(
                chunk_inputs, shape=(xa_chunk.shape[0], xb_chunk.shape[0]),
                order=out_tensor.order, index=idx)
            out_chunks.append(chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, shape=out_tensor.shape,
                                  order=out_tensor.order,
                                  chunks=out_chunks,
                                  nsplits=(xa.nsplits[0], xb.nsplits[0]))

    @classmethod
    def tile(cls, op):
        # make sure every inputs have known shape
        if has_unknown_shape(*op.inputs):
            yield

        xa = op.xa.rechunk({1: op.xa.shape[1]})
        xb = op.xb.rechunk({1: op.xb.shape[1]})
        xa, xb = yield from recursive_tile(xa, xb)

        # rechunk w, v, vi into one chunk if any of them has value
        extra_inputs = [None] * 3
        for i, ei in enumerate([op.w, op.v, op.vi]):
            if ei is None:
                continue
            new_ei = yield from recursive_tile(ei.rechunk(ei.shape))
            extra_inputs[i] = new_ei
        w, v, vi = extra_inputs

        if len(xa.chunks) == 1 and len(xb.chunks) == 1:
            # only 1 chunk
            return cls._tile_one_chunk(op, xa, xb, w, v, vi)
        else:
            return cls._tile_chunks(op, xa, xb, w, v, vi)

    @classmethod
    def execute(cls, ctx, op):
        from scipy.spatial.distance import cdist

        inputs, device_id, xp = as_same_device(
            [ctx[inp.key] for inp in op.inputs], device=op.device, ret_extra=True)

        if xp is cp:  # pragma: no cover
            raise NotImplementedError('`cdist` does not support running on GPU yet')

        with device(device_id):
            inputs_iter = iter(inputs)
            xa = next(inputs_iter)
            xb = next(inputs_iter)
            kw = dict()
            if op.p is not None:
                kw['p'] = op.p
            if op.w is not None:
                kw['w'] = next(inputs_iter)
            if op.v is not None:
                kw['V'] = next(inputs_iter)
            if op.vi is not None:
                kw['VI'] = next(inputs_iter)

        ctx[op.outputs[0].key] = cdist(
            ensure_own_data(xa), ensure_own_data(xb), metric=op.metric, **kw)


@require_module('scipy.spatial.distance')
def cdist(XA, XB, metric='euclidean', **kwargs):
    """
    Compute distance between each pair of the two collections of inputs.

    See Notes for common calling conventions.

    Parameters
    ----------
    XA : Tensor
        An :math:`m_A` by :math:`n` tensor of :math:`m_A`
        original observations in an :math:`n`-dimensional space.
        Inputs are converted to float type.
    XB : Tensor
        An :math:`m_B` by :math:`n` tensor of :math:`m_B`
        original observations in an :math:`n`-dimensional space.
        Inputs are converted to float type.
    metric : str or callable, optional
        The distance metric to use.  If a string, the distance function can be
        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
        'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
        'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
        'wminkowski', 'yule'.
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
        Default: var(vstack([XA, XB]), axis=0, ddof=1)

        VI : Tensor
        The inverse of the covariance matrix for Mahalanobis.
        Default: inv(cov(vstack([XA, XB].T))).T

        out : Tensor
        The output tensor
        If not None, the distance matrix Y is stored in this tensor.
        Note: metric independent, it will become a regular keyword arg in a
        future scipy version

    Returns
    -------
    Y : Tensor
        A :math:`m_A` by :math:`m_B` distance matrix is returned.
        For each :math:`i` and :math:`j`, the metric
        ``dist(u=XA[i], v=XB[j])`` is computed and stored in the
        :math:`ij` th entry.

    Raises
    ------
    ValueError
        An exception is thrown if `XA` and `XB` do not have
        the same number of columns.

    Notes
    -----
    The following are common calling conventions:

    1. ``Y = cdist(XA, XB, 'euclidean')``

       Computes the distance between :math:`m` points using
       Euclidean distance (2-norm) as the distance metric between the
       points. The points are arranged as :math:`m`
       :math:`n`-dimensional row vectors in the matrix X.

    2. ``Y = cdist(XA, XB, 'minkowski', p=2.)``

       Computes the distances using the Minkowski distance
       :math:`||u-v||_p` (:math:`p`-norm) where :math:`p \\geq 1`.

    3. ``Y = cdist(XA, XB, 'cityblock')``

       Computes the city block or Manhattan distance between the
       points.

    4. ``Y = cdist(XA, XB, 'seuclidean', V=None)``

       Computes the standardized Euclidean distance. The standardized
       Euclidean distance between two n-vectors ``u`` and ``v`` is

       .. math::

          \\sqrt{\\sum {(u_i-v_i)^2 / V[x_i]}}.

       V is the variance vector; V[i] is the variance computed over all
       the i'th components of the points. If not passed, it is
       automatically computed.

    5. ``Y = cdist(XA, XB, 'sqeuclidean')``

       Computes the squared Euclidean distance :math:`||u-v||_2^2` between
       the vectors.

    6. ``Y = cdist(XA, XB, 'cosine')``

       Computes the cosine distance between vectors u and v,

       .. math::

          1 - \\frac{u \\cdot v}
                   {{||u||}_2 {||v||}_2}

       where :math:`||*||_2` is the 2-norm of its argument ``*``, and
       :math:`u \\cdot v` is the dot product of :math:`u` and :math:`v`.

    7. ``Y = cdist(XA, XB, 'correlation')``

       Computes the correlation distance between vectors u and v. This is

       .. math::

          1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}
                   {{||(u - \\bar{u})||}_2 {||(v - \\bar{v})||}_2}

       where :math:`\\bar{v}` is the mean of the elements of vector v,
       and :math:`x \\cdot y` is the dot product of :math:`x` and :math:`y`.


    8. ``Y = cdist(XA, XB, 'hamming')``

       Computes the normalized Hamming distance, or the proportion of
       those vector elements between two n-vectors ``u`` and ``v``
       which disagree. To save memory, the matrix ``X`` can be of type
       boolean.

    9. ``Y = cdist(XA, XB, 'jaccard')``

       Computes the Jaccard distance between the points. Given two
       vectors, ``u`` and ``v``, the Jaccard distance is the
       proportion of those elements ``u[i]`` and ``v[i]`` that
       disagree where at least one of them is non-zero.

    10. ``Y = cdist(XA, XB, 'chebyshev')``

       Computes the Chebyshev distance between the points. The
       Chebyshev distance between two n-vectors ``u`` and ``v`` is the
       maximum norm-1 distance between their respective elements. More
       precisely, the distance is given by

       .. math::

          d(u,v) = \\max_i {|u_i-v_i|}.

    11. ``Y = cdist(XA, XB, 'canberra')``

       Computes the Canberra distance between the points. The
       Canberra distance between two points ``u`` and ``v`` is

       .. math::

         d(u,v) = \\sum_i \\frac{|u_i-v_i|}
                              {|u_i|+|v_i|}.

    12. ``Y = cdist(XA, XB, 'braycurtis')``

       Computes the Bray-Curtis distance between the points. The
       Bray-Curtis distance between two points ``u`` and ``v`` is


       .. math::

            d(u,v) = \\frac{\\sum_i (|u_i-v_i|)}
                          {\\sum_i (|u_i+v_i|)}

    13. ``Y = cdist(XA, XB, 'mahalanobis', VI=None)``

       Computes the Mahalanobis distance between the points. The
       Mahalanobis distance between two points ``u`` and ``v`` is
       :math:`\\sqrt{(u-v)(1/V)(u-v)^T}` where :math:`(1/V)` (the ``VI``
       variable) is the inverse covariance. If ``VI`` is not None,
       ``VI`` will be used as the inverse covariance matrix.

    14. ``Y = cdist(XA, XB, 'yule')``

       Computes the Yule distance between the boolean
       vectors. (see `yule` function documentation)

    15. ``Y = cdist(XA, XB, 'matching')``

       Synonym for 'hamming'.

    16. ``Y = cdist(XA, XB, 'dice')``

       Computes the Dice distance between the boolean vectors. (see
       `dice` function documentation)

    17. ``Y = cdist(XA, XB, 'kulsinski')``

       Computes the Kulsinski distance between the boolean
       vectors. (see `kulsinski` function documentation)

    18. ``Y = cdist(XA, XB, 'rogerstanimoto')``

       Computes the Rogers-Tanimoto distance between the boolean
       vectors. (see `rogerstanimoto` function documentation)

    19. ``Y = cdist(XA, XB, 'russellrao')``

       Computes the Russell-Rao distance between the boolean
       vectors. (see `russellrao` function documentation)

    20. ``Y = cdist(XA, XB, 'sokalmichener')``

       Computes the Sokal-Michener distance between the boolean
       vectors. (see `sokalmichener` function documentation)

    21. ``Y = cdist(XA, XB, 'sokalsneath')``

       Computes the Sokal-Sneath distance between the vectors. (see
       `sokalsneath` function documentation)


    22. ``Y = cdist(XA, XB, 'wminkowski', p=2., w=w)``

       Computes the weighted Minkowski distance between the
       vectors. (see `wminkowski` function documentation)

    23. ``Y = cdist(XA, XB, f)``

       Computes the distance between all pairs of vectors in X
       using the user supplied 2-arity function f. For example,
       Euclidean distance between the vectors could be computed
       as follows::

         dm = cdist(XA, XB, lambda u, v: np.sqrt(((u-v)**2).sum()))

       Note that you should avoid passing a reference to one of
       the distance functions defined in this library. For example,::

         dm = cdist(XA, XB, sokalsneath)

       would calculate the pair-wise distances between the vectors in
       X using the Python function `sokalsneath`. This would result in
       sokalsneath being called :math:`{n \\choose 2}` times, which
       is inefficient. Instead, the optimized C version is more
       efficient, and we call it using the following syntax::

         dm = cdist(XA, XB, 'sokalsneath')

    Examples
    --------
    Find the Euclidean distances between four 2-D coordinates:

    >>> from mars.tensor.spatial import distance
    >>> coords = [(35.0456, -85.2672),
    ...           (35.1174, -89.9711),
    ...           (35.9728, -83.9422),
    ...           (36.1667, -86.7833)]
    >>> distance.cdist(coords, coords, 'euclidean').execute()
    array([[ 0.    ,  4.7044,  1.6172,  1.8856],
           [ 4.7044,  0.    ,  6.0893,  3.3561],
           [ 1.6172,  6.0893,  0.    ,  2.8477],
           [ 1.8856,  3.3561,  2.8477,  0.    ]])


    Find the Manhattan distance from a 3-D point to the corners of the unit
    cube:

    >>> import mars.tensor as mt
    >>> a = mt.array([[0, 0, 0],
    ...               [0, 0, 1],
    ...               [0, 1, 0],
    ...               [0, 1, 1],
    ...               [1, 0, 0],
    ...               [1, 0, 1],
    ...               [1, 1, 0],
    ...               [1, 1, 1]])
    >>> b = mt.array([[ 0.1,  0.2,  0.4]])
    >>> distance.cdist(a, b, 'cityblock').execute()
    array([[ 0.7],
           [ 0.9],
           [ 1.3],
           [ 1.5],
           [ 1.5],
           [ 1.7],
           [ 2.1],
           [ 2.3]])

    """
    XA = astensor(XA, order='C')
    XB = astensor(XB, order='C')

    if XA.issparse() or XB.issparse():
        raise ValueError('Sparse tensors are not supported by this function.')

    s = XA.shape
    sB = XB.shape

    if len(s) != 2:
        raise ValueError('XA must be a 2-dimensional array.')
    if len(sB) != 2:
        raise ValueError('XB must be a 2-dimensional array.')
    if s[1] != sB[1]:
        raise ValueError('XA and XB must have the same number of columns '
                         '(i.e. feature dimension.)')

    mA = s[0]
    mB = sB[0]
    out = kwargs.pop("out", None)
    if out is not None:
        if not hasattr(out, 'shape'):
            raise TypeError('return arrays must be a tensor')
        if out.shape != (mA, mB):
            raise ValueError("Output tensor has incorrect shape.")
        if out.dtype != np.double:
            raise ValueError("Output tensor must be double type.")

    if not isinstance(metric, str) and not callable(metric):
        raise TypeError('3rd argument metric must be a string identifier '
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

    if len(kwargs) > 0:
        raise TypeError(f"`cdist` got an unexpected keyword argument '{next(iter(kwargs))}'")

    op = TensorCdist(metric=metric,
                     p=p, w=w, v=v, vi=vi, dtype=np.dtype(float))
    shape = (XA.shape[0], XB.shape[0])
    ret = op(XA, XB, shape)

    if out is None:
        return ret
    else:
        out.data = ret.data
        return out
