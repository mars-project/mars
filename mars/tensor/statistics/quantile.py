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

from collections.abc import Iterable

import numpy as np

from ... import opcodes as OperandDef
from ...core import ENTITY_TYPE, recursive_tile
from ...core.context import get_context
from ...serialization.serializables import KeyField, AnyField, \
    StringField, BoolField
from ...utils import has_unknown_shape
from ..datasource import tensor as astensor
from ..base import moveaxis, where
from ..indexing import take
from ..arithmetic import isnan, add
from ..reduction import any as tensor_any
from ..operands import TensorOperand, TensorOperandMixin
from ..core import TENSOR_TYPE, TENSOR_CHUNK_TYPE, TensorOrder
from ..utils import check_out_param
from ..array_utils import as_same_device, device
from .core import _ureduce


def _quantile_is_valid(q):
    # avoid expensive reductions, relevant for arrays with < O(1000) elements
    if q.ndim == 1 and q.size < 10:
        for i in range(q.size):
            if q[i] < 0.0 or q[i] > 1.0:
                return False
    else:
        # faster than any()
        if np.count_nonzero(q < 0.0) or np.count_nonzero(q > 1.0):
            return False
    return True


def _quantile_ureduce_func(a, q, axis=None, out=None, overwrite_input=False,
                           interpolation='linear', keepdims=False):
    a = astensor(a)
    out = astensor(out) if out is not None else None

    if q.ndim == 0:
        # Do not allow 0-d arrays because following code fails for scalar
        zerod = True
        q = q[None]
    else:
        zerod = False

    # prepare a for partitioning
    if overwrite_input:
        if axis is None:
            ap = a.ravel()
        else:
            ap = a
    else:
        if axis is None:
            ap = a.flatten()
        else:
            ap = a.copy()

    if axis is None:
        axis = 0

    Nx = ap.shape[axis]
    indices = q * (Nx - 1)

    # round fractional indices according to interpolation method
    if interpolation == 'lower':
        indices = np.floor(indices).astype(np.intp)
    elif interpolation == 'higher':
        indices = np.ceil(indices).astype(np.intp)
    elif interpolation == 'midpoint':
        indices = 0.5 * (np.floor(indices) + np.ceil(indices))
    elif interpolation == 'nearest':
        indices = np.around(indices).astype(np.intp)
    else:
        assert interpolation == 'linear'
        # keep index as fraction and interpolate

    n = np.array(False, dtype=bool)  # check for nan's flag
    if indices.dtype == np.intp:  # take the points along axis
        # Check if the array contains any nan's
        if np.issubdtype(a.dtype, np.inexact):
            indices = np.concatenate((indices, [-1]))

        ap.partition(indices, axis=axis, need_align=True)
        # ensure axis with q-th is first
        ap = moveaxis(ap, axis, 0)
        axis = 0

        # Check if the array contains any nan's
        if np.issubdtype(a.dtype, np.inexact):
            indices = indices[:-1]
            n = isnan(ap[-1:, ...])

        if zerod:
            indices = indices[0]
        r = take(ap, indices, axis=axis, out=out)

    else:  # weight the points above and below the indices
        indices_below = np.floor(indices).astype(np.intp)
        indices_above = indices_below + 1
        indices_above[indices_above > Nx - 1] = Nx - 1

        # Check if the array contains any nan's
        if np.issubdtype(a.dtype, np.inexact):
            indices_above = np.concatenate((indices_above, [-1]))

        weights_above = indices - indices_below
        weights_below = 1 - weights_above

        weights_shape = [1, ] * ap.ndim
        weights_shape[axis] = len(indices)
        weights_below.shape = weights_shape
        weights_above.shape = weights_shape

        ap.partition(np.concatenate((indices_below, indices_above)),
                     axis=axis, need_align=True)

        # ensure axis with q-th is first
        ap = moveaxis(ap, axis, 0)
        weights_below = np.moveaxis(weights_below, axis, 0)
        weights_above = np.moveaxis(weights_above, axis, 0)
        axis = 0

        # Check if the array contains any nan's
        if np.issubdtype(a.dtype, np.inexact):
            indices_above = indices_above[:-1]
            n = isnan(ap[-1:, ...])

        x1 = take(ap, indices_below, axis=axis) * weights_below
        x2 = take(ap, indices_above, axis=axis) * weights_above

        # ensure axis with q-th is first
        x1 = moveaxis(x1, axis, 0)
        x2 = moveaxis(x2, axis, 0)

        if zerod:
            x1 = x1.squeeze(0)
            x2 = x2.squeeze(0)

        if out is not None:
            r = add(x1, x2, out=out)
        else:
            r = add(x1, x2)

    if isinstance(n, TENSOR_TYPE):
        if zerod:
            if ap.ndim == 1:
                r.data = where(tensor_any(n), a.dtype.type(np.nan), r).data
                if out is not None:
                    out.data = r.data
            else:
                r[:] = where(tensor_any(n),
                             where(n.squeeze(0), a.dtype.type(np.nan), r),
                             r)
        else:
            if r.ndim == 1:
                r[:] = where(tensor_any(n), np.full(r.shape, a.dtype.type(np.nan)), r)
            else:
                r[:] = where(tensor_any(n),
                             where(n.repeat(q.size, 0), a.dtype.type(np.nan), r),
                             r)

    return r


q_error_msg = 'Quantiles must be in the range [0, 1]'


class TensorQuantile(TensorOperand, TensorOperandMixin):
    __slots__ = 'q_error_msg',
    _op_type_ = OperandDef.QUANTILE

    _a = KeyField('a')
    _q = AnyField('q')
    _axis = AnyField('axis')
    _out = KeyField('out')
    _overwrite_input = BoolField('overwrite_input')
    _interpolation = StringField('interpolation')
    _keepdims = BoolField('keepdims')

    def __init__(self, q=None, axis=None, out=None, overwrite_input=None,
                 interpolation=None, keepdims=None, **kw):
        self.q_error_msg = kw.pop('q_error_msg', q_error_msg)
        super().__init__(_q=q, _axis=axis, _interpolation=interpolation,
                         _out=out, _overwrite_input=overwrite_input,
                         _keepdims=keepdims, **kw)

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._a = self._inputs[0]
        if isinstance(self._q, (TENSOR_TYPE, TENSOR_CHUNK_TYPE)):
            self._q = self._inputs[1]
        if isinstance(self._out, (TENSOR_TYPE, TENSOR_CHUNK_TYPE)):
            self._out = self._inputs[-1]

    @property
    def a(self):
        return self._a

    @property
    def q(self):
        return self._q

    @property
    def axis(self):
        return self._axis

    @property
    def out(self):
        return self._out

    @property
    def overwrite_input(self):
        return self._overwrite_input

    @property
    def interpolation(self):
        return self._interpolation

    @property
    def keepdims(self):
        return self._keepdims

    def __call__(self, a, q=None, out=None):
        shape = [self._q.size] if self._q.ndim > 0 else []
        if self._axis is None:
            exclude_axes = set(range(a.ndim))
        elif isinstance(self._axis, tuple):
            exclude_axes = set(self._axis)
        else:
            exclude_axes = {self._axis}
        for ax, s in enumerate(a.shape):
            if ax not in exclude_axes:
                shape.append(s)
            elif self._keepdims:
                shape.append(1)
        inputs = [a] if q is None else [a, q]
        order = TensorOrder.C_ORDER
        if out is not None:
            inputs.append(out)
            order = out.order
            shape = out.shape
        t = self.new_tensor(inputs, shape=tuple(shape), order=order)
        if out is not None:
            check_out_param(out, t, 'same_kind')
            out.data = t.data
            return out
        else:
            return t

    @classmethod
    def _tile(cls, op, q):
        r, k = _ureduce(op.a, func=_quantile_ureduce_func, q=q,
                        axis=op.axis, out=op.out,
                        overwrite_input=op.overwrite_input,
                        interpolation=op.interpolation)
        if op.keepdims:
            return r.reshape(q.shape + k)
        else:
            return r

    @classmethod
    def _tile_one_chunk(cls, op, q):
        in_tensor = op.inputs[0]
        out_tensor = op.outputs[0]
        chunk_op = op.copy().reset_key()
        chunk_op._q = q
        chunk_inputs = [in_tensor.chunks[0]]
        if op.out is not None:
            chunk_inputs.append(op.out.chunks[0])
        chunk = chunk_op.new_chunk(chunk_inputs, shape=out_tensor.shape,
                                   index=(0,) * out_tensor.ndim,
                                   order=out_tensor.order)
        op = op.copy()
        return op.new_tensors(op.inputs, shape=out_tensor.shape,
                              order=out_tensor.order,
                              nsplits=tuple((s,) for s in out_tensor.shape),
                              chunks=[chunk])

    @classmethod
    def tile(cls, op):
        if isinstance(op.q, TENSOR_TYPE):
            # trigger execution of `q`
            yield op.q.chunks

            ctx = get_context()
            # get q's data
            q_chunk_keys = [c.key for c in op.q.chunks]
            q_data = ctx.get_chunks_result(q_chunk_keys)
            op._q = q = np.concatenate(q_data)
            if not _quantile_is_valid(q):
                raise ValueError(op.q_error_msg)
        else:
            if has_unknown_shape(*op.inputs):
                yield
            q = np.asarray(op.q)

        if len(op.a.chunks) == 1 and (op.out is None or len(op.out.chunks) == 1):
            return cls._tile_one_chunk(op, q)
        else:
            tiled = yield from recursive_tile(cls._tile(op, q))
            return [tiled]

    @classmethod
    def execute(cls, ctx, op):
        inputs, device_id, xp = as_same_device(
            [ctx[inp.key] for inp in op.inputs], device=op.device, ret_extra=True)
        a = inputs[0]
        out = inputs[-1].copy() if op.out is not None else None

        with device(device_id):
            ctx[op.outputs[0].key] = xp.quantile(a, q=op.q, axis=op.axis, out=out,
                                                 interpolation=op.interpolation,
                                                 keepdims=op.keepdims)


INTERPOLATION_TYPES = {'linear', 'lower', 'higher', 'midpoint', 'nearest'}


def _quantile_unchecked(a, q, axis=None, out=None, overwrite_input=False,
                        interpolation='linear', keepdims=False,
                        q_error_msg=None, handle_non_numeric=None):
    a = astensor(a)
    raw_dtype = a.dtype
    need_view_back = False
    if handle_non_numeric and not np.issubdtype(a.dtype, np.number):
        # enable handle_non_numeric is often used
        # to handle the datetime-like dtype
        a = a.astype('i8')
        need_view_back = True
    if isinstance(q, ENTITY_TYPE):
        q = astensor(q)
        # do check in tile
        q_input = q
    else:
        q_input = None

    if isinstance(axis, Iterable):
        axis = tuple(axis)

    if q.ndim > 1:
        raise ValueError('`q` should be a scalar or array of float')

    if out is not None and not isinstance(out, TENSOR_TYPE):
        raise TypeError(f'`out` should be a tensor, got {type(out)}')

    if interpolation not in INTERPOLATION_TYPES:
        raise ValueError("interpolation can only be 'linear', 'lower' "
                         "'higher', 'midpoint', or 'nearest'")

    # infer dtype
    q_tiny = np.random.rand(2 if q.size % 2 == 0 else 1).astype(q.dtype)
    if handle_non_numeric and not np.issubdtype(a.dtype, np.number):
        dtype = a.dtype
    else:
        dtype = np.quantile(np.empty(1, dtype=a.dtype), q_tiny,
                            interpolation=interpolation).dtype
    op = TensorQuantile(q=q, axis=axis, out=out, overwrite_input=overwrite_input,
                        interpolation=interpolation, keepdims=keepdims,
                        handle_non_numeric=handle_non_numeric, q_error_msg=q_error_msg,
                        dtype=dtype, gpu=a.op.gpu)
    ret = op(a, q=q_input, out=out)
    if need_view_back:
        ret = ret.astype(raw_dtype)
    return ret


def quantile(a, q, axis=None, out=None, overwrite_input=False,
             interpolation='linear', keepdims=False, **kw):
    """
    Compute the q-th quantile of the data along the specified axis.

    Parameters
    ----------
    a : array_like
        Input tensor or object that can be converted to a tensor.
    q : array_like of float
        Quantile or sequence of quantiles to compute, which must be between
        0 and 1 inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the quantiles are computed. The
        default is to compute the quantile(s) along a flattened
        version of the tensor.
    out : Tensor, optional
        Alternative output tensor in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
        Just for compatibility with Numpy, would not take effect.
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        This optional parameter specifies the interpolation method to
        use when the desired quantile lies between two data points
        ``i < j``:

            * linear: ``i + (j - i) * fraction``, where ``fraction``
              is the fractional part of the index surrounded by ``i``
              and ``j``.
            * lower: ``i``.
            * higher: ``j``.
            * nearest: ``i`` or ``j``, whichever is nearest.
            * midpoint: ``(i + j) / 2``.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the
        result will broadcast correctly against the original tensor `a`.

    Returns
    -------
    quantile : scalar or Tensor
        If `q` is a single quantile and `axis=None`, then the result
        is a scalar. If multiple quantiles are given, first axis of
        the result corresponds to the quantiles. The other axes are
        the axes that remain after the reduction of `a`. If the input
        contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the
        same as that of the input. If `out` is specified, that tensor is
        returned instead.

    See Also
    --------
    mean
    percentile : equivalent to quantile, but with q in the range [0, 100].
    median : equivalent to ``quantile(..., 0.5)``
    nanquantile

    Notes
    -----
    Given a vector ``V`` of length ``N``, the q-th quantile of
    ``V`` is the value ``q`` of the way from the minimum to the
    maximum in a sorted copy of ``V``. The values and distances of
    the two nearest neighbors as well as the `interpolation` parameter
    will determine the quantile if the normalized ranking does not
    match the location of ``q`` exactly. This function is the same as
    the median if ``q=0.5``, the same as the minimum if ``q=0.0`` and the
    same as the maximum if ``q=1.0``.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> a = mt.array([[10, 7, 4], [3, 2, 1]])
    >>> a.execute()
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> mt.quantile(a, 0.5).execute()
    3.5
    >>> mt.quantile(a, 0.5, axis=0).execute()
    array([6.5, 4.5, 2.5])
    >>> mt.quantile(a, 0.5, axis=1).execute()
    array([7.,  2.])
    >>> mt.quantile(a, 0.5, axis=1, keepdims=True).execute()
    array([[7.],
           [2.]])
    >>> m = mt.quantile(a, 0.5, axis=0)
    >>> out = mt.zeros_like(m)
    >>> mt.quantile(a, 0.5, axis=0, out=out).execute()
    array([6.5, 4.5, 2.5])
    >>> m.execute()
    array([6.5, 4.5, 2.5])
    """

    handle_non_numeric = kw.pop('handle_non_numeric', None)
    if len(kw) > 0:  # pragma: no cover
        raise TypeError('quantile() got an unexpected keyword '
                        f'argument \'{next(iter(kw))}\'')

    if not isinstance(q, ENTITY_TYPE):
        q = np.asanyarray(q)
        # do check instantly if q is not a tensor
        if not _quantile_is_valid(q):
            raise ValueError(q_error_msg)

    return _quantile_unchecked(a, q, axis=axis, out=out, overwrite_input=overwrite_input,
                               interpolation=interpolation, keepdims=keepdims,
                               handle_non_numeric=handle_non_numeric)
