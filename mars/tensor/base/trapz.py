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
from ...core import recursive_tile
from ...serialization.serializables import KeyField, Float64Field, Int8Field
from ...utils import has_unknown_shape
from ..array_utils import as_same_device, device
from ..core import TensorOrder
from ..datasource import tensor as astensor
from ..operands import TensorOperand, TensorOperandMixin
from ..utils import validate_axis


class TensorTrapz(TensorOperand, TensorOperandMixin):
    _op_type_ = opcodes.TRAPZ

    _y = KeyField('y')
    _x = KeyField('x')
    _dx = Float64Field('dx')
    _axis = Int8Field('axis')

    def __init__(self, y=None, x=None, dx=None, axis=None, **kw):
        super().__init__(_y=y, _x=x, _dx=dx, _axis=axis, **kw)

    @property
    def y(self):
        return self._y

    @property
    def x(self):
        return self._x

    @property
    def dx(self):
        return self._dx

    @property
    def axis(self):
        return self._axis

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._y = self._inputs[0]
        if self._x is not None:
            self._x = self._inputs[-1]

    def __call__(self, y, x=None):
        inputs = [y]
        order = y.order
        if x is not None:
            x = astensor(x)
            inputs.append(x)
            if x.order == TensorOrder.C_ORDER:
                order = TensorOrder.C_ORDER

        shape = tuple(s for ax, s in enumerate(y.shape)
                      if ax != self._axis)
        dtype = np.trapz(np.empty(1, dtype=y.dtype)).dtype
        return self.new_tensor(inputs, shape=shape, dtype=dtype,
                               order=order)

    @classmethod
    def tile(cls, op: "TensorTrapz"):
        from .diff import diff

        y = astensor(op.y)
        x = op.x
        axis = op.axis

        if x is not None:
            x = astensor(x)
            # rechunk x to make x.nsplits == y.nsplits
            if has_unknown_shape(x, y):
                yield
            x = yield from recursive_tile(x.rechunk(y.nsplits))

        if len(y.chunks) == 1:
            return cls._tile_one_chunk(op, y, x)

        if x is None:
            d = op.dx
        else:
            if x.ndim == 1:
                d = diff(x)
                # reshape to correct shape
                shape = [1]*y.ndim
                shape[axis] = d.shape[0]
                d = d.reshape(shape)
            else:
                d = diff(x, axis=axis)
        nd = y.ndim
        slice1 = [slice(None)]*nd
        slice2 = [slice(None)]*nd
        slice1[axis] = slice(1, None)
        slice2[axis] = slice(None, -1)
        ret = (d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0).sum(axis)
        return [(yield from recursive_tile(ret))]

    @classmethod
    def _tile_one_chunk(cls, op, y, x):
        out = op.outputs[0]
        chunk_op = op.copy().reset_key()
        inputs = [y.chunks[0]]
        if x is not None:
            inputs.append(x.chunks[0])
        chunk = chunk_op.new_chunk(inputs, shape=out.shape,
                                   order=out.order,
                                   index=(0,) * out.ndim)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, shape=out.shape, order=out.order,
                                  nsplits=tuple((s,) for s in out.shape),
                                  chunks=[chunk])

    @classmethod
    def execute(cls, ctx, op: "TensorTrapz"):
        inputs, device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        y = inputs[0]
        if len(inputs) > 1:
            x = inputs[-1]
        else:
            x = None

        with device(device_id):
            ctx[op.outputs[0].key] = xp.trapz(y, x=x, dx=op.dx,
                                              axis=op.axis)


def trapz(y, x=None, dx=1.0, axis=-1):
    """
    Integrate along the given axis using the composite trapezoidal rule.

    Integrate `y` (`x`) along given axis.

    Parameters
    ----------
    y : array_like
        Input tensor to integrate.
    x : array_like, optional
        The sample points corresponding to the `y` values. If `x` is None,
        the sample points are assumed to be evenly spaced `dx` apart. The
        default is None.
    dx : scalar, optional
        The spacing between sample points when `x` is None. The default is 1.
    axis : int, optional
        The axis along which to integrate.

    Returns
    -------
    trapz : float
        Definite integral as approximated by trapezoidal rule.

    See Also
    --------
    sum, cumsum

    Notes
    -----
    Image [2]_ illustrates trapezoidal rule -- y-axis locations of points
    will be taken from `y` tensor, by default x-axis distances between
    points will be 1.0, alternatively they can be provided with `x` tensor
    or with `dx` scalar.  Return value will be equal to combined area under
    the red lines.


    References
    ----------
    .. [1] Wikipedia page: https://en.wikipedia.org/wiki/Trapezoidal_rule

    .. [2] Illustration image:
           https://en.wikipedia.org/wiki/File:Composite_trapezoidal_rule_illustration.png

    Examples
    --------
    >>> import mars.tensor as mt
    >>> mt.trapz([1,2,3]).execute()
    4.0
    >>> mt.trapz([1,2,3], x=[4,6,8]).execute()
    8.0
    >>> mt.trapz([1,2,3], dx=2).execute()
    8.0
    >>> a = mt.arange(6).reshape(2, 3)
    >>> a.execute()
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> mt.trapz(a, axis=0).execute()
    array([1.5, 2.5, 3.5])
    >>> mt.trapz(a, axis=1).execute()
    array([2.,  8.])

    """
    y = astensor(y)
    axis = validate_axis(y.ndim, axis)
    op = TensorTrapz(y=y, x=x, dx=dx, axis=axis)
    return op(y, x=x)
