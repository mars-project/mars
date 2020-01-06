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

from ...core import Base, Entity
from ..arithmetic import truediv
from .quantile import _quantile_unchecked, _quantile_is_valid


q_error_msg = "Percentiles must be in the range [0, 100]"


def percentile(a, q, axis=None, out=None, overwrite_input=False,
               interpolation='linear', keepdims=False):
    """
    Compute the q-th percentile of the data along the specified axis.

    Returns the q-th percentile(s) of the array elements.

    Parameters
    ----------
    a : array_like
        Input tensor or object that can be converted to a tensor.
    q : array_like of float
        Percentile or sequence of percentiles to compute, which must be between
        0 and 100 inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the percentiles are computed. The
        default is to compute the percentile(s) along a flattened
        version of the tensor.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
        Just for compatibility with Numpy, would not take effect.
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        This optional parameter specifies the interpolation method to
        use when the desired percentile lies between two data points
        ``i < j``:

        * 'linear': ``i + (j - i) * fraction``, where ``fraction``
          is the fractional part of the index surrounded by ``i``
          and ``j``.
        * 'lower': ``i``.
        * 'higher': ``j``.
        * 'nearest': ``i`` or ``j``, whichever is nearest.
        * 'midpoint': ``(i + j) / 2``.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the
        result will broadcast correctly against the original array `a`.

    Returns
    -------
    percentile : scalar or ndarray
        If `q` is a single percentile and `axis=None`, then the result
        is a scalar. If multiple percentiles are given, first axis of
        the result corresponds to the percentiles. The other axes are
        the axes that remain after the reduction of `a`. If the input
        contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the
        same as that of the input. If `out` is specified, that array is
        returned instead.

    See Also
    --------
    mean
    median : equivalent to ``percentile(..., 50)``
    nanpercentile
    quantile : equivalent to percentile, except with q in the range [0, 1].

    Notes
    -----
    Given a vector ``V`` of length ``N``, the q-th percentile of
    ``V`` is the value ``q/100`` of the way from the minimum to the
    maximum in a sorted copy of ``V``. The values and distances of
    the two nearest neighbors as well as the `interpolation` parameter
    will determine the percentile if the normalized ranking does not
    match the location of ``q`` exactly. This function is the same as
    the median if ``q=50``, the same as the minimum if ``q=0`` and the
    same as the maximum if ``q=100``.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> a = mt.array([[10, 7, 4], [3, 2, 1]])
    >>> a.execute()
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> mt.percentile(a, 50).execute()
    3.5
    >>> mt.percentile(a, 50, axis=0).execute()
    array([6.5, 4.5, 2.5])
    >>> mt.percentile(a, 50, axis=1).execute()
    array([7.,  2.])
    >>> mt.percentile(a, 50, axis=1, keepdims=True).execute()
    array([[7.],
           [2.]])

    >>> m = mt.percentile(a, 50, axis=0)
    >>> out = mt.zeros_like(m)
    >>> mt.percentile(a, 50, axis=0, out=out).execute()
    array([6.5, 4.5, 2.5])
    >>> m.execute()
    array([6.5, 4.5, 2.5])

    The different types of interpolation can be visualized graphically:

    .. plot::

        import matplotlib.pyplot as plt
        import mars.tensor as mt
        import numpy as np

        a = mt.arange(4)
        p = mt.linspace(0, 100, 6001)
        ax = plt.gca()
        lines = [
            ('linear', None),
            ('higher', '--'),
            ('lower', '--'),
            ('nearest', '-.'),
            ('midpoint', '-.'),
        ]
        for interpolation, style in lines:
            ax.plot(
                np.asarray(p), np.asarray(mt.percentile(a, p, interpolation=interpolation)),
                label=interpolation, linestyle=style)
        ax.set(
            title='Interpolation methods for list: ' + str(a),
            xlabel='Percentile',
            ylabel='List item returned',
            yticks=np.asarray(a))
        ax.legend()
        plt.show()

    """
    if not isinstance(q, (Base, Entity)):
        q = np.asanyarray(q)
        q = np.true_divide(q, 100)
        # do check instantly if q is not a tensor
        if not _quantile_is_valid(q):
            raise ValueError(q_error_msg)
    else:
        q = truediv(q, 100)

    return _quantile_unchecked(a, q, axis=axis, out=out, overwrite_input=overwrite_input,
                               interpolation=interpolation, keepdims=keepdims,
                               q_error_msg=q_error_msg)
