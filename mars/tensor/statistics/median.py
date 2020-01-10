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

from .quantile import quantile


def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """
    Compute the median along the specified axis.

    Returns the median of the tensor elements.

    Parameters
    ----------
    a : array_like
        Input tensor or object that can be converted to a tensor.
    axis : {int, sequence of int, None}, optional
        Axis or axes along which the medians are computed. The default
        is to compute the median along a flattened version of the tensor.
        A sequence of axes is supported since version 1.9.0.
    out : Tensor, optional
        Alternative output tensor in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
        Just for compatibility with Numpy, would not take effect.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.

    Returns
    -------
    median : Tensor
        A new tensor holding the result. If the input contains integers
        or floats smaller than ``float64``, then the output data-type is
        ``np.float64``.  Otherwise, the data-type of the output is the
        same as that of the input. If `out` is specified, that tensor is
        returned instead.

    See Also
    --------
    mean, percentile

    Notes
    -----
    Given a vector ``V`` of length ``N``, the median of ``V`` is the
    middle value of a sorted copy of ``V``, ``V_sorted`` - i
    e., ``V_sorted[(N-1)/2]``, when ``N`` is odd, and the average of the
    two middle values of ``V_sorted`` when ``N`` is even.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> a = mt.array([[10, 7, 4], [3, 2, 1]])
    >>> a.execute()
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> mt.median(a).execute()
    3.5
    >>> mt.median(a, axis=0).execute()
    array([6.5, 4.5, 2.5])
    >>> mt.median(a, axis=1).execute()
    array([7.,  2.])
    >>> m = mt.median(a, axis=0)
    >>> out = mt.zeros_like(m)
    >>> mt.median(a, axis=0, out=m).execute()
    array([6.5,  4.5,  2.5])
    >>> m.execute()
    array([6.5,  4.5,  2.5])
    """
    return quantile(a, 0.5, axis=axis, out=out,
                    overwrite_input=overwrite_input, keepdims=keepdims)
