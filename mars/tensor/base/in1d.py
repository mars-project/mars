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

from typing import Union

import numpy as np

from ...typing import TileableType
from .. import asarray


def in1d(ar1: Union[TileableType, np.ndarray],
         ar2: Union[TileableType, np.ndarray, list],
         assume_unique: bool = False,
         invert: bool = False):
    """
    Test whether each element of a 1-D tensor is also present in a second tensor.

    Returns a boolean tensor the same length as `ar1` that is True
    where an element of `ar1` is in `ar2` and False otherwise.

    We recommend using :func:`isin` instead of `in1d` for new code.

    Parameters
    ----------
    ar1 : (M,) Tensor
        Input tensor.
    ar2 : array_like
        The values against which to test each value of `ar1`.
    assume_unique : bool, optional
        If True, the input tensors are both assumed to be unique, which
        can speed up the calculation.  Default is False.
    invert : bool, optional
        If True, the values in the returned tensor are inverted (that is,
        False where an element of `ar1` is in `ar2` and True otherwise).
        Default is False. ``np.in1d(a, b, invert=True)`` is equivalent
        to (but is faster than) ``np.invert(in1d(a, b))``.

    Returns
    -------
    in1d : (M,) Tensor, bool
        The values `ar1[in1d]` are in `ar2`.

    See Also
    --------
    isin                  : Version of this function that preserves the
                            shape of ar1.
    numpy.lib.arraysetops : Module with a number of other functions for
                            performing set operations on arrays.

    Notes
    -----
    `in1d` can be considered as an element-wise function version of the
    python keyword `in`, for 1-D sequences. ``in1d(a, b)`` is roughly
    equivalent to ``mt.array([item in b for item in a])``.
    However, this idea fails if `ar2` is a set, or similar (non-sequence)
    container:  As ``ar2`` is converted to a tensor, in those cases
    ``asarray(ar2)`` is an object tensor rather than the expected tensor of
    contained values.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> test = mt.array([0, 1, 2, 5, 0])
    >>> states = [0, 2]
    >>> mask = mt.in1d(test, states)
    >>> mask.execute()
    array([ True, False,  True, False,  True])
    >>> test[mask].execute()
    array([0, 2, 0])
    >>> mask = mt.in1d(test, states, invert=True)
    >>> mask.execute()
    array([False,  True, False,  True, False])
    >>> test[mask].execute()
    array([1, 5])
    """
    from .isin import isin

    ar1 = asarray(ar1).ravel()
    ar2 = asarray(ar2).ravel()
    return isin(ar1, ar2, assume_unique=assume_unique, invert=invert)
