# Copyright 1999-2022 Alibaba Group Holding Ltd.
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


def setdiff1d(ar1, ar2, assume_unique=False):
    """
    Find the set difference of two tensors.

    Return the unique values in `ar1` that are not in `ar2`.

    Parameters
    ----------
    ar1 : array_like
        Input tensor.
    ar2 : array_like
        Input comparison tensor.
    assume_unique : bool
        If True, the input tensors are both assumed to be unique, which
        can speed up the calculation.  Default is False.

    Returns
    -------
    setdiff1d : Tensor
        1D tensor of values in `ar1` that are not in `ar2`. The result
        is sorted when `assume_unique=False`, but otherwise only sorted
        if the input is sorted.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> a = mt.array([1, 2, 3, 2, 4, 1])
    >>> b = mt.array([3, 4, 5, 6])
    >>> mt.setdiff1d(a, b).execute()
    array([1, 2])

    """

    from ..datasource.array import asarray
    from .in1d import in1d
    from .unique import unique

    if assume_unique:
        ar1 = asarray(ar1).ravel()
    else:
        ar1 = unique(ar1)
        ar2 = unique(ar2)
    return ar1[in1d(ar1, ar2, assume_unique=True, invert=True)]
