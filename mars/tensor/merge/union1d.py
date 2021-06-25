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


def union1d(ar1, ar2, aggregate_size=None):
    """
    Find the union of two tensors.

    Return the unique, sorted tensor of values that are in either of the two
    input tensors.

    Parameters
    ----------
    ar1, ar2 : array_like
        Input tensors. They are flattened if they are not already 1D.

    Returns
    -------
    union1d : Tensor
        Unique, sorted union of the input tensors.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.union1d([-1, 0, 1], [-2, 0, 2]).execute()
    array([-2, -1,  0,  1,  2])

    To find the union of more than two arrays, use functools.reduce:

    >>> from functools import reduce
    >>> reduce(mt.union1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2])).execute()
    array([1, 2, 3, 4, 6])
    """

    from ..base import unique, sort
    from .concatenate import concatenate

    result = unique(concatenate((ar1, ar2), axis=None),
                    aggregate_size=aggregate_size)
    if aggregate_size == 1:
        return result
    # make sure the result is sorted
    # TODO(xuye.qin): remove when `mt.unique` supports sort shuffle
    return sort(result)
