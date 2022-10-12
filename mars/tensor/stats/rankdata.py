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

from ... import tensor as mt
import numpy as np


def rankdata(a, method="average", *, axis=None):
    """Assign ranks to data, dealing with ties appropriately.
    By default (``axis=None``), the data array is first flattened, and a flat
    array of ranks is returned. Separately reshape the rank array to the
    shape of the data array if desired (see Examples).
    Ranks begin at 1.  The `method` argument controls how ranks are assigned
    to equal values.  See [1]_ for further discussion of ranking methods.
    Parameters
    ----------
    a : array_like
        The array of values to be ranked.
    method : {'average', 'min', 'max', 'dense', 'ordinal'}, optional
        The method used to assign ranks to tied elements.
        The following methods are available (default is 'average'):
          * 'average': The average of the ranks that would have been assigned to
            all the tied values is assigned to each value.
          * 'min': The minimum of the ranks that would have been assigned to all
            the tied values is assigned to each value.  (This is also
            referred to as "competition" ranking.)
          * 'max': The maximum of the ranks that would have been assigned to all
            the tied values is assigned to each value.
          * 'dense': Like 'min', but the rank of the next highest element is
            assigned the rank immediately after those assigned to the tied
            elements.
          * 'ordinal': All values are given a distinct rank, corresponding to
            the order that the values occur in `a`.
    axis : {None, int}, optional
        Axis along which to perform the ranking. If ``None``, the data array
        is first flattened.
    Returns
    -------
    ranks : ndarray
         An array of size equal to the size of `a`, containing rank
         scores.
    References
    ----------
    .. [1] "Ranking", https://en.wikipedia.org/wiki/Ranking
    Examples
    --------
    >>> from mars.tensor.stats import rankdata
    >>> rankdata([0, 2, 3, 2]).execute()
    array([ 1. ,  2.5,  4. ,  2.5])
    >>> rankdata([0, 2, 3, 2], method='min').execute()
    array([ 1,  2,  4,  2])
    >>> rankdata([0, 2, 3, 2], method='max').execute()
    array([ 1,  3,  4,  3])
    >>> rankdata([0, 2, 3, 2], method='dense').execute()
    array([ 1,  2,  3,  2])
    >>> rankdata([0, 2, 3, 2], method='ordinal').execute()
    array([ 1,  2,  4,  3])
    >>> rankdata([[0, 2], [3, 2]]).reshape(2,2).execute()
    array([[1. , 2.5],
          [4. , 2.5]])
    >>> rankdata([[0, 2, 2], [3, 2, 5]], axis=1).execute()
    array([[1. , 2.5, 2.5],
           [2. , 1. , 3. ]])
    """
    if method not in ("average", "min", "max", "dense", "ordinal"):
        raise ValueError('unknown method "{0}"'.format(method))

    if axis is not None:
        a = np.asarray(a)
        if a.size == 0:
            np.core.multiarray.normalize_axis_index(axis, a.ndim)
            dt = np.float64 if method == "average" else np.int_
            return mt.empty(a.shape, dtype=dt)
        return mt.tensor(np.apply_along_axis(rankdata, axis, a, method))

    arr = mt.ravel(mt.asarray(a))
    algo = "mergesort" if method == "ordinal" else "quicksort"
    sorter = mt.argsort(arr, kind=algo)

    inv = mt.empty(sorter.size, dtype=np.intp)
    inv[sorter] = mt.arange(sorter.size, dtype=np.intp)

    if method == "ordinal":
        return inv + 1

    arr = arr[sorter]
    obs = mt.r_[True, arr[1:] != arr[:-1]]
    dense = obs.cumsum()[inv]

    if method == "dense":
        return dense

    count = mt.r_[mt.nonzero(obs)[0], len(obs)]

    if method == "max":
        return count[dense]

    if method == "min":
        return count[dense - 1] + 1

    return 0.5 * (count[dense] + count[dense - 1] + 1)
