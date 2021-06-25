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

from ..datasource import tensor as astensor
from ..core import TENSOR_TYPE


def shuffle(random_state, x, axis=0):
    r"""
    Modify a sequence in-place by shuffling its contents.
    The order of sub-arrays is changed but their contents remains the same.

    Parameters
    ----------
    x : array_like
        The array or list to be shuffled.
    axis : int, optional
        The axis which `x` is shuffled along. Default is 0.

    Returns
    -------
    None

    Examples
    --------
    >>> import mars.tensor as mt
    >>> rng = mt.random.RandomState()
    >>> arr = mt.arange(10)
    >>> rng.shuffle(arr)
    >>> arr.execute()
    array([0, 1, 4, 2, 8, 6, 5, 9, 3, 7]) # random

    >>> arr = mt.arange(9).reshape((3, 3))
    >>> rng.shuffle(arr)
    >>> arr.execute()
    array([[6, 7, 8], # random
           [0, 1, 2],
           [3, 4, 5]])
    """
    from .permutation import permutation

    if isinstance(x, (list, np.ndarray, TENSOR_TYPE)):
        x = astensor(x)
    else:
        raise TypeError('x should be list, numpy ndarray or tensor')

    ret = permutation(random_state, x, axis=axis)
    x.data = ret.data
