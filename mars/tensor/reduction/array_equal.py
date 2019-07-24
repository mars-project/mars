#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2018 Alibaba Group Holding Ltd.
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


def array_equal(a1, a2):
    """
    True if two tensors have the same shape and elements, False otherwise.

    Parameters
    ----------
    a1, a2 : array_like
        Input arrays.

    Returns
    -------
    b : bool
        Returns True if the tensors are equal.

    See Also
    --------
    allclose: Returns True if two tensors are element-wise equal within a
              tolerance.
    array_equiv: Returns True if input tensors are shape consistent and all
                 elements equal.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.array_equal([1, 2], [1, 2]).execute()
    True
    >>> mt.array_equal(mt.array([1, 2]), mt.array([1, 2])).execute()
    True
    >>> mt.array_equal([1, 2], [1, 2, 3]).execute()
    False
    >>> mt.array_equal([1, 2], [1, 4]).execute()
    False

    """
    from ..datasource import tensor as astensor
    from ..datasource.scalar import scalar
    from .all import all

    try:
        a1, a2 = astensor(a1), astensor(a2)
    except Exception:
        return scalar(False)

    if a1.shape != a2.shape:
        return scalar(False)
    return all(astensor(a1 == a2))
