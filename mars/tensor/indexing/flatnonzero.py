#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from .nonzero import nonzero


def flatnonzero(a):
    """
    Return indices that are non-zero in the flattened version of a.

    This is equivalent to a.ravel().nonzero()[0].

    Parameters
    ----------
    a : Tensor
        Input tensor.

    Returns
    -------
    res : Tensor
        Output tensor, containing the indices of the elements of `a.ravel()`
        that are non-zero.

    See Also
    --------
    nonzero : Return the indices of the non-zero elements of the input tensor.
    ravel : Return a 1-D tensor containing the elements of the input tensor.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> x = mt.arange(-2, 3)
    >>> x.execute()
    array([-2, -1,  0,  1,  2])
    >>> mt.flatnonzero(x).execute()
    array([0, 1, 3, 4])

    Use the indices of the non-zero elements as an index array to extract
    these elements:

    >>> x.ravel()[mt.flatnonzero(x)].execute()  # TODO(jisheng): accomplish this after fancy indexing is supported

    """
    from ..base import ravel
    return nonzero(ravel(a))[0]
