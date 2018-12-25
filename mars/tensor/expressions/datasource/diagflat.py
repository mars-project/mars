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

from ...core import Tensor
from .array import tensor as astensor
from .diag import diag


def diagflat(v, k=0, sparse=None, gpu=None, chunk_size=None):
    """
    Create a two-dimensional tensor with the flattened input as a diagonal.

    Parameters
    ----------
    v : array_like
        Input data, which is flattened and set as the `k`-th
        diagonal of the output.
    k : int, optional
        Diagonal to set; 0, the default, corresponds to the "main" diagonal,
        a positive (negative) `k` giving the number of the diagonal above
        (below) the main.
    sparse: bool, optional
        Create sparse tensor if True, False as default
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension

    Returns
    -------
    out : Tensor
        The 2-D output tensor.

    See Also
    --------
    diag : MATLAB work-alike for 1-D and 2-D tensors.
    diagonal : Return specified diagonals.
    trace : Sum along diagonals.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.diagflat([[1,2], [3,4]]).execute()
    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]])

    >>> mt.diagflat([1,2], 1).execute()
    array([[0, 1, 0],
           [0, 0, 2],
           [0, 0, 0]])

    """
    if not isinstance(v, Tensor):
        v = astensor(v).op.data
    return diag(v.flatten(), k=k, sparse=sparse, gpu=gpu, chunk_size=chunk_size)
