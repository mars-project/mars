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

from .eye import eye


def identity(n, dtype=None, sparse=False, gpu=False, chunk_size=None):
    """
    Return the identity tensor.

    The identity tensor is a square array with ones on
    the main diagonal.

    Parameters
    ----------
    n : int
        Number of rows (and columns) in `n` x `n` output.
    dtype : data-type, optional
        Data-type of the output.  Defaults to ``float``.
    sparse: bool, optional
        Create sparse tensor if True, False as default
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    chunks : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension

    Returns
    -------
    out : Tensor
        `n` x `n` array with its main diagonal set to one,
        and all other elements 0.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.identity(3).execute()
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])

    """
    return eye(n, dtype=dtype, sparse=sparse, gpu=gpu, chunk_size=chunk_size)
