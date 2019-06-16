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

from numbers import Integral

import numpy as np

from .... import operands
from ...core import TENSOR_TYPE
from ..datasource import arange, array
from .core import TensorRandomOperandMixin


class TensorChoice(operands.Choice, TensorRandomOperandMixin):
    __slots__ = '_a', '_size', '_replace', '_p'

    _into_one_chunk_fields_ = ['_a', '_p']
    _input_fields_ = ['_a', '_p']

    def __init__(self, state=None, size=None, replace=None,
                 dtype=None, gpu=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super(TensorChoice, self).__init__(_state=state, _size=size,
                                           _replace=replace, _dtype=dtype, _gpu=gpu, **kw)

    def __call__(self, a, p, chunk_size=None):
        return self.new_tensor([a, p], None, raw_chunk_size=chunk_size)


def choice(random_state, a, size=None, replace=True, p=None, chunk_size=None, gpu=None):
    """
    Generates a random sample from a given 1-D array

    Parameters
    -----------
    a : 1-D array-like or int
        If a tensor, a random sample is generated from its elements.
        If an int, the random sample is generated as if a were mt.arange(a)
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    replace : boolean, optional
        Whether the sample is with or without replacement
    p : 1-D array-like, optional
        The probabilities associated with each entry in a.
        If not given the sample assumes a uniform distribution over all
        entries in a.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default

    Returns
    --------
    samples : single item or tensor
        The generated random samples

    Raises
    -------
    ValueError
        If a is an int and less than zero, if a or p are not 1-dimensional,
        if a is an array-like of size 0, if p is not a vector of
        probabilities, if a and p have different lengths, or if
        replace=False and the sample size is greater than the population
        size

    See Also
    ---------
    randint, shuffle, permutation

    Examples
    ---------
    Generate a uniform random sample from mt.arange(5) of size 3:

    >>> import mars.tensor as mt

    >>> mt.random.choice(5, 3).execute()
    array([0, 3, 4])
    >>> #This is equivalent to mt.random.randint(0,5,3)

    Generate a non-uniform random sample from np.arange(5) of size 3:

    >>> mt.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0]).execute()
    array([3, 3, 0])

    Generate a uniform random sample from mt.arange(5) of size 3 without
    replacement:

    >>> mt.random.choice(5, 3, replace=False).execute()
    array([3,1,0])
    >>> #This is equivalent to np.random.permutation(np.arange(5))[:3]

    Generate a non-uniform random sample from mt.arange(5) of size
    3 without replacement:

    >>> mt.random.choice(5, 3, replace=False, p=[0.1, 0, 0.3, 0.6, 0]).execute()
    array([2, 3, 0])

    Any of the above can be repeated with an arbitrary array-like
    instead of just integers. For instance:

    >>> aa_milne_arr = ['pooh', 'rabbit', 'piglet', 'Christopher']
    >>> np.random.choice(aa_milne_arr, 5, p=[0.5, 0.1, 0.1, 0.3])
    array(['pooh', 'pooh', 'pooh', 'Christopher', 'piglet'],
          dtype='|S11')
    """

    if isinstance(a, Integral):
        if a <= 0:
            raise ValueError('a must be greater than 0')
        a = arange(a, chunk_size=chunk_size)
        dtype = np.random.choice(1, size=(), p=np.array([1]) if p is not None else p).dtype
    else:
        if not isinstance(a, TENSOR_TYPE):
            a = np.asarray(a)
        a = array(a, chunk_size=a.size).rechunk(a.size)  # do rechunk if a is already a tensor
        if a.ndim != 1:
            raise ValueError('a must be one dimensional')
        dtype = a.dtype

    if p is not None:
        if not isinstance(p, TENSOR_TYPE):
            p = np.asarray(p)
            if not np.isclose(p.sum(), 1, rtol=1e-7, atol=0):
                raise ValueError('probabilities do not sum to 1')
            p = array(p, chunk_size=p.size)
        else:
            p = p.rechunk(p.size)
        if p.ndim != 1:
            raise ValueError('p must be one dimensional')

    if size is None:
        length = 1
    else:
        try:
            tuple(size)
            length = np.prod(size)
        except TypeError:
            length = size
    if replace is False and length > a.size:
        raise ValueError("Cannot take a larger sample than population when 'replace=False'")

    size = random_state._handle_size(size)
    op = TensorChoice(state=random_state._state, replace=replace,
                      size=size, dtype=dtype, gpu=gpu)
    return op(a, p, chunk_size=chunk_size)
