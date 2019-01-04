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

import numpy as np

from .... import operands
from .core import TensorRandomOperandMixin


class TensorRandint(operands.Randint, TensorRandomOperandMixin):
    __slots__ = '_low', '_high', '_density', '_size'

    def __init__(self, state=None, size=None, dtype=None,
                 low=None, high=None, sparse=False, density=None, gpu=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super(TensorRandint, self).__init__(_state=state, _size=size,
                                            _low=low, _high=high, _dtype=dtype,
                                            _sparse=sparse, _density=density,
                                            _gpu=gpu, **kw)

    def __call__(self, chunk_size=None):
        return self.new_tensor(None, None, raw_chunk_size=chunk_size)


def randint(random_state, low, high=None, size=None, dtype='l', density=None,
            chunk_size=None, gpu=None):
    """
    Return random integers from `low` (inclusive) to `high` (exclusive).

    Return random integers from the "discrete uniform" distribution of
    the specified dtype in the "half-open" interval [`low`, `high`). If
    `high` is None (the default), then results are from [0, `low`).

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is one above the
        *highest* such integer).
    high : int, optional
        If provided, one above the largest (signed) integer to be drawn
        from the distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    dtype : dtype, optional
        Desired dtype of the result. All dtypes are determined by their
        name, i.e., 'int64', 'int', etc, so byteorder is not available
        and a specific precision may have different C types depending
        on the platform. The default value is 'np.int'.
    density: float, optional
        if density specified, a sparse tensor will be created
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    dtype : data-type, optional
      Data-type of the returned tensor.

    Returns
    -------
    out : int or Tensor of ints
        `size`-shaped tensor of random integers from the appropriate
        distribution, or a single such random int if `size` not provided.

    See Also
    --------
    random.random_integers : similar to `randint`, only for the closed
        interval [`low`, `high`], and 1 is the lowest value if `high` is
        omitted. In particular, this other one is the one to use to generate
        uniformly distributed discrete non-integers.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.random.randint(2, size=10).execute()
    array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0])
    >>> mt.random.randint(1, size=10).execute()
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    Generate a 2 x 4 tensor of ints between 0 and 4, inclusive:

    >>> mt.random.randint(5, size=(2, 4)).execute()
    array([[4, 0, 2, 1],
           [3, 2, 2, 0]])
    """
    sparse = bool(density)
    size = random_state._handle_size(size)
    op = TensorRandint(state=random_state._state, low=low, high=high, size=size, dtype=dtype,
                       gpu=gpu, sparse=sparse, density=density)
    return op(chunk_size=chunk_size)
