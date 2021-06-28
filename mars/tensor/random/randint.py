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

import numpy as np

from ... import opcodes as OperandDef
from ...serialization.serializables import Int64Field, Float64Field
from ..array_utils import array_module
from ..utils import gen_random_seeds
from .core import TensorRandomOperandMixin, TensorSimpleRandomData


class TensorRandint(TensorSimpleRandomData, TensorRandomOperandMixin):
    _op_type_ = OperandDef.RAND_RANDINT

    _fields_ = '_low', '_high', '_density', '_size'
    _low = Int64Field('low')
    _high = Int64Field('high')
    _density = Float64Field('density')
    _func_name = 'randint'

    def __init__(self, size=None, dtype=None,
                 low=None, high=None, density=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super().__init__(_size=size, _low=low, _high=high,
                         _density=density, dtype=dtype, **kw)

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    @property
    def density(self):
        return self._density

    def __call__(self, chunk_size=None):
        return self.new_tensor(None, None, raw_chunk_size=chunk_size)

    @classmethod
    def execute(cls, ctx, op):
        if op.sparse:
            cls.execute_sparse(ctx, op)
        else:
            super().execute(ctx, op)

    @classmethod
    def execute_sparse(cls, ctx, op):
        from ...lib.sparse import SparseNDArray
        from ...lib.sparse.core import cps, sps

        xp = array_module(op.gpu)
        if op.seed:
            rs = np.random.RandomState(op.seed)
        else:
            rs = None

        chunk = op.outputs[0]
        if chunk.ndim > 2:
            raise NotImplementedError

        low = 1 if op.low == 0 else op.low

        rs = rs or xp.random
        size = int(np.ceil(np.prod(chunk.shape) * op.density))
        xps = cps if op.gpu else sps
        ij = xp.empty((2, size))
        ij[0] = rs.randint(chunk.shape[0], size=size)
        ij[1] = rs.randint(chunk.shape[1], size=size)
        data = rs.randint(low, op.high, size=size).astype(op.dtype)
        m = xps.coo_matrix((data, ij), chunk.shape).tocsr()
        m.data[m.data >= op.high] = op.high - 1

        # scipy.sparse is too slow, we remove the precise version due to the performance
        # m = sps.random(*chunk.shape, density=op.density, format='csr')
        # m.data = (rs or xp.random).randint(low, op.high, size=m.data.size)\
        #     .astype(op.dtype)

        ctx[chunk.key] = SparseNDArray(m)

    @classmethod
    def estimate_size(cls, ctx, op):
        chunk = op.outputs[0]
        if not op.sparse or not getattr(op, '_density', None):
            super().estimate_size(ctx, op)
        else:
            # use density to estimate real memory usage
            nbytes = int(chunk.nbytes * getattr(chunk.op, '_density'))
            ctx[chunk.key] = (nbytes, nbytes)


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
    seed = gen_random_seeds(1, random_state.to_numpy())[0]
    op = TensorRandint(seed=seed, low=low, high=high, size=size, dtype=dtype,
                       gpu=gpu, sparse=sparse, density=density)
    return op(chunk_size=chunk_size)
