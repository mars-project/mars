#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

from ...config import options


def arithmetic_operand(cls=None, init=True, sparse_mode=None):
    def _decorator(cls):
        def __init__(self, casting='same_kind', err=None, dtype=None, sparse=False, **kw):
            err = err if err is not None else np.geterr()
            super(cls, self).__init__(_casting=casting, _err=err, _dtype=dtype, _sparse=sparse, **kw)

        def _is_sparse_binary_and_const(x1, x2):
            if hasattr(x1, 'issparse') and x1.issparse():
                if np.isscalar(x2) or hasattr(x2, 'issparse') and x2.issparse():
                    return True
                else:
                    return False
            if hasattr(x2, 'issparse') and x2.issparse():
                if np.isscalar(x1) or hasattr(x1, 'issparse') and x1.issparse():
                    return True
                else:
                    return False
            return False

        def _is_sparse_binary_or_const(x1, x2):
            if (hasattr(x1, 'issparse') and x1.issparse()) or \
                    (hasattr(x2, 'issparse') and x2.issparse()):
                return True
            return False

        _is_sparse_dict = dict(
            always_false=lambda *_: False,
            unary=lambda x: x.issparse(),
            binary_and=_is_sparse_binary_and_const,
            binary_or=_is_sparse_binary_or_const,
        )
        for v in _is_sparse_dict.values():
            v.__name__ = '_is_sparse'

        if init:
            cls.__init__ = __init__

        if sparse_mode in _is_sparse_dict:
            cls._is_sparse = staticmethod(_is_sparse_dict[sparse_mode])
        elif sparse_mode is not None:  # pragma: no cover
            raise ValueError('Unsupported sparse mode: %s' % sparse_mode)

        return cls

    if cls is not None:
        return _decorator(cls)
    else:
        return _decorator


def tree_add(dtype, chunks, idx, shape, sparse=False):
    """
    Generate tree add plan.

    Assume combine size as 4, given a input chunks with size 8,
    we will generate tree add plan like:

    op op op op    op op op op
     |        |     |        |
      --------       --------
      tree_add        tree_add
          |             |
           -------------
              tree_add

    :param dtype: data type for tree added chunk
    :param chunks: input chunks
    :param idx: index of result chunk
    :param shape: shape of result chunk
    :param sparse: return value is sparse or dense
    :return: result chunk
    """
    from .add import TensorTreeAdd

    combine_size = options.combine_size

    while len(chunks) > combine_size:
        new_chunks = []
        for i in range(0, len(chunks), combine_size):
            chks = chunks[i: i + combine_size]
            if len(chks) == 1:
                chk = chks[0]
            else:
                chk_op = TensorTreeAdd(dtype=dtype, sparse=sparse)
                chk = chk_op.new_chunk(chks, shape=shape)
            new_chunks.append(chk)
        chunks = new_chunks

    op = TensorTreeAdd(dtype=dtype, sparse=sparse)
    return op.new_chunk(chunks, shape=shape, index=idx, order=chunks[0].order)


def tree_op_estimate_size(ctx, op):
    chunk = op.outputs[0]
    sum_inputs = sum(ctx[inp.key][0] for inp in op.inputs)
    if not chunk.is_sparse():
        calc_size = chunk_size = chunk.nbytes
        if np.isnan(calc_size):
            chunk_size = calc_size = sum_inputs
    else:
        calc_size = sum_inputs
        chunk_size = min(sum_inputs, chunk.nbytes + np.dtype(np.int64).itemsize * np.prod(chunk.shape) * chunk.ndim)
        if np.isnan(chunk_size):
            chunk_size = sum_inputs
    ctx[chunk.key] = (chunk_size, calc_size)
