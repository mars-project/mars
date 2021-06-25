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

from ...config import options


def arithmetic_operand(cls=None, init=True, sparse_mode=None):
    def _decorator(cls):
        def __init__(self, casting='same_kind', err=None, **kw):
            err = err if err is not None else np.geterr()
            super(cls, self).__init__(_casting=casting, _err=err, **kw)

        def _is_sparse_binary_and_const(x1, x2):
            if all(np.isscalar(x) for x in [x1, x2]):
                return False
            if all(np.isscalar(x) or (hasattr(x, 'issparse') and x.issparse())
                   for x in [x1, x2]):
                return True
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
            raise ValueError(f'Unsupported sparse mode: {sparse_mode}')

        return cls

    if cls is not None:
        return _decorator(cls)
    else:
        return _decorator


class TreeReductionBuilder:
    def __init__(self, combine_size=None):
        self._combine_size = combine_size or options.combine_size

    def _build_reduction(self, inputs, final=False):
        raise NotImplementedError

    def build(self, inputs):
        combine_size = self._combine_size
        while len(inputs) > self._combine_size:
            new_inputs = []
            for i in range(0, len(inputs), combine_size):
                objs = inputs[i: i + combine_size]
                if len(objs) == 1:
                    obj = objs[0]
                else:
                    obj = self._build_reduction(objs, final=False)
                new_inputs.append(obj)
            inputs = new_inputs

        if len(inputs) == 1:
            return inputs[0]
        return self._build_reduction(inputs, final=True)


def chunk_tree_add(dtype, chunks, idx, shape, sparse=False, combine_size=None):
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
    :param combine_size: combine size
    :return: result chunk
    """
    class ChunkAddBuilder(TreeReductionBuilder):
        def _build_reduction(self, inputs, final=False):
            from .add import TensorTreeAdd
            op = TensorTreeAdd(args=inputs, dtype=dtype, sparse=sparse)
            if not final:
                return op.new_chunk(inputs, shape=shape)
            else:
                return op.new_chunk(inputs, shape=shape, index=idx, order=chunks[0].order)

    return ChunkAddBuilder(combine_size).build(chunks)


def tree_op_estimate_size(ctx, op):
    chunk = op.outputs[0]
    if not chunk.is_sparse():
        max_inputs = max(ctx[inp.key][0] for inp in op.inputs)
        calc_size = chunk_size = chunk.nbytes
        if np.isnan(calc_size):
            chunk_size = calc_size = max_inputs
    else:
        sum_inputs = sum(ctx[inp.key][0] for inp in op.inputs)
        calc_size = sum_inputs
        chunk_size = min(sum_inputs, chunk.nbytes + np.dtype(np.int64).itemsize * np.prod(chunk.shape) * chunk.ndim)
        if np.isnan(chunk_size):
            chunk_size = sum_inputs
    ctx[chunk.key] = (chunk_size, calc_size)
