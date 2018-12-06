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


from ....config import options


def tree_add(dtype, chunks, idx, shape):
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
    :return: result chunk
    """
    from .add import TensorTreeAdd

    combine_size = options.tensor.combine_size

    while len(chunks) > combine_size:
        new_chunks = []
        for i in range(0, len(chunks), combine_size):
            chks = chunks[i: i + combine_size]
            if len(chks) == 1:
                chk = chks[0]
            else:
                chk_op = TensorTreeAdd(dtype=dtype)
                chk = chk_op.new_chunk(chks, shape)
            new_chunks.append(chk)
        chunks = new_chunks

    op = TensorTreeAdd(dtype=dtype)
    return op.new_chunk(chunks, shape, index=idx)
