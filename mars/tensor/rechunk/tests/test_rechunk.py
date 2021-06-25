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

from mars.core import tile
from mars.tensor.datasource import ones
from mars.tensor.indexing.slice import TensorSlice
from mars.tensor.rechunk.rechunk import compute_rechunk

    
def test_compute_rechunk():
    tensor = ones((12, 8), chunk_size=((4, 4, 3, 1), (3, 3, 2)))
    tensor = tile(tensor)
    new_tensor = compute_rechunk(tensor, ((9, 2, 1), (2, 1, 4, 1)))

    assert len(new_tensor.chunks) == 12
    assert len(new_tensor.chunks[0].inputs) == 3
    assert isinstance(new_tensor.chunks[0].inputs[0].op, TensorSlice)
    assert new_tensor.chunks[0].inputs[0].inputs[0] is tensor.chunks[0].data
    assert new_tensor.chunks[0].inputs[0].op.slices == [slice(None, None, None), slice(None, 2, None)]
    assert new_tensor.chunks[0].inputs[1].inputs[0] is tensor.chunks[3].data
    assert new_tensor.chunks[0].inputs[1].op.slices == [slice(None, None, None), slice(None, 2, None)]
    assert new_tensor.chunks[0].inputs[2].inputs[0] is tensor.chunks[6].data
    assert new_tensor.chunks[0].inputs[2].op.slices == [slice(None, 1, None), slice(None, 2, None)]
    assert new_tensor.chunks[-1].inputs[0] is tensor.chunks[-1].data
    assert new_tensor.chunks[-1].op.slices == [slice(None, None, None), slice(1, None, None)]


def test_rechunk():
    tensor = ones((12, 9), chunk_size=4)
    new_tensor = tensor.rechunk(3)
    t, new_tensor = tile(tensor, new_tensor)

    assert len(new_tensor.chunks) == 12
    assert new_tensor.chunks[0].inputs[0] == t.chunks[0].data
    assert len(new_tensor.chunks[1].inputs) == 2
    assert new_tensor.chunks[1].inputs[0].op.slices == [slice(None, 3, None), slice(3, None, None)]
    assert new_tensor.chunks[1].inputs[1].op.slices == [slice(None, 3, None), slice(None, 2, None)]
    assert len(new_tensor.chunks[-1].inputs) == 2
    assert new_tensor.chunks[-1].inputs[0].op.slices == [slice(1, None, None), slice(2, None, None)]
    assert new_tensor.chunks[-1].inputs[1].op.slices == [slice(1, None, None), slice(None, None, None)]


def test_sparse():
    tensor = ones((7, 12), chunk_size=4).tosparse()
    new_tensor = tensor.rechunk(5)
    new_tensor = tile(new_tensor)

    assert new_tensor.issparse() is True
    assert all(c.issparse() for c in new_tensor.chunks) is True


def test_order():
    tensor = ones((7, 12), chunk_size=4, order='F')
    new_tensor = tensor.rechunk(5)

    assert new_tensor.order.value
