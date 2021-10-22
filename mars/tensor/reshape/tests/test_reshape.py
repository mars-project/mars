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
import pytest

from ....core import tile
from ....core.operand import OperandStage
from ...datasource import ones
from ..reshape import TensorReshape


def test_reshape():
    a = ones((10, 20, 30), chunk_size=5)
    b = a.reshape(10, 600)

    b = tile(b)

    assert tuple(sum(s) for s in b.nsplits) == (10, 600)

    a = ones((10, 600), chunk_size=5)
    b = a.reshape(10, 30, 20)

    b = tile(b)

    assert tuple(sum(s) for s in b.nsplits) == (10, 30, 20)

    a = ones((10, 600), chunk_size=5)
    a.shape = [10, 30, 20]

    a = tile(a)

    assert tuple(sum(s) for s in a.nsplits) == (10, 30, 20)

    # test reshape unknown shape
    c = a[a > 0]
    d = c.reshape(10, 600)
    assert d.shape == (10, 600)
    d = c.reshape(-1, 10)
    assert len(d.shape) == 2
    assert np.isnan(d.shape[0])
    assert d.shape[1]

    with pytest.raises(TypeError):
        a.reshape((10, 30, 20), other_argument=True)


def test_shuffle_reshape():
    a = ones((31, 27), chunk_size=10)
    b = a.reshape(27, 31)
    b.op.extra_params["_reshape_with_shuffle"] = True

    b = tile(b)

    assert tuple(sum(s) for s in b.nsplits) == (27, 31)
    assert isinstance(b.chunks[0].op, TensorReshape)
    assert b.chunks[0].op.stage == OperandStage.reduce

    shuffle_map_sample = b.chunks[0].inputs[0].inputs[0]
    assert isinstance(shuffle_map_sample.op, TensorReshape)
    assert shuffle_map_sample.op.stage == OperandStage.map
