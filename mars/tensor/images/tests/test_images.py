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

import os
import tempfile

import numpy as np
import pytest

try:
    from PIL import Image
except ImportError:
    Image = None

from ....core import tile
from ...images import imread


@pytest.mark.skipif(not Image, reason="Pillow not installed")
def test_imread():
    with tempfile.TemporaryDirectory() as tempdir:
        raws = []
        for i in range(10):
            array = np.random.randint(0, 256, 2500 * 3, dtype=np.uint8).reshape(
                (50, 50, 3)
            )
            raws.append(array)
            im = Image.fromarray(array)
            im.save(os.path.join(tempdir, f"random_{i}.png"))

        t = imread(os.path.join(tempdir, "random_0.png"))
        assert t.shape == (50, 50, 3)
        assert t.dtype == np.dtype("uint8")

        tiled = tile(t)
        assert len(tiled.chunks) == 1
        assert tiled.chunks[0].shape == (50, 50, 3)
        assert tiled.chunks[0].dtype == np.dtype("uint8")

        t = imread(os.path.join(tempdir, "random_*.png"), chunk_size=3)
        assert t.shape == (10, 50, 50, 3)

        tiled = tile(t)
        assert len(tiled.chunks) == 4
        assert tiled.nsplits == ((3, 3, 3, 1), (50,), (50,), (3,))
        assert tiled.chunks[0].dtype == np.dtype("uint8")
        assert tiled.chunks[0].index == (0, 0, 0, 0)
        assert tiled.chunks[0].shape == (3, 50, 50, 3)
        assert tiled.chunks[1].index == (1, 0, 0, 0)
        assert tiled.chunks[1].shape == (3, 50, 50, 3)
        assert tiled.chunks[2].index == (2, 0, 0, 0)
        assert tiled.chunks[2].shape == (3, 50, 50, 3)
        assert tiled.chunks[3].index == (3, 0, 0, 0)
        assert tiled.chunks[3].shape == (1, 50, 50, 3)
