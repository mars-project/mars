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

from ...images import imread


@pytest.mark.skipif(not Image, reason="Pillow not installed")
def test_imread_execution(setup):
    with tempfile.TemporaryDirectory() as tempdir:
        raws = []
        for i in range(10):
            array = np.random.randint(0, 256, 2500, dtype=np.uint8).reshape((50, 50))
            raws.append(array)
            im = Image.fromarray(array)
            im.save(os.path.join(tempdir, f"random_{i}.png"))
        # Single image
        t = imread(os.path.join(tempdir, "random_0.png"))
        res = t.execute().fetch()
        np.testing.assert_array_equal(res, raws[0])

        t2 = imread(os.path.join(tempdir, "random_*.png"))
        res = t2.execute().fetch()
        np.testing.assert_array_equal(np.sort(res, axis=0), np.sort(raws, axis=0))

        t3 = imread(os.path.join(tempdir, "random_*.png"), chunk_size=4)
        res = t3.execute().fetch()
        np.testing.assert_array_equal(np.sort(res, axis=0), np.sort(raws, axis=0))

        t4 = imread(os.path.join(tempdir, "random_*.png"), chunk_size=4)
        res = t4.execute().fetch()
        np.testing.assert_array_equal(np.sort(res, axis=0), np.sort(raws, axis=0))
