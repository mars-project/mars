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


import unittest

import numpy as np

from mars.tensor import tensor
from mars.tiles import get_tiled

try:
    import scipy
    from scipy.special import gammaln as scipy_gammaln, erf as scipy_erf

    from mars.tensor.special.gammaln import gammaln, TensorGammaLnUnary
    from mars.tensor.special.erf import erf, TensorErfUnary
except ImportError:
    scipy = None


@unittest.skipIf(scipy is None, 'scipy not installed')
class Test(unittest.TestCase):
    def testGammaln(self):
        raw = np.random.rand(10, 8, 5)
        t = tensor(raw, chunk_size=3)

        r = gammaln(t)
        expect = scipy_gammaln(raw)

        self.assertEqual(r.shape, raw.shape)
        self.assertEqual(r.dtype, expect.dtype)

        r = r.tiles()
        t = get_tiled(t)

        self.assertEqual(r.nsplits, t.nsplits)
        for c in r.chunks:
            self.assertIsInstance(c.op, TensorGammaLnUnary)
            self.assertEqual(c.index, c.inputs[0].index)
            self.assertEqual(c.shape, c.inputs[0].shape)

    def testElf(self):
        raw = np.random.rand(10, 8, 5)
        t = tensor(raw, chunk_size=3)

        r = erf(t)
        expect = scipy_erf(raw)

        self.assertEqual(r.shape, raw.shape)
        self.assertEqual(r.dtype, expect.dtype)

        r = r.tiles()
        t = get_tiled(t)

        self.assertEqual(r.nsplits, t.nsplits)
        for c in r.chunks:
            self.assertIsInstance(c.op, TensorErfUnary)
            self.assertEqual(c.index, c.inputs[0].index)
            self.assertEqual(c.shape, c.inputs[0].shape)
