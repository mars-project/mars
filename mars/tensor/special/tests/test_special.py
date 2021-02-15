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
    from scipy.special import (
        gammaln as scipy_gammaln,
        erf as scipy_erf,
        betainc as scipy_betainc,
    )
    from mars.tensor.special.gamma_funcs import (
        gammaln, TensorGammaln,
        betainc, TensorBetaInc,
    )
    from mars.tensor.special.err_fresnel import erf, TensorErf
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
            self.assertIsInstance(c.op, TensorGammaln)
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
            self.assertIsInstance(c.op, TensorErf)
            self.assertEqual(c.index, c.inputs[0].index)
            self.assertEqual(c.shape, c.inputs[0].shape)

    def testBetaInc(self):
        raw1 = np.random.rand(4, 3, 2)
        raw2 = np.random.rand(4, 3, 2)
        raw3 = np.random.rand(4, 3, 2)
        a = tensor(raw1, chunk_size=3)
        b = tensor(raw2, chunk_size=3)
        c = tensor(raw3, chunk_size=3)

        r = betainc(a, b, c)
        expect = scipy_betainc(raw1, raw2, raw3)

        self.assertEqual(r.shape, raw1.shape)
        self.assertEqual(r.dtype, expect.dtype)

        r = r.tiles()
        tiled_a = get_tiled(a)

        self.assertEqual(r.nsplits, tiled_a.nsplits)
        for chunk in r.chunks:
            self.assertIsInstance(chunk.op, TensorBetaInc)
            self.assertEqual(chunk.index, chunk.inputs[0].index)
            self.assertEqual(chunk.shape, chunk.inputs[0].shape)

        betainc(a, b, c, out=a)
        expect = scipy_betainc(raw1, raw2, raw3)

        self.assertEqual(a.shape, raw1.shape)
        self.assertEqual(a.dtype, expect.dtype)

        tiled_a = a.tiles()
        b = get_tiled(b)

        self.assertEqual(tiled_a.nsplits, b.nsplits)
        for c in r.chunks:
            self.assertIsInstance(c.op, TensorBetaInc)
            self.assertEqual(c.index, c.inputs[0].index)
            self.assertEqual(c.shape, c.inputs[0].shape)
