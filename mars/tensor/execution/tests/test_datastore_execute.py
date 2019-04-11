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

import tempfile
import shutil
import unittest

import numpy as np
import scipy.sparse as sps
try:
    import tiledb
except (ImportError, OSError):  # pragma: no cover
    tiledb = None

from mars.executor import Executor
from mars.tests.core import TestBase
from mars.tensor import tensor, arange, totiledb


class Test(TestBase):
    def setUp(self):
        super(Test, self).setUp()
        self.executor = Executor()

    @unittest.skipIf(tiledb is None, 'tiledb not installed')
    def testStoreTileDBExecution(self):
        ctx = tiledb.Ctx()

        tempdir = tempfile.mkdtemp()
        try:
            # store TileDB dense array
            expected = np.random.rand(8, 4, 3)
            a = tensor(expected, chunk_size=(3, 3, 2))
            save = totiledb(tempdir, a, ctx=ctx)
            self.executor.execute_tensor(save)

            with tiledb.DenseArray(uri=tempdir, ctx=ctx) as arr:
                np.testing.assert_allclose(expected, arr.read_direct())
        finally:
            shutil.rmtree(tempdir)

        tempdir = tempfile.mkdtemp()
        try:
            # store tensor with 1 chunk to TileDB dense array
            a = arange(12)
            save = totiledb(tempdir, a, ctx=ctx)
            self.executor.execute_tensor(save)

            with tiledb.DenseArray(uri=tempdir, ctx=ctx) as arr:
                np.testing.assert_allclose(np.arange(12), arr.read_direct())
        finally:
            shutil.rmtree(tempdir)

        tempdir = tempfile.mkdtemp()
        try:
            # store 2-d TileDB sparse array
            expected = sps.random(8, 7, density=0.1)
            a = tensor(expected, chunk_size=(3, 5))
            save = totiledb(tempdir, a, ctx=ctx)
            self.executor.execute_tensor(save)

            with tiledb.SparseArray(uri=tempdir, ctx=ctx) as arr:
                data = arr[:, :]
                coords = data['coords']
                value = data[arr.attr(0).name]
                ij = tuple(coords[arr.domain.dim(k).name] for k in range(arr.ndim))
                result = sps.coo_matrix((value, ij), shape=arr.shape)

                np.testing.assert_allclose(expected.toarray(), result.toarray())
        finally:
            shutil.rmtree(tempdir)
