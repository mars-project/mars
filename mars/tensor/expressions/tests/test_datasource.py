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

import unittest
import shutil
import tempfile
import mars.dataframe as md
from mars.tensor.expressions.datasource.from_dataframe import from_dataframe
import numpy as np

try:
    import tiledb
except (ImportError, OSError):  # pragma: no cover
    tiledb = None

from mars.tests.core import TestBase
from mars.tensor.expressions.datasource import array, fromtiledb, TensorTileDBDataSource


class Test(TestBase):
    def testFromArray(self):
        x = array([1, 2, 3])
        self.assertEqual(x.shape, (3,))

        y = array([x, x])
        self.assertEqual(y.shape, (2, 3))

        z = array((x, x, x))
        self.assertEqual(z.shape, (3, 3))

    @unittest.skipIf(tiledb is None, 'TileDB not installed')
    def testFromTileDB(self):
        ctx = tiledb.Ctx()

        for sparse in (True, False):
            dom = tiledb.Domain(
                tiledb.Dim(ctx=ctx, name="i", domain=(1, 30), tile=7, dtype=np.int32),
                tiledb.Dim(ctx=ctx, name="j", domain=(1, 20), tile=3, dtype=np.int32),
                tiledb.Dim(ctx=ctx, name="k", domain=(1, 10), tile=4, dtype=np.int32),
                ctx=ctx,
            )
            schema = tiledb.ArraySchema(ctx=ctx, domain=dom, sparse=sparse,
                                        attrs=[tiledb.Attr(ctx=ctx, name='a', dtype=np.float32)])

            tempdir = tempfile.mkdtemp()
            try:
                # create tiledb array
                array_type = tiledb.DenseArray if not sparse else tiledb.SparseArray
                array_type.create(tempdir, schema)

                tensor = fromtiledb(tempdir)
                self.assertIsInstance(tensor.op, TensorTileDBDataSource)
                self.assertEqual(tensor.op.issparse(), sparse)
                self.assertEqual(tensor.shape, (30, 20, 10))
                self.assertEqual(tensor.extra_params.raw_chunk_size, (7, 3, 4))
                self.assertIsNone(tensor.op.tiledb_config)
                self.assertEqual(tensor.op.tiledb_uri, tempdir)
                self.assertIsNone(tensor.op.tiledb_key)
                self.assertIsNone(tensor.op.tiledb_timestamp)

                tensor.tiles()

                self.assertEqual(len(tensor.chunks), 105)
                self.assertIsInstance(tensor.chunks[0].op, TensorTileDBDataSource)
                self.assertEqual(tensor.chunks[0].op.issparse(), sparse)
                self.assertEqual(tensor.chunks[0].shape, (7, 3, 4))
                self.assertIsNone(tensor.chunks[0].op.tiledb_config)
                self.assertEqual(tensor.chunks[0].op.tiledb_uri, tempdir)
                self.assertIsNone(tensor.chunks[0].op.tiledb_key)
                self.assertIsNone(tensor.chunks[0].op.tiledb_timestamp)
                self.assertEqual(tensor.chunks[0].op.tiledb_dim_starts, (1, 1, 1))

                # test axis_offsets of chunk op
                self.assertEqual(tensor.chunks[0].op.axis_offsets, (0, 0, 0))
                self.assertEqual(tensor.chunks[1].op.axis_offsets, (0, 0, 4))
                self.assertEqual(tensor.cix[0, 2, 2].op.axis_offsets, (0, 6, 8))
                self.assertEqual(tensor.cix[0, 6, 2].op.axis_offsets, (0, 18, 8))
                self.assertEqual(tensor.cix[4, 6, 2].op.axis_offsets, (28, 18, 8))

                tensor2 = fromtiledb(tempdir, ctx=ctx)
                self.assertEqual(tensor2.op.tiledb_config, ctx.config().dict())

                tensor2.tiles()

                self.assertEqual(tensor2.chunks[0].op.tiledb_config, ctx.config().dict())
            finally:
                shutil.rmtree(tempdir)

    @unittest.skipIf(tiledb is None, 'TileDB not installed')
    def testDimStartFloat(self):
        ctx = tiledb.Ctx()

        dom = tiledb.Domain(
            tiledb.Dim(ctx=ctx, name="i", domain=(0.0, 6.0), tile=6, dtype=np.float64),
            ctx=ctx,
        )
        schema = tiledb.ArraySchema(ctx=ctx, domain=dom, sparse=True,
                                    attrs=[tiledb.Attr(ctx=ctx, name='a', dtype=np.float32)])

        tempdir = tempfile.mkdtemp()
        try:
            # create tiledb array
            tiledb.SparseArray.create(tempdir, schema)

            with self.assertRaises(ValueError):
                fromtiledb(tempdir, ctx=ctx)
        finally:
            shutil.rmtree(tempdir)

    def testFromDataFrame(self):
        mdf = md.DataFrame({'a': [0, 1, 2], 'b': [3, 4, 5]}, index=['c', 'd', 'e'], chunk_size=2)
        tensor = from_dataframe(mdf)
        self.assertEqual(tensor.shape, (3, 2))
        self.assertEqual(mdf.dtypes[0], tensor.dtype)
