#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import mars.tensor as mt
import mars.dataframe as md
from mars.session import new_session
from mars.tests.core import flaky, require_ray
from mars.utils import lazy_import

ray = lazy_import('ray', globals=globals())


@require_ray
class Test(unittest.TestCase):
    def tearDown(self) -> None:
        ray.shutdown()

    def testRayTask(self):
        with new_session(backend='ray').as_default():
            # test tensor task
            raw = np.random.rand(100, 100)
            t = (mt.tensor(raw, chunk_size=30) + 1).sum().to_numpy()
            self.assertAlmostEqual(t, (raw + 1).sum())

            # test DataFrame task
            raw = pd.DataFrame(np.random.random((20, 4)), columns=list('abcd'))
            df = md.DataFrame(raw, chunk_size=5)
            r = df.describe().to_pandas()
            pd.testing.assert_frame_equal(r, raw.describe())

            # test update shape
            raw = np.random.rand(100)
            t = mt.tensor(raw, chunk_size=30)
            selected = (t[t > 0.5] + 1).execute()
            r = selected.to_numpy()
            expected = raw[raw > 0.5] + 1
            np.testing.assert_array_equal(r, expected)

            with tempfile.TemporaryDirectory() as tempdir:
                file_path = os.path.join(tempdir, 'test.csv')

                df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64),
                                  columns=['a', 'b', 'c'])
                df.to_csv(file_path)

                mdf = md.read_csv(file_path)
                r = mdf.groupby('a').agg({'c': 'sum'}).to_pandas()
                expected = df.groupby('a').agg({'c': 'sum'})
                pd.testing.assert_frame_equal(r, expected)

    def testOperandSerialization(self):
        from mars.ray.core import operand_serializer, operand_deserializer

        df = md.DataFrame(mt.random.rand(10, 3), columns=list('abc'))
        r = df.sort_values(by='a')
        op = r.op

        new_op_wrapper = operand_deserializer(operand_serializer(op))
        new_op = new_op_wrapper.op

        self.assertEqual(op.by, new_op.by)
        self.assertIsInstance(new_op, type(op))
        for c1, c2 in zip(op.inputs, new_op.inputs):
            self.assertEqual(c1.key, c2.key)

        for c1, c2 in zip(op.outputs, new_op.outputs):
            self.assertEqual(c1.key, c2.key)

    @flaky(max_runs=3)
    def testRayClusterMode(self):
        def _test():
            t = mt.random.RandomState(0).rand(100, 4, chunk_size=30)
            df = md.DataFrame(t, columns=list('abcd'))
            r = df.describe().execute()
            self.assertEqual(r.shape, (8, 4))

        try:
            with new_session(backend='ray', _load_code_from_local=True).as_default():
                _test()
        except TypeError:  # ray >= 1.2.0
            import ray

            job_config = ray.job_config.JobConfig(code_search_path=['.'])
            with new_session(backend='ray', job_config=job_config).as_default():
                _test()
