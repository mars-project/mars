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
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from mars import dataframe as md
from mars.learn.tests.integrated.base import LearnIntegrationTestBase
from mars.session import new_session


@unittest.skipIf(sys.platform == 'win32', 'does not run in windows')
class Test(LearnIntegrationTestBase):
    @property
    def _extra_worker_options(self):
        # overwrite iloc that ensuring no iloc is executed
        return ['--load-modules', 'mars.optimizes.tileable_graph.tests.integrated.raise_iloc']

    def testDistributedReadCSVHead(self):
        service_ep = 'http://127.0.0.1:' + self.web_port
        timeout = 120 if 'CI' in os.environ else -1

        with new_session(service_ep) as sess:
            rs = np.random.RandomState(0)

            # test md.read_csv().head()
            with tempfile.TemporaryDirectory() as d:
                file_path = os.path.join(d, 'test.csv')

                df = pd.DataFrame({
                    'a': rs.rand(100),
                    'b': [f's{i}' for i in range(100)],
                })
                df.to_csv(file_path, index=False)

                chunk_bytes = os.stat(file_path).st_size // 3 - 2
                mdf = md.read_csv(file_path, chunk_bytes=chunk_bytes)

                r = mdf.head(3)
                result = r.execute(session=sess, timeout=timeout).fetch()
                expected = df.head(3)
                pd.testing.assert_frame_equal(result, expected)
