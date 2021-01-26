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

import itertools
import os
import sys
import unittest

import numpy as np
import pandas as pd

try:
    import sklearn
except ImportError:
    sklearn = None

import mars.dataframe as md
import mars.tensor as mt
from mars.learn.model_selection import train_test_split
from mars.session import new_session
from mars.tests.integrated.base import IntegrationTestBase


@unittest.skipIf(sklearn is None, 'sklearn not installed')
@unittest.skipIf(sys.platform == 'win32', "plasma don't support windows")
class Test(IntegrationTestBase):
    def testDistributedSplit(self):
        service_ep = 'http://127.0.0.1:' + self.web_port
        timeout = 120 if 'CI' in os.environ else -1
        with new_session(service_ep) as sess:
            run_kwargs = {'timeout': timeout}

            rs = np.random.RandomState(0)
            df_raw = pd.DataFrame(rs.rand(10, 4))
            df = md.DataFrame(df_raw, chunk_size=5)
            X, y = df.iloc[:, :-1], df.iloc[:, -1]

            for x_to_tensor, y_to_tensor in itertools.product(range(1), range(1)):
                x = X
                if x_to_tensor:
                    x = mt.tensor(x)
                yy = y
                if y_to_tensor:
                    yy = mt.tensor(yy)

                x_train, x_test, y_train, y_test = train_test_split(
                    x, y, random_state=0, session=sess, run_kwargs=run_kwargs)
                self.assertIsInstance(x_train, type(x))
                self.assertIsInstance(x_test, type(x))
                self.assertIsInstance(y_train, type(yy))
                self.assertIsInstance(y_test, type(yy))
