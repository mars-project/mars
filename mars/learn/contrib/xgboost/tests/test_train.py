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

import mars.tensor as mt
import mars.dataframe as md
from mars.learn.contrib.xgboost import train, MarsDMatrix
from mars.session import new_session

try:
    import xgboost
    from xgboost import Booster
except ImportError:
    xgboost = None


@unittest.skipIf(xgboost is None, 'XGBoost not installed')
class Test(unittest.TestCase):
    def setUp(self):
        n_rows = 1000
        n_columns = 10
        chunk_size = 20
        rs = mt.random.RandomState(0)
        self.X = rs.rand(n_rows, n_columns, chunk_size=chunk_size)
        self.y = rs.rand(n_rows, chunk_size=chunk_size)
        self.X_df = md.DataFrame(self.X)

    def testLocalTrainTensor(self):
        new_session().as_default()
        dtrain = MarsDMatrix(self.X, self.y)
        booster = train({}, dtrain, num_boost_round=2)
        self.assertIsInstance(booster, Booster)

    def testLocalTrainDataFrame(self):
        new_session().as_default()
        dtrain = MarsDMatrix(self.X_df, self.y)
        booster = train({}, dtrain, num_boost_round=2)
        self.assertIsInstance(booster, Booster)
