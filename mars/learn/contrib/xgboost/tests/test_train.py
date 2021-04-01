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

import mars.tensor as mt
import mars.dataframe as md
from mars.core import get_tiled
from mars.session import new_session
from mars.tests.core import TestBase
from mars.learn.contrib.xgboost import train, MarsDMatrix
from mars.learn.contrib.xgboost.dmatrix import ToDMatrix
from mars.learn.contrib.xgboost.train import XGBTrain
from mars.context import ContextBase, ChunkMeta, RunningMode

try:
    import xgboost
    from xgboost import Booster
except ImportError:
    xgboost = None


@unittest.skipIf(xgboost is None, 'XGBoost not installed')
class Test(TestBase):
    def setUp(self):
        n_rows = 1000
        n_columns = 10
        chunk_size = 20
        rs = mt.random.RandomState(0)
        self.X = rs.rand(n_rows, n_columns, chunk_size=chunk_size)
        self.y = rs.rand(n_rows, chunk_size=chunk_size)
        self.X_df = md.DataFrame(self.X)
        self.y_series = md.Series(self.y)
        self.weight = rs.rand(n_rows, chunk_size=chunk_size)
        x_sparse = np.random.rand(n_rows, n_columns)
        x_sparse[np.arange(n_rows), np.random.randint(n_columns, size=n_rows)] = np.nan
        self.X_sparse = mt.tensor(x_sparse, chunk_size=chunk_size).tosparse(missing=np.nan)

        self.session = new_session().as_default()
        self._old_executor = self.session._sess._executor
        self.ctx, self.executor = self._create_test_context(self._old_executor)

    def tearDown(self) -> None:
        self.session._sess._executor = self._old_executor

    def testDmatrix(self):
        x = self.X
        cond = x[:, 0] < 1
        x = x[cond]

        y = self.y[cond]
        # reset shape to mock the case that tileable's shape is doomed,
        # but chunks do not
        y._shape = self.y.shape
        dmatrix = MarsDMatrix(x, y)
        dmatrix.execute()

    def testLocalTrainTensor(self):
        dtrain = MarsDMatrix(self.X, self.y)
        booster = train({}, dtrain, num_boost_round=2)
        self.assertIsInstance(booster, Booster)

    def testLocalTrainSparseTensor(self):
        dtrain = MarsDMatrix(self.X_sparse, self.y)
        booster = train({}, dtrain, num_boost_round=2)
        self.assertIsInstance(booster, Booster)

    def testLocalTrainDataFrame(self):
        dtrain = MarsDMatrix(self.X_df, self.y_series)
        booster = train({}, dtrain, num_boost_round=2)
        self.assertIsInstance(booster, Booster)

    def testDistributedTile(self):
        X, y, w = self.X, self.y, self.weight

        X = X.tiles()
        y = y.tiles()
        w = w.tiles()

        workers = ['addr1:1', 'addr2:1']
        chunk_to_workers = dict()
        X_chunk_to_workers = {c.key: workers[i % 2] for i, c in enumerate(X.chunks)}
        chunk_to_workers.update(X_chunk_to_workers)
        y_chunk_to_workers = {c.key: workers[i % 2] for i, c in enumerate(y.chunks)}
        chunk_to_workers.update(y_chunk_to_workers)
        w_chunk_to_workers = {c.key: workers[i % 2] for i, c in enumerate(w.chunks)}
        chunk_to_workers.update(w_chunk_to_workers)

        class MockDistributedDictContext(ContextBase):
            @property
            def running_mode(self):
                return RunningMode.distributed

            def get_chunk_metas(self, chunk_keys):
                metas = []
                for ck in chunk_keys:
                    if ck in chunk_to_workers:
                        metas.append(ChunkMeta(chunk_size=None, chunk_shape=None,
                                               workers=[chunk_to_workers[ck]]))
                    else:
                        metas.append(ChunkMeta(chunk_size=None, chunk_shape=None,
                                               workers=None))
                return metas

        dmatrix = ToDMatrix(data=X, label=y, weight=w)()
        model = XGBTrain(dtrain=dmatrix)()

        with MockDistributedDictContext():
            model = model.tiles()
            dmatrix = get_tiled(dmatrix)

            # 2 workers
            self.assertEqual(len(dmatrix.chunks), 2)
            self.assertEqual(len(model.chunks), 2)
