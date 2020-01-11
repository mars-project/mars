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

import mars.tensor as mt
import mars.dataframe as md
from mars.session import new_session
from mars.tiles import get_tiled
from mars.learn.operands import OutputType
from mars.learn.contrib.xgboost import train, MarsDMatrix
from mars.learn.contrib.xgboost.dmatrix import ToDMatrix
from mars.learn.contrib.xgboost.train import XGBTrain
from mars.context import ContextBase, ChunkMeta, RunningMode, LocalContext
from mars.graph import DAG

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
        self.y_series = md.Series(self.y)
        self.weight = rs.rand(n_rows, chunk_size=chunk_size)

    def testSerializeLocalTrain(self):
        sess = new_session()

        with LocalContext(sess._sess):
            dmatrix = ToDMatrix(data=self.X, label=self.y)()
            model = XGBTrain(dtrain=dmatrix)()

            graph = model.build_graph(tiled=True)
            DAG.from_json(graph.to_json())

            dmatrix = ToDMatrix(data=self.X_df, label=self.y_series,
                                output_types=[OutputType.dataframe])()
            model = XGBTrain(dtrain=dmatrix)()

            graph = model.build_graph(tiled=True)
            DAG.from_json(graph.to_json())

            new_X = mt.random.rand(1000, 10, chunk_size=(1000, 5))
            new_X, new_y = ToDMatrix(data=new_X, label=self.y, multi_output=True)()
            dmatrix = ToDMatrix(data=new_X, label=new_y)()
            dmatrix = dmatrix.tiles()

            self.assertEqual(len(dmatrix.chunks), 1)

    def testLocalTrainTensor(self):
        new_session().as_default()
        dtrain = MarsDMatrix(self.X, self.y)
        booster = train({}, dtrain, num_boost_round=2)
        self.assertIsInstance(booster, Booster)

    def testLocalTrainDataFrame(self):
        new_session().as_default()
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
