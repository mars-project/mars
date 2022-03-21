# Copyright 1999-2022 Alibaba Group Holding Ltd.
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

import mars
import mars.tensor as mt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from mars.learn import metrics


class MetricsSuite:
    """
    Benchmark learn metrics.
    """

    params = [20_000, 100_000]

    def setup(self, chunk_size: int):
        X, y = make_classification(100_000, random_state=0)
        self.raw_X, self.raw_y = X, y
        clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
        self.raw_pred_y = clf.predict_proba(X)[:, 1]
        self._session = mars.new_session()
        self.y = mt.tensor(self.raw_y, chunk_size=chunk_size)
        self.pred_y = mt.tensor(self.raw_pred_y, chunk_size=chunk_size)

    def teardown(self, chunk_size: int):
        self._session.stop_server()

    def time_roc_curve_auc(self, chunk_size: int):
        fpr, tpr, _ = metrics.roc_curve(self.y, self.pred_y)
        metrics.auc(fpr, tpr)

    def time_roc_auc_score(self, chunk_size: int):
        metrics.roc_auc_score(self.y, self.pred_y)
