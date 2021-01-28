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
import unittest

import numpy as np
import pandas as pd
try:
    import sklearn
    from sklearn.metrics import roc_curve as sklearn_roc_curve, auc as sklearn_auc, \
        accuracy_score as sklearn_accuracy_score
except ImportError:
    sklearn = None

from mars import dataframe as md
from mars.learn.metrics import roc_curve, auc, accuracy_score
from mars.tests.integrated.base import IntegrationTestBase
from mars.session import new_session


@unittest.skipIf(sklearn is None, 'sklearn not installed')
@unittest.skipIf(sys.platform == 'win32', "plasma don't support windows")
class Test(IntegrationTestBase):
    def testRocCurveAuc(self):
        service_ep = 'http://127.0.0.1:' + self.web_port
        timeout = 120 if 'CI' in os.environ else -1
        with new_session(service_ep) as sess:
            run_kwargs = {'timeout': timeout}

            rs = np.random.RandomState(0)
            raw = pd.DataFrame({'a': rs.randint(0, 10, (10,)),
                                'b': rs.rand(10)})

            df = md.DataFrame(raw)
            y = df['a'].to_tensor().astype('int')
            pred = df['b'].to_tensor().astype('float')
            fpr, tpr, thresholds = roc_curve(y, pred, pos_label=2,
                                             session=sess, run_kwargs=run_kwargs)
            m = auc(fpr, tpr, session=sess, run_kwargs=run_kwargs)

            sk_fpr, sk_tpr, sk_threshod = sklearn_roc_curve(raw['a'].to_numpy().astype('int'),
                                                            raw['b'].to_numpy().astype('float'),
                                                            pos_label=2)
            expect_m = sklearn_auc(sk_fpr, sk_tpr)
            self.assertAlmostEqual(m.fetch(session=sess), expect_m)

    def testAccuracyScore(self):
        service_ep = 'http://127.0.0.1:' + self.web_port
        timeout = 120 if 'CI' in os.environ else -1
        with new_session(service_ep) as sess:
            run_kwargs = {'timeout': timeout}

            rs = np.random.RandomState(0)
            raw = pd.DataFrame({'a': rs.randint(0, 10, (10,)),
                                'b': rs.randint(0, 10, (10,))})

            df = md.DataFrame(raw)
            y = df['a'].to_tensor().astype('int')
            pred = df['b'].astype('int')

            score = accuracy_score(y, pred, session=sess, run_kwargs=run_kwargs)
            expect = sklearn_accuracy_score(raw['a'].t_numpy().astype('int'),
                                            raw['b'].to_numpy().astype('int'))
            self.assertAlmostEqual(score.fetch()), expect
