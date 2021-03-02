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
import unittest

import numpy as np
import pandas as pd

import mars
from mars import dataframe as md
from mars import tensor as mt
from mars.learn.proxima.core import proxima
from mars.learn.proxima.simple_index import build_index, search_index
from mars.learn.proxima.simple_index.tests.test_simple_index import gen_data, proxima_build_and_query
from mars.tests.integrated.base import IntegrationTestBase
from mars.session import new_session


@unittest.skipIf(proxima is None, 'proxima not installed')
class Test(IntegrationTestBase):
    def setUp(self):
        self.old_cwd = os.getcwd()
        os.chdir(os.path.dirname(os.path.dirname(mars.__file__)))
        super().setUp()

    def tearDown(self):
        os.chdir(self.old_cwd)
        super().tearDown()

    def testDistributedProxima(self):
        # params
        doc_count, query_count, dimension = 200, 15, 10
        topk = 10
        doc_chunk, query_chunk = 50, 5

        service_ep = 'http://127.0.0.1:' + self.web_port
        with new_session(service_ep) as sess:
            # data
            doc, query = gen_data(doc_count=doc_count, query_count=query_count, dimension=dimension)

            df = md.DataFrame(pd.DataFrame(doc), chunk_size=(doc_chunk, dimension))
            q = mt.tensor(query, chunk_size=(query_chunk, dimension))

            index = build_index(df, session=sess, column_number=2)

            # proxima_data
            pk_p, distance_p = proxima_build_and_query(doc, query, topk)

            pk_m, distance_m = search_index(q, topk, index, session=sess)

            # testing
            np.testing.assert_array_equal(pk_p, pk_m)
            np.testing.assert_array_equal(distance_p, distance_m)
