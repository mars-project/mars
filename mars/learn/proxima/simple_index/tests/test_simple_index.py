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

try:
    import pyproxima2
except ImportError:  # pragma: no cover
    pyproxima2 = None

import mars.dataframe as md
import mars.tensor as mt
from mars.learn.proxima.simple_index import build_index, search_index
from mars.session import new_session
from mars.tests.core import ExecutorForTest


@unittest.skipIf(pyproxima2 is None, 'pyproxima2 not installed')
class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.session = new_session().as_default()
        self._old_executor = self.session._sess._executor
        self.executor = self.session._sess._executor = \
            ExecutorForTest('numpy', storage=self.session._sess._context)

    def tearDown(self) -> None:
        self.session._sess._executor = self._old_executor

    def prepareData(self, doc_count, query_count, dimension):
        rs = np.random.RandomState(0)
        doc = pd.DataFrame(rs.rand(doc_count, dimension).astype(np.float32))
        query = rs.rand(query_count, dimension).astype(np.float32)
        return doc, query

    def computeDataViaMars(self, doc, query, dimension, topk, doc_chunk, query_chunk, index_builder, index_searcher):
        doc = md.DataFrame(pd.DataFrame(doc), chunk_size=(doc_chunk, dimension))
        query = mt.tensor(query, chunk_size=(query_chunk, dimension))

        index = build_index(doc, doc.index, index_builder=index_builder, session=self.session)
        paths = index.fetch()
        if not isinstance(paths, list):
            paths = [paths]

        try:
            for path in paths:
                with open(path, 'rb') as f:
                    self.assertGreater(len(f.read()), 0)

            pk2, distance = search_index(query, range(len(query)), index, topk, index_searcher=index_searcher,
                                         session=self.session)
            self.assertEqual(pk2.shape, (len(query), topk))
            self.assertEqual(distance.shape, (len(query), topk))
            return pk2, distance
        finally:
            for path in paths:
                os.remove(path)

    def computeDataViaProxima(self, doc, query, dimension, topk,
                              index_builder, builder_params,
                              index_converter, index_converter_params,
                              index_searcher, searcher_params,
                              index_reformer, index_reformer_params):
        import pyproxima2 as proxima

        # holder
        holder = proxima.IndexHolder(type=proxima.IndexMeta.FT_FP32,
                                     dimension=dimension)
        holder.mount(np.array(doc))  # 批量挂载 默认pk从0开始
        # for pk, record in zip(doc.index, np.array(doc)):
        #     holder.emplace(pk, record)

        # converter
        meta = proxima.IndexMeta(proxima.IndexMeta.FT_FP32, dimension=dimension)
        if index_converter is not None:
            converter = proxima.IndexConverter(name=index_converter, meta=meta, params=index_converter_params)
            converter.train_and_transform(holder)
            holder = converter.result()
            meta = converter.meta()

        # builder && dumper
        builder = proxima.IndexBuilder(name=index_builder,
                                       meta=meta,
                                       params=builder_params)
        builder = builder.train_and_build(holder)
        dumper = proxima.IndexDumper(name="MemoryDumper", path="test.index")
        builder.dump(dumper)
        dumper.close()

        # indexflow for search
        flow = proxima.IndexFlow(container_name='MemoryContainer', container_params={},
                                 searcher_name=index_searcher, searcher_params=searcher_params,
                                 # measure_name='Euclidean', measure_params={},
                                 reformer_name=index_reformer, reformer_params=index_reformer_params
                                 )
        flow.load("test.index")
        keys, scores = proxima.IndexUtility.ann_search(searcher=flow, query=query, topk=topk, threads=1)
        return np.asarray(keys), np.asarray(scores)

    def testBuildAndSearchIndex(self):
        # params
        doc_count, query_count, dimension, topk = 200, 15, 5, 2
        index_builder, index_searcher = "SsgBuilder", "SsgSearcher"
        builder_params = {}
        searcher_params = {}
        index_converter = None
        index_converter_params = {}
        index_reformer = ""
        index_reformer_params = {}

        doc_chunk, query_chunk = 50, 5

        # data
        doc, query = self.prepareData(doc_count=doc_count, query_count=query_count, dimension=dimension)

        # mars_data
        pk_m, distance_m = self.computeDataViaMars(doc, query, dimension=dimension, topk=topk, doc_chunk=doc_chunk,
                                                   query_chunk=query_chunk, index_builder=index_builder,
                                                   index_searcher=index_searcher)

        # proxima_data
        pk_p, distance_p = self.computeDataViaProxima(doc=doc, query=query, dimension=dimension, topk=topk,
                                                      index_builder=index_builder, builder_params=builder_params,
                                                      index_converter=index_converter,
                                                      index_converter_params=index_converter_params,
                                                      index_searcher=index_searcher, searcher_params=searcher_params,
                                                      index_reformer=index_reformer,
                                                      index_reformer_params=index_reformer_params)

        # testing
        np.testing.assert_array_equal(pk_p, pk_m)
        np.testing.assert_array_equal(distance_p, distance_m)
