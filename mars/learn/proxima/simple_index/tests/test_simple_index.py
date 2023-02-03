# Copyright 1999-2021 Alibaba Group Holding Ltd.
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

import numpy as np
import pandas as pd
import pytest

from ..... import dataframe as md
from ..... import tensor as mt
from ...core import proxima
from .. import build_index, search_index, recall


def proxima_build_and_query(
    doc,
    query,
    topk,
    measure_name=None,
    dimension=None,
    index_builder=None,
    builder_params=None,
    index_converter=None,
    index_converter_params=None,
    index_searcher=None,
    searcher_params=None,
    index_reformer=None,
    index_reformer_params=None,
):
    if measure_name is None:
        measure_name = "SquaredEuclidean"
    if dimension is None:
        dimension = doc.shape[1]
    if index_builder is None:
        index_builder = "SsgBuilder"
    if builder_params is None:
        builder_params = {}
    if index_converter_params is None:
        index_converter_params = {}
    if index_searcher is None:
        index_searcher = ""
    if searcher_params is None:
        searcher_params = {}
    if index_reformer is None:
        index_reformer = ""
    if index_reformer_params is None:
        index_reformer_params = {}

    map_dtype = {
        np.dtype(np.float32): proxima.IndexMeta.FT_FP32,
        np.dtype(np.int16): proxima.IndexMeta.FT_INT16,
    }
    # holder
    holder = proxima.IndexHolder(type=map_dtype[doc.dtypes[0]], dimension=dimension)
    holder.mount(np.array(doc))  # add batch data, pk starts from 0

    # converter
    meta = proxima.IndexMeta(
        map_dtype[doc.dtypes[0]], dimension=dimension, measure_name=measure_name
    )
    if index_converter is not None:
        if index_converter == "MipsConverter":
            measure_name = ""
        converter = proxima.IndexConverter(
            name=index_converter, meta=meta, params=index_converter_params
        )
        converter.train_and_transform(holder)
        holder = converter.result()
        meta = converter.meta()

    # builder && dumper
    builder = proxima.IndexBuilder(name=index_builder, meta=meta, params=builder_params)
    builder = builder.train_and_build(holder)
    dumper = proxima.IndexDumper(name="MemoryDumper", path="test.index")
    builder.dump(dumper)
    dumper.close()

    # indexflow for search
    flow = proxima.IndexFlow(
        container_name="MemoryContainer",
        container_params={},
        searcher_name=index_searcher,
        searcher_params=searcher_params,
        measure_name=measure_name,
        measure_params={},
        reformer_name=index_reformer,
        reformer_params=index_reformer_params,
    )
    flow.load("test.index")
    keys, scores = proxima.IndexUtility.ann_search(
        searcher=flow, query=query, topk=topk, threads=1
    )
    return np.asarray(keys), np.asarray(scores)


def gen_data(doc_count, query_count, dimension, dtype=np.float32):
    if dtype == np.float32:
        rs = np.random.RandomState(0)
        doc = pd.DataFrame(rs.rand(doc_count, dimension).astype(dtype))
        query = rs.rand(query_count, dimension).astype(dtype)
    elif dtype == np.int32:
        rs = np.random.RandomState(0)
        doc = pd.DataFrame((rs.rand(doc_count, dimension) * 1000).astype(dtype))
        query = (rs.rand(query_count, dimension) * 1000).astype(dtype)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")
    return doc, query


@pytest.mark.skipif(proxima is None, reason="proxima not installed")
def build_and_query(
    doc,
    query,
    topk,
    column_number,
    row_number,
    threads=1,
    dimension=None,
    measure_name=None,
    index_builder=None,
    builder_params=None,
    index_converter=None,
    index_converter_params=None,
    index_searcher=None,
    searcher_params=None,
    index_reformer=None,
    index_reformer_params=None,
):
    if measure_name is None:
        measure_name = "SquaredEuclidean"
    if dimension is None:
        dimension = doc.shape[1]
    if index_builder is None:
        index_builder = "SsgBuilder"
    if builder_params is None:
        builder_params = {}
    if index_converter_params is None:
        index_converter_params = {}
    if index_searcher is None:
        index_searcher = ""
    if searcher_params is None:
        searcher_params = {}
    if index_reformer is None:
        index_reformer = ""
    if index_reformer_params is None:
        index_reformer_params = {}

    doc = md.DataFrame(pd.DataFrame(doc))
    query = mt.tensor(query)

    index = build_index(
        tensor=doc,
        need_shuffle=False,
        column_number=column_number,
        distance_metric=measure_name,
        dimension=dimension,
        index_builder=index_builder,
        index_builder_params=builder_params,
        index_converter=index_converter,
        index_converter_params=index_converter_params,
    )
    paths = index.fetch()
    if not isinstance(paths, list):
        paths = [paths]

    try:
        for path in paths:
            with open(path, "rb") as f:
                assert len(f.read()) > 0

        pk2, distance = search_index(
            tensor=query,
            threads=threads,
            row_number=row_number,
            distance_metric=measure_name,
            dimension=dimension,
            topk=topk,
            index=index,
            index_searcher=index_searcher,
            index_searcher_params=searcher_params,
            index_reformer=index_reformer,
            index_reformer_params=index_reformer_params,
        )
        assert pk2.shape == (len(query), topk)
        assert distance.shape == (len(query), topk)
        return pk2, distance
    finally:
        for path in paths:
            os.remove(path)


def consistency_checking(
    doc,
    query,
    dimension,
    topk,
    measure_name,
    column_number,
    row_number,
    threads,
    index_builder,
    builder_params,
    index_converter,
    index_converter_params,
    index_searcher,
    searcher_params,
    index_reformer,
    index_reformer_params,
    decimal=6,
):
    # proxima_data
    pk_p, distance_p = proxima_build_and_query(
        doc=doc,
        query=query,
        dimension=dimension,
        topk=topk,
        measure_name=measure_name,
        index_builder=index_builder,
        builder_params=builder_params,
        index_converter=index_converter,
        index_converter_params=index_converter_params,
        index_searcher=index_searcher,
        searcher_params=searcher_params,
        index_reformer=index_reformer,
        index_reformer_params=index_reformer_params,
    )

    # mars_data
    pk_m, distance_m = build_and_query(
        doc,
        query,
        dimension=dimension,
        topk=topk,
        threads=threads,
        measure_name=measure_name,
        column_number=column_number,
        row_number=row_number,
        index_builder=index_builder,
        builder_params=builder_params,
        index_converter=index_converter,
        index_converter_params=index_converter_params,
        index_searcher=index_searcher,
        searcher_params=searcher_params,
        index_reformer=index_reformer,
        index_reformer_params=index_reformer_params,
    )

    # testing
    np.testing.assert_array_equal(pk_p, pk_m)
    np.testing.assert_array_almost_equal(distance_p, distance_m, decimal=decimal)


@pytest.mark.skipif(proxima is None, reason="proxima not installed")
def test_build_and_search_index(setup):
    # for now, test SquaredEuclidean and Euclidean only,
    # TODO: add more tests for "Canberra", "Chebyshev"
    #  "Manhattan" when ready

    # L2 space
    # params
    doc_count, query_count, dimension, topk = 200, 15, 5, 3
    threads, column_number, row_number = 4, 2, 2
    measure_name_lists = ["SquaredEuclidean", "Euclidean"]
    index_builder_lists = [
        "SsgBuilder",
        "HnswBuilder",
        "LinearBuilder",
        "ClusteringBuilder",
        "GcBuilder",
        "QcBuilder",
    ]
    builder_params_lists = [
        {},
        {},
        {},
        {"proxima.hc.builder.max_document_count": doc_count},
        {"proxima.gc.builder.centroid_count": "16"},
        {"proxima.qc.builder.centroid_count": "16"},
    ]
    index_searcher_lists = [
        "SsgSearcher",
        "HnswSearcher",
        "LinearSearcher",
        "ClusteringSearcher",
        "GcSearcher",
        "QcSearcher",
    ]
    searcher_params = {}
    index_converter, index_converter_params = None, {}
    index_reformer, index_reformer_params = "", {}

    # data
    doc, query = gen_data(
        doc_count=doc_count, query_count=query_count, dimension=dimension
    )

    # test
    for i, index_builder in enumerate(index_builder_lists):
        for measure_name in measure_name_lists:
            consistency_checking(
                doc=doc,
                query=query,
                dimension=dimension,
                topk=topk,
                threads=threads,
                measure_name=measure_name,
                column_number=column_number,
                row_number=row_number,
                index_builder=index_builder,
                builder_params=builder_params_lists[i],
                index_converter=index_converter,
                index_converter_params=index_converter_params,
                index_searcher=index_searcher_lists[i],
                searcher_params=searcher_params,
                index_reformer=index_reformer,
                index_reformer_params=index_reformer_params,
            )

    # L2 space with HalfFloatConverter
    # params
    doc_count, query_count, dimension, topk = 200, 15, 5, 3
    threads, column_number, row_number = 4, 2, 2
    measure_name_lists = ["SquaredEuclidean", "Euclidean"]
    index_builder_lists = [
        "SsgBuilder",
        "HnswBuilder",
        "LinearBuilder",
        "ClusteringBuilder",
        "GcBuilder",
        "QcBuilder",
    ]
    builder_params_lists = [
        {},
        {},
        {},
        {"proxima.hc.builder.max_document_count": doc_count},
        {"proxima.gc.builder.centroid_count": "16"},
        {"proxima.qc.builder.centroid_count": "16"},
    ]
    index_searcher_lists = [
        "SsgSearcher",
        "HnswSearcher",
        "LinearSearcher",
        "ClusteringSearcher",
        "GcSearcher",
        "QcSearcher",
    ]
    index_converter_lists = [
        "HalfFloatConverter",
        "HalfFloatConverter",
        "HalfFloatConverter",
        "HalfFloatConverter",
        "HalfFloatConverter",
        "HalfFloatConverter",
    ]
    searcher_params = {}
    index_converter, index_converter_params = None, {}
    index_reformer, index_reformer_params = "", {}

    # data
    doc, query = gen_data(
        doc_count=doc_count, query_count=query_count, dimension=dimension
    )

    # test
    for i, index_builder in enumerate(index_builder_lists):
        for measure_name in measure_name_lists:
            consistency_checking(
                doc=doc,
                query=query,
                dimension=dimension,
                topk=topk,
                threads=threads,
                measure_name=measure_name,
                column_number=column_number,
                row_number=row_number,
                index_builder=index_builder,
                builder_params=builder_params_lists[i],
                index_converter=index_converter_lists[i],
                index_converter_params=index_converter_params,
                index_searcher=index_searcher_lists[i],
                searcher_params=searcher_params,
                index_reformer=index_reformer,
                index_reformer_params=index_reformer_params,
                decimal=7,
            )

    # L2 space with Int8QuantizerConverter
    # params
    doc_count, query_count, dimension, topk = 2000, 1, 32, 5
    threads, column_number, row_number = 4, 2, 1

    measure_name_lists = ["SquaredEuclidean", "Euclidean"]
    index_builder_lists = [
        "SsgBuilder",
        "HnswBuilder",
        "LinearBuilder",
        "ClusteringBuilder",
        "GcBuilder",
        "QcBuilder",
    ]
    builder_params_lists = [
        {},
        {},
        {},
        {"proxima.hc.builder.max_document_count": doc_count},
        {"proxima.gc.builder.centroid_count": "16"},
        {
            "proxima.qc.builder.centroid_count": "16",
            "proxima.qc.builder.quantizer_class": "Int8QuantizerConverter",
        },
    ]
    index_searcher_lists = [
        "SsgSearcher",
        "HnswSearcher",
        "LinearSearcher",
        "ClusteringSearcher",
        "GcSearcher",
        "QcSearcher",
    ]
    searcher_params_lists = [
        {},
        {},
        {},
        {"proxima.hc.searcher.scan_ratio": 1},
        {"proxima.gc.searcher.scan_ratio": 1},
        {"proxima.qc.searcher.scan_ratio": 1},
    ]
    index_converter_lists = [
        "Int8QuantizerConverter",
        "Int8QuantizerConverter",
        "Int8QuantizerConverter",
        "Int8QuantizerConverter",
        "Int8QuantizerConverter",
        None,
    ]
    index_converter_params = {}
    index_reformer, index_reformer_params = "", {}

    # data
    doc, query = gen_data(
        doc_count=doc_count, query_count=query_count, dimension=dimension
    )

    # test
    for i, index_builder in enumerate(index_builder_lists):
        for measure_name in measure_name_lists:
            consistency_checking(
                doc=doc,
                query=query,
                dimension=dimension,
                topk=topk,
                threads=threads,
                measure_name=measure_name,
                column_number=column_number,
                row_number=row_number,
                index_builder=index_builder,
                builder_params=builder_params_lists[i],
                index_converter=index_converter_lists[i],
                index_converter_params=index_converter_params,
                index_searcher=index_searcher_lists[i],
                searcher_params=searcher_params_lists[i],
                index_reformer=index_reformer,
                index_reformer_params=index_reformer_params,
                decimal=2,
            )

    # L2 space with Int4QuantizerConverter
    # params
    doc_count, query_count, dimension, topk = 2000, 1, 32, 5
    threads, column_number, row_number = 4, 2, 1

    measure_name_lists = ["SquaredEuclidean", "Euclidean"]
    index_builder_lists = [
        "SsgBuilder",
        "HnswBuilder",
        "LinearBuilder",
        "ClusteringBuilder",
        "GcBuilder",
        "QcBuilder",
    ]
    builder_params_lists = [
        {},
        {},
        {},
        {"proxima.hc.builder.max_document_count": doc_count},
        {"proxima.gc.builder.centroid_count": "16"},
        {
            "proxima.qc.builder.centroid_count": "16",
            "proxima.qc.builder.quantizer_class": "Int4QuantizerConverter",
        },
    ]
    index_searcher_lists = [
        "SsgSearcher",
        "HnswSearcher",
        "LinearSearcher",
        "ClusteringSearcher",
        "GcSearcher",
        "QcSearcher",
    ]
    searcher_params_lists = [
        {},
        {},
        {},
        {"proxima.hc.searcher.scan_ratio": 1},
        {"proxima.gc.searcher.scan_ratio": 1},
        {"proxima.qc.searcher.scan_ratio": 1},
    ]
    index_converter_lists = [
        "Int4QuantizerConverter",
        "Int4QuantizerConverter",
        "Int4QuantizerConverter",
        "Int4QuantizerConverter",
        "Int4QuantizerConverter",
        None,
    ]
    index_converter_params = {}
    index_reformer, index_reformer_params = "", {}

    # data
    doc, query = gen_data(
        doc_count=doc_count,
        query_count=query_count,
        dimension=dimension,
        dtype=np.float32,
    )

    for i, index_builder in enumerate(index_builder_lists):
        for measure_name in measure_name_lists:
            consistency_checking(
                doc=doc,
                query=query,
                dimension=dimension,
                topk=topk,
                threads=threads,
                measure_name=measure_name,
                column_number=column_number,
                row_number=row_number,
                index_builder=index_builder,
                builder_params=builder_params_lists[i],
                index_converter=index_converter_lists[i],
                index_converter_params=index_converter_params,
                index_searcher=index_searcher_lists[i],
                searcher_params=searcher_params_lists[i],
                index_reformer=index_reformer,
                index_reformer_params=index_reformer_params,
                decimal=2,
            )

    # L2 space with NormalizeConverter
    # params
    doc_count, query_count, dimension, topk = 2000, 1, 32, 5
    threads, column_number, row_number = 4, 2, 1

    measure_name_lists = ["SquaredEuclidean", "Euclidean"]
    index_builder_lists = [
        "SsgBuilder",
        "HnswBuilder",
        "LinearBuilder",
        "ClusteringBuilder",
        "GcBuilder",
        "QcBuilder",
    ]
    builder_params_lists = [
        {},
        {},
        {},
        {"proxima.hc.builder.max_document_count": doc_count},
        {"proxima.gc.builder.centroid_count": "16"},
        {"proxima.qc.builder.centroid_count": "16"},
    ]
    index_searcher_lists = [
        "SsgSearcher",
        "HnswSearcher",
        "LinearSearcher",
        "ClusteringSearcher",
        "GcSearcher",
        "QcSearcher",
    ]
    searcher_params_lists = [
        {},
        {},
        {},
        {"proxima.hc.searcher.scan_ratio": 1},
        {"proxima.gc.searcher.scan_ratio": 1},
        {"proxima.qc.searcher.scan_ratio": 1},
    ]
    index_converter_lists = [
        "NormalizeConverter",
        "NormalizeConverter",
        "NormalizeConverter",
        "NormalizeConverter",
        "NormalizeConverter",
        "NormalizeConverter",
    ]
    index_converter_params = {}
    index_reformer, index_reformer_params = "", {}
    # data
    doc, query = gen_data(
        doc_count=doc_count,
        query_count=query_count,
        dimension=dimension,
        dtype=np.float32,
    )

    for i, index_builder in enumerate(index_builder_lists):
        for measure_name in measure_name_lists:
            consistency_checking(
                doc,
                query,
                dimension=dimension,
                topk=topk,
                threads=threads,
                measure_name=measure_name,
                column_number=column_number,
                row_number=row_number,
                index_builder=index_builder,
                builder_params=builder_params_lists[i],
                index_converter=index_converter_lists[i],
                index_converter_params=index_converter_params,
                index_searcher=index_searcher_lists[i],
                searcher_params=searcher_params_lists[i],
                index_reformer=index_reformer,
                index_reformer_params=index_reformer_params,
                decimal=2,
            )

    # InnerProduct space
    # params
    doc_count, query_count, dimension, topk = 200, 15, 5, 2
    threads, column_number, row_number = 4, 2, 2

    measure_name_lists = ["InnerProduct"]
    index_builder_lists = [
        "LinearBuilder",
        "QcBuilder",
        "HnswBuilder",
        "SsgBuilder",
        "ClusteringBuilder",
        "GcBuilder",
    ]
    builder_params_lists = [
        {},
        {"proxima.qc.builder.centroid_count": "16"},
        {},
        {},
        {"proxima.hc.builder.max_document_count": doc_count},
        {"proxima.gc.builder.centroid_count": "16"},
    ]
    index_searcher_lists = [
        "LinearSearcher",
        "QcSearcher",
        "HnswSearcher",
        "SsgSearcher",
        "ClusteringSearcher",
        "GcSearcher",
    ]
    index_converter_lists = [
        None,
        None,
        "MipsConverter",
        "MipsConverter",
        "MipsConverter",
        "MipsConverter",
    ]

    searcher_params = {}
    index_converter_params = {}
    index_reformer, index_reformer_params = "", {}

    # data
    doc, query = gen_data(
        doc_count=doc_count, query_count=query_count, dimension=dimension
    )

    for i, index_builder in enumerate(index_builder_lists):
        for measure_name in measure_name_lists:
            consistency_checking(
                doc,
                query,
                dimension=dimension,
                topk=topk,
                threads=threads,
                measure_name=measure_name,
                column_number=column_number,
                row_number=row_number,
                index_builder=index_builder,
                builder_params=builder_params_lists[i],
                index_converter=index_converter_lists[i],
                index_converter_params=index_converter_params,
                index_searcher=index_searcher_lists[i],
                searcher_params=searcher_params,
                index_reformer=index_reformer,
                index_reformer_params=index_reformer_params,
                decimal=5,
            )


@pytest.mark.skipif(proxima is None, reason="proxima not installed")
def test_build_and_search_index_with_filesystem(setup):
    with tempfile.TemporaryDirectory() as f:
        # params
        doc_count, query_count, dimension = 2000, 50, 10
        topk = 10

        # data
        doc, query = gen_data(
            doc_count=doc_count, query_count=query_count, dimension=dimension
        )

        df = md.DataFrame(pd.DataFrame(doc))
        q = mt.tensor(query)

        index = build_index(tensor=df, index_path=f, column_number=2)

        assert len(os.listdir(f)) > 0

        # proxima_data
        pk_p, distance_p = proxima_build_and_query(doc, query, topk)
        pk_m, distance_m = search_index(tensor=q, topk=topk, index=index, row_number=5)

        # testing
        np.testing.assert_array_equal(pk_p, pk_m)
        np.testing.assert_array_equal(distance_p, distance_m)


@pytest.mark.skipif(proxima is None, reason="proxima not installed")
def test_build_and_search_index_with_filesystem_download(setup):
    with tempfile.TemporaryDirectory() as f:
        # params
        doc_count, query_count, dimension = 2000, 15, 10
        topk = 10
        doc_chunk, query_chunk = 1000, 5

        # data
        doc, query = gen_data(
            doc_count=doc_count, query_count=query_count, dimension=dimension
        )

        df = md.DataFrame(pd.DataFrame(doc), chunk_size=(doc_chunk, dimension))
        q = mt.tensor(query, chunk_size=(query_chunk, dimension))

        index = build_index(tensor=df, index_path=f, column_number=2)

        assert len(os.listdir(f)) > 0

        search_index(q[0:5], topk, index)
        search_index(q[5:10], topk, index)
        search_index(q[10:15], topk, index)


@pytest.mark.skipif(proxima is None, reason="proxima not installed")
def test_recall(setup):
    # params
    doc_count, query_count, dimension = 2000, 150, 20
    topk = 100
    sample_count = 100

    # data
    doc, query = gen_data(
        doc_count=doc_count, query_count=query_count, dimension=dimension
    )

    # proxima_data
    pk_p, distance_p = build_and_query(
        doc,
        query,
        dimension=dimension,
        topk=topk,
        threads=1,
        column_number=2,
        row_number=3,
    )
    assert isinstance(
        recall(
            doc=doc,
            query=query,
            topk=topk,
            sample_count=sample_count,
            pk_p=pk_p,
            distance_p=distance_p,
            column_number=2,
            row_number=2,
        ),
        dict,
    )
