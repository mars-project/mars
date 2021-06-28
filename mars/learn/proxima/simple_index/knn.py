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

import random

import pandas as pd

from .... import dataframe as md
from .... import tensor as mt
from . import build_index, search_index


def sample_data(query, sample_count=10000):
    if sample_count > query.shape[0]:
        sample_count = query.shape[0]

    idx = random.sample(range(query.shape[0]), sample_count)
    sample_query = query[idx, :]
    return sample_query, idx


def linear_build_and_search(doc, query, topk, column_number=None, row_number=None, dimension=None, measure_name=None,
                            threads=4):
    if measure_name is None:
        measure_name = "SquaredEuclidean"
    if dimension is None:
        dimension = doc.shape[1]

    index = build_index(tensor=doc, dimension=dimension, column_number=column_number,
                        distance_metric=measure_name,
                        index_builder="LinearBuilder")

    pk_l, distance_l = search_index(tensor=query, threads=threads, row_number=row_number,
                                    distance_metric=measure_name, dimension=dimension,
                                    topk=topk, index=index)

    return pk_l, distance_l


def build_and_search(doc, query, topk, doc_chunk, query_chunk,
                     index_path=None, threads=4, dimension=None, measure_name=None,
                     need_shuffle=False, storage_options=None,
                     index_builder=None, builder_params=None,
                     index_converter=None, index_converter_params=None,
                     index_searcher=None, searcher_params=None,
                     index_reformer=None, index_reformer_params=None):
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

    doc = md.DataFrame(pd.DataFrame(doc), chunk_size=(doc_chunk, dimension))
    query = mt.tensor(query, chunk_size=(query_chunk, dimension))

    index = build_index(doc, dimension, index_path,
                        need_shuffle, measure_name,
                        index_builder, builder_params,
                        index_converter, index_converter_params,
                        topk, storage_options)

    pk2, distance = search_index(query, topk, index, threads, dimension,
                                 measure_name, index_searcher, searcher_params,
                                 index_reformer, index_reformer_params,
                                 storage_options)

    return pk2, distance
