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
import tempfile
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import mars.dataframe as md
from mars.core import TileableGraph, TileableGraphBuilder, enter_mode
from mars.optimization.logical.tileable import optimize


@pytest.fixture(scope='module')
def gen_data1():
    with tempfile.TemporaryDirectory() as tempdir:
        df = pd.DataFrame({'a': [3, 4, 5, 3, 5, 4, 1, 2, 3],
                           'b': [1, 3, 4, 5, 6, 5, 4, 4, 4],
                           'c': list('aabaaddce'),
                           'd': list('abaaaddce')})
        yield df, tempdir


@enter_mode(build=True)
def test_groupby_read_csv(gen_data1):
    pdf, tempdir = gen_data1
    file_path = os.path.join(tempdir, 'test.csv')
    pdf.to_csv(file_path)

    df1 = md.read_csv(file_path)
    df2 = df1.groupby('c').agg({'a': 'sum'})
    df3 = df2 + 1
    graph = TileableGraph([df3.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    opt_df1 = records.get_optimization_result(df1.data)
    assert opt_df1 is not None
    opt_df2 = records.get_optimization_result(df2.data)
    assert opt_df2 is not None
    opt_df3 = records.get_optimization_result(df3.data)
    assert opt_df3 is None
    assert opt_df1 in graph.predecessors(opt_df2)
    assert opt_df1 in opt_df2.inputs
    assert opt_df1.op.usecols == ['a', 'c']
    assert opt_df2 in graph.predecessors(df3.data)
    assert opt_df2 in df3.inputs

    df4 = md.read_csv(file_path, usecols=['a', 'b', 'c'])
    df5 = df4.groupby('c').agg({'b': 'sum'})
    graph = TileableGraph([df5.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    opt_df4 = records.get_optimization_result(df4.data)
    assert opt_df4 is not None
    opt_df5 = records.get_optimization_result(df5.data)
    assert opt_df5 is not None
    assert opt_df4.op.usecols == ['b', 'c']

    df6 = md.read_csv(file_path)
    df7 = df6.groupby('c').agg({'b': 'sum'})
    df8 = df6.groupby('b').agg({'a': 'sum'})
    graph = TileableGraph([df7.data, df8.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    opt_df6 = records.get_optimization_result(df6.data)
    assert opt_df6 is not None
    opt_df7 = records.get_optimization_result(df7.data)
    assert opt_df7 is not None
    opt_df8 = records.get_optimization_result(df8.data)
    assert opt_df8 is not None
    assert opt_df6.op.usecols == ['a', 'b', 'c']
    # original tileable should not be modified
    assert df7.inputs[0] is df6.data
    assert df8.inputs[0] is df6.data

    # test data source in result tileables
    graph = TileableGraph([df6.data, df7.data, df8.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    opt_df6 = records.get_optimization_result(df6.data)
    assert opt_df6 is None
    opt_df7 = records.get_optimization_result(df7.data)
    assert opt_df7 is None
    opt_df8 = records.get_optimization_result(df8.data)
    assert opt_df8 is None


@enter_mode(build=True)
def test_groupby_prune_read_parquet(gen_data1):
    pdf, tempdir = gen_data1
    file_path = os.path.join(tempdir, 'test.parquet')
    pdf.to_parquet(file_path)

    df1 = md.read_parquet(file_path)
    df2 = df1.groupby('c').agg({'a': 'sum'})
    graph = TileableGraph([df2.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    opt_df1 = records.get_optimization_result(df1.data)
    assert opt_df1 is not None
    opt_df2 = records.get_optimization_result(df2.data)
    assert opt_df2 is not None
    assert opt_df1.op.columns == ['a', 'c']
    # original tileable should not be modified
    assert df2.inputs[0] is df1.data

    df3 = df1.groupby('c', as_index=False).c.agg({'cnt': 'count'})
    graph = TileableGraph([df3.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    opt_df1 = records.get_optimization_result(df1.data)
    assert opt_df1 is not None
    opt_df3 = records.get_optimization_result(df3.data)
    assert opt_df3 is not None
    assert opt_df1.op.columns == ['c']


@enter_mode(build=True)
def test_getitem_prune_read_parquet(gen_data1):
    pdf, tempdir = gen_data1
    file_path = os.path.join(tempdir, 'test.parquet')
    pdf.to_parquet(file_path)

    df1 = md.read_parquet(file_path)
    df2 = df1.c
    df3 = df1[['a']]
    graph = TileableGraph([df2.data, df3.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)

    opt_df1 = records.get_optimization_result(df1.data)
    assert opt_df1 is not None
    opt_df2 = records.get_optimization_result(df2.data)
    assert opt_df2 is not None
    opt_df3 = records.get_optimization_result(df3.data)
    assert opt_df3 is not None
    assert opt_df1 in graph.predecessors(opt_df2)
    assert opt_df1 in opt_df2.inputs
    assert opt_df1 in graph.predecessors(opt_df3)
    assert opt_df1 in opt_df3.inputs
    assert opt_df1.op.columns == ['a', 'c']
    assert opt_df1 in graph.predecessors(opt_df3)
    assert opt_df1 in opt_df3.inputs
    # original tileable should not be modified
    assert df2.inputs[0] is df1.data
    assert df3.inputs[0] is df1.data


@pytest.fixture(scope='module')
def gen_data2():
    with tempfile.TemporaryDirectory() as tempdir:
        df = pd.DataFrame({'a': np.arange(10).astype(np.int64, copy=False),
                           'b': [f's{i}' for i in range(10)],
                           'c': np.random.rand(10),
                           'd': [datetime.fromtimestamp(time.time() + 3600 * (i - 5))
                                 for i in range(10)]})
        yield df, tempdir


@enter_mode(build=True)
def test_groupby_prune_read_sql(gen_data2):
    pdf, tempdir = gen_data2
    uri = 'sqlite:///' + os.path.join(tempdir, 'test.db')
    table_name = 'test'
    pdf.to_sql(table_name, uri, index=False)

    # test read df with columns
    df1 = md.read_sql_table('test', uri, chunk_size=4)
    df2 = df1.groupby('a', as_index=False).a.agg({'cnt': 'count'})
    graph = TileableGraph([df2.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    opt_df1 = records.get_optimization_result(df1.data)
    assert opt_df1 is not None
    opt_df2 = records.get_optimization_result(df2.data)
    assert opt_df2 is not None
    assert opt_df1.op.columns == ['a']
    # original tileable should not be modified
    assert df2.inputs[0] is df1.data


@enter_mode(build=True)
def test_groupby_and_getitem(gen_data1):
    pdf, tempdir = gen_data1
    file_path = os.path.join(tempdir, 'test.csv')
    pdf.to_csv(file_path)

    df1 = md.read_csv(file_path)
    df2 = df1.groupby('c').agg({'a': 'sum'})
    df3 = df1[['b', 'a']]
    graph = TileableGraph([df2.data, df3.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    opt_df1 = records.get_optimization_result(df1.data)
    assert opt_df1 is not None
    opt_df2 = records.get_optimization_result(df2.data)
    assert opt_df2 is not None
    assert opt_df1 in graph.predecessors(opt_df2)
    opt_df3 = records.get_optimization_result(df3.data)
    assert opt_df3 is not None
    assert opt_df1 in graph.predecessors(opt_df3)
    assert opt_df1.op.usecols == ['a', 'b', 'c']
    # original tileable should not be modified
    assert df2.inputs[0] is df1.data
    assert df3.inputs[0] is df1.data


@enter_mode(build=True)
def test_cannot_prune(gen_data1):
    pdf, tempdir = gen_data1
    file_path = os.path.join(tempdir, 'test.csv')
    pdf.to_csv(file_path)

    df1 = md.read_csv(file_path)
    df2 = df1.groupby('c').agg({'a': 'sum'})
    # does not support prune
    df3 = df1 + 1
    graph = TileableGraph([df2.data, df3.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    opt_df1 = records.get_optimization_result(df1.data)
    assert opt_df1 is None
    opt_df2 = records.get_optimization_result(df2.data)
    assert opt_df2 is None
    opt_df3 = records.get_optimization_result(df3.data)
    assert opt_df3 is None

    df1 = md.read_csv(file_path)
    df2 = df1.groupby('c').agg({'a': 'sum'})
    # does not support prune, another rule
    df3 = df1.head(3)
    graph = TileableGraph([df2.data, df3.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    opt_df1 = records.get_optimization_result(df1.data)
    assert opt_df1 is None
    opt_df2 = records.get_optimization_result(df2.data)
    assert opt_df2 is None
    opt_df3 = records.get_optimization_result(df3.data)
    assert opt_df3 is None

    df1 = md.read_csv(file_path)
    df2 = df1[df1.dtypes.index.tolist()]
    graph = TileableGraph([df2.data])
    next(TileableGraphBuilder(graph).build())
    # all columns selected
    records = optimize(graph)
    opt_df1 = records.get_optimization_result(df1.data)
    assert opt_df1 is None
    opt_df2 = records.get_optimization_result(df2.data)
    assert opt_df2 is None
