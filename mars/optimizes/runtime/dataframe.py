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

from ...dataframe.operands import ObjectType
from ...dataframe.utils import parse_index
from ...utils import replace_inputs


class DataFrameRuntimeOptimizeRule:
    @staticmethod
    def match(chunk, graph, keys):
        return False

    @classmethod
    def apply(cls, chunk, graph, keys):
        pass

    @staticmethod
    def _replace_successor_inputs(chunk, new_chunk, graph):
        graph.add_node(new_chunk)

        for succ in list(graph.iter_successors(chunk)):
            replace_inputs(succ, chunk, new_chunk)
            graph.add_edge(new_chunk, succ)

        graph.remove_node(chunk)


class DataFrameRuntimeOptimizer:
    _rules = []

    def __init__(self, graph):
        self._graph = graph

    @staticmethod
    def register_rule(rule):
        DataFrameRuntimeOptimizer._rules.append(rule)

    @classmethod
    def is_available(cls):
        return True

    def optimize(self, keys=None):
        visited = set()
        for c in list(self._graph.topological_iter()):
            if c in visited:
                continue
            visited.add(c)
            for rule in self._rules:
                if c not in self._graph:
                    continue
                if rule.match(c, self._graph, keys):
                    rule.apply(c, self._graph, keys)


class DataSourceGetitemRule(DataFrameRuntimeOptimizeRule):
    @staticmethod
    def match(chunk, graph, keys):
        from ...dataframe.datasource.read_csv import DataFrameReadCSV
        from ...dataframe.datasource.read_sql import DataFrameReadSQL
        from ...dataframe.indexing.getitem import DataFrameIndex

        op = chunk.op
        inputs = graph.predecessors(chunk)
        if len(inputs) != 1:
            return False
        else:
            input_successors = graph.successors(inputs[0])
            if len(input_successors) == 1 and isinstance(op, DataFrameIndex) and \
                op.col_names is not None and isinstance(inputs[0].op, (DataFrameReadCSV, DataFrameReadSQL)) and \
                    inputs[0].key not in keys:
                return True
            return False

    @classmethod
    def apply(cls, chunk, graph, keys):
        data_source_chunk = graph.predecessors(chunk)[0]
        data_source_usecols = data_source_chunk.op.usecols or []
        if isinstance(chunk.op.col_names, list):
            usecols = chunk.op.col_names
        else:
            usecols = [chunk.op.col_names]
        extra_cols = [col for col in data_source_usecols if col not in usecols]
        usecols += extra_cols

        # delete datasource chunk from graph
        graph.remove_node(data_source_chunk)

        getitem_data_source_chunk_op = data_source_chunk.op.copy().reset_key()
        getitem_data_source_chunk_op._keep_usecols_order = True
        getitem_data_source_chunk_params = data_source_chunk.params
        dtypes = getitem_data_source_chunk_params.pop('dtypes')
        getitem_data_source_chunk_params['_key'] = chunk.key
        if chunk.ndim == 1:
            getitem_data_source_chunk_op._usecols = usecols[0]
            name = usecols[0]
            getitem_data_source_chunk_params['shape'] = (data_source_chunk.shape[0],)
            getitem_data_source_chunk_params['name'] = name
            getitem_data_source_chunk_params['dtype'] = dtypes[name]
            source_chunk = getitem_data_source_chunk_op.new_chunk(
                data_source_chunk.inputs, kws=[getitem_data_source_chunk_params],
                object_type=ObjectType.series).data
        else:
            getitem_data_source_chunk_op._usecols = usecols
            dtypes = dtypes[usecols]
            getitem_data_source_chunk_params['shape'] = (data_source_chunk.shape[0], len(usecols))
            getitem_data_source_chunk_params['dtypes'] = dtypes
            getitem_data_source_chunk_params['columns_value'] = parse_index(dtypes.index, store_data=True)
            source_chunk = getitem_data_source_chunk_op.new_chunk(
                data_source_chunk.inputs, kws=[getitem_data_source_chunk_params],
                object_type=ObjectType.dataframe).data

        cls._replace_successor_inputs(chunk, source_chunk, graph)


class DataSourceHeadRule(DataFrameRuntimeOptimizeRule):
    @staticmethod
    def match(chunk, graph, keys):
        from ...dataframe.datasource.read_csv import DataFrameReadCSV
        from ...dataframe.datasource.read_sql import DataFrameReadSQL
        from ...dataframe.indexing.iloc import DataFrameIlocGetItem, SeriesIlocGetItem

        op = chunk.op
        inputs = graph.predecessors(chunk)
        if len(inputs) == 1 and isinstance(op, (DataFrameIlocGetItem, SeriesIlocGetItem)) and \
                op.is_head() and isinstance(inputs[0].op, (DataFrameReadCSV, DataFrameReadSQL)) and \
                inputs[0].key not in keys:
            return True
        return False

    @classmethod
    def apply(cls, chunk, graph, keys):
        from ...dataframe.utils import parse_index

        data_source_chunk = graph.predecessors(chunk)[0]
        nrows = data_source_chunk.op.nrows or 0
        head = chunk.op.indexes[0].stop
        # delete datasource chunk from graph
        graph.remove_node(data_source_chunk)

        head_data_source_chunk_op = data_source_chunk.op.copy().reset_key()
        head_data_source_chunk_op._nrows = max(nrows, head)
        head_data_source_chunk_params = data_source_chunk.params
        head_data_source_chunk_params['_key'] = chunk.key
        head_data_source_chunk_params['shape'] = (head,) + chunk.shape[1:]
        if chunk.index_value.has_value():
            pd_index = chunk.index_value.to_pandas()[:head]
            head_data_source_chunk_params['index_value'] = parse_index(pd_index)
        head_data_source_chunk = head_data_source_chunk_op.new_chunk(
            data_source_chunk.inputs, kws=[head_data_source_chunk_params],
            object_type=chunk.op.object_type).data

        cls._replace_successor_inputs(chunk, head_data_source_chunk, graph)


DataFrameRuntimeOptimizer.register_rule(DataSourceHeadRule)
DataFrameRuntimeOptimizer.register_rule(DataSourceGetitemRule)
