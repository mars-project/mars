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


class DataFrameRuntimeOptimizeRule:
    @staticmethod
    def match(chunk, graph, keys):
        return False

    @staticmethod
    def apply(chunk, graph, keys):
        pass


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
                if rule.match(c, self._graph, keys):
                    rule.apply(c, self._graph, keys)


class ReadCSVHeadRule(DataFrameRuntimeOptimizeRule):
    @staticmethod
    def match(chunk, graph, keys):
        from ....dataframe.indexing.iloc import DataFrameIlocGetItem
        from ....dataframe.datasource.read_csv import DataFrameReadCSV

        op = chunk.op
        inputs = graph.predecessors(chunk)
        if len(inputs) == 1 and isinstance(op, DataFrameIlocGetItem) and \
                op.is_head() and isinstance(inputs[0].op, DataFrameReadCSV) and \
                inputs[0].key not in keys:
            return True
        return False

    @staticmethod
    def apply(chunk, graph, keys):
        read_csv_chunk = graph.predecessors(chunk)[0]
        nrows = read_csv_chunk.op.nrows or 0
        head = chunk.op.indexes[0].stop
        # delete read_csv from graph
        graph.remove_node(read_csv_chunk)

        head_read_csv_chunk_op = read_csv_chunk.op.copy().reset_key()
        head_read_csv_chunk_op._nrows = max(nrows, head)
        head_read_csv_chunk_params = read_csv_chunk.params
        head_read_csv_chunk_params['_key'] = chunk.key
        head_read_csv_chunk = head_read_csv_chunk_op.new_chunk(
            read_csv_chunk.inputs, kws=[head_read_csv_chunk_params]).data
        graph.add_node(head_read_csv_chunk)

        for succ in list(graph.iter_successors(chunk)):
            succ_inputs = succ.inputs
            new_succ_inputs = []
            for succ_input in succ_inputs:
                if succ_input is chunk:
                    new_succ_inputs.append(head_read_csv_chunk)
                else:
                    new_succ_inputs.append(succ_input)
            succ.inputs = new_succ_inputs
            graph.add_edge(head_read_csv_chunk, succ)

        graph.remove_node(chunk)


DataFrameRuntimeOptimizer.register_rule(ReadCSVHeadRule)
