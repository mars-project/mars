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

import numpy as np

import mars.tensor as mt
from mars.scheduler import OperandState
from mars.scheduler.analyzer import GraphAnalyzer
from mars.scheduler.chunkmeta import WorkerMeta
from mars.graph import DAG


class Test(unittest.TestCase):
    def testDepths(self):
        from mars.tensor.arithmetic import TensorAdd
        from mars.tensor.base import TensorSplit
        from mars.tensor.datasource import TensorOnes

        arr = mt.ones(12, chunk_size=4)
        arr_split = mt.split(arr, 2)
        arr_sum = arr_split[0] + arr_split[1]

        graph = arr_sum.build_graph(compose=False, tiled=True)
        analyzer = GraphAnalyzer(graph, {})

        depths = analyzer.calc_depths()
        for n in graph:
            if isinstance(n.op, TensorOnes):
                self.assertEqual(0, depths[n.op.key])
            elif isinstance(n.op, TensorSplit):
                self.assertEqual(1, depths[n.op.key])
            elif isinstance(n.op, TensorAdd):
                self.assertLessEqual(2, depths[n.op.key])

    def testDescendantSize(self):
        arr = mt.random.rand(10, 10, chunk_size=4)
        arr2 = mt.random.rand(10, 10, chunk_size=4)
        arr_dot = arr.dot(arr2)

        graph = arr_dot.build_graph(compose=False, tiled=True)
        analyzer = GraphAnalyzer(graph, {})

        depths = analyzer.calc_depths()
        descendants = analyzer.calc_descendant_sizes()
        nodes = sorted(graph, key=lambda n: depths[n.op.key])
        for idx in range(len(nodes) - 1):
            self.assertGreaterEqual(descendants[nodes[idx].op.key],
                                    descendants[nodes[idx + 1].op.key])

    def testInitialAssignsWithInputs(self):
        import numpy as np
        from mars.tensor.random import TensorRandint
        from mars.tensor.arithmetic import TensorTreeAdd

        n1 = TensorRandint(state=np.random.RandomState(0),
                           dtype=np.float32()).new_chunk(None, shape=(10, 10))
        n2 = TensorRandint(state=np.random.RandomState(1),
                           dtype=np.float32()).new_chunk(None, shape=(10, 10))

        n3 = TensorTreeAdd(args=[], dtype=np.float32()).new_chunk(None, shape=(10, 10))
        n3.op._inputs = [n1, n2]
        n4 = TensorTreeAdd(args=[], dtype=np.float32()).new_chunk(None, shape=(10, 10))
        n4.op._inputs = [n3]

        graph = DAG()
        graph.add_node(n1)
        graph.add_node(n3)
        graph.add_node(n4)
        graph.add_edge(n1, n3)
        graph.add_edge(n3, n4)

        analyzer = GraphAnalyzer(graph, {})
        ext_chunks = analyzer.collect_external_input_chunks(initial=False)
        self.assertListEqual(ext_chunks[n3.op.key], [n2.key])
        self.assertEqual(len(analyzer.collect_external_input_chunks(initial=True)), 0)

    @staticmethod
    def _build_chunk_dag(node_str, edge_str):
        from mars.tensor.random import TensorRandint
        from mars.tensor.arithmetic import TensorTreeAdd

        char_dag = DAG()
        for s in node_str.split(','):
            char_dag.add_node(s.strip())
        for s in edge_str.split(','):
            l, r = s.split('->')
            char_dag.add_edge(l.strip(), r.strip())

        chunk_dag = DAG()
        str_to_chunk = dict()
        for s in char_dag.topological_iter():
            if char_dag.count_predecessors(s):
                c = TensorTreeAdd(args=[], _key=s, dtype=np.float32()).new_chunk(None, shape=(10, 10))
                inputs = c.op._inputs = [str_to_chunk[ps] for ps in char_dag.predecessors(s)]
            else:
                c = TensorRandint(_key=s, dtype=np.float32()).new_chunk(None, shape=(10, 10))
                inputs = []
            str_to_chunk[s] = c
            chunk_dag.add_node(c)
            for inp in inputs:
                chunk_dag.add_edge(inp, c)
        return chunk_dag, str_to_chunk

    def testFullInitialAssign(self):
        r"""
        Proper initial allocation should divide the graph like

        0   1 2   3  |  4   5 6   7  |  8   9 10 11
         \ /   \ /   |   \ /   \ /   |   \ /   \ /
         12    13    |   14    15    |   16    17
        """
        graph, str_to_chunk = self._build_chunk_dag(
            '0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17',
            '0 -> 12, 1 -> 12, 2 -> 13, 3 -> 13, 4 -> 14, 5 -> 14, 6 -> 15, 7 -> 15, '
            '8 -> 16, 9 -> 16, 10 -> 17, 11 -> 17'
        )
        inputs = [(str_to_chunk[str(i)], str_to_chunk[str(i + 1)]) for i in range(0, 12, 2)]

        analyzer = GraphAnalyzer(graph, dict(w1=24, w2=24, w3=24))
        assignments = analyzer.calc_operand_assignments(analyzer.get_initial_operand_keys())
        for inp in inputs:
            self.assertEqual(1, len(set(assignments[n.op.key] for n in inp)))

    def testSameKeyAssign(self):
        r"""
        Proper initial allocation should divide the graph like

         0   1   |   2   3   |   4   5
        | | | |  |  | | | |  |  | | | |
        6   7    |  8   9    |  10  11
        """
        graph, _ = self._build_chunk_dag(
            '0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11',
            '0 -> 6, 0 -> 6, 1 -> 7, 1 -> 7, 2 -> 8, 2 -> 8, 3 -> 9, 3 -> 9, '
            '4 -> 10, 4 -> 10, 5 -> 11, 5 -> 11'
        )
        analyzer = GraphAnalyzer(graph, dict(w1=24, w2=24, w3=24))
        assignments = analyzer.calc_operand_assignments(analyzer.get_initial_operand_keys())
        self.assertEqual(len(assignments), 6)

    def testAssignWithPreviousData(self):
        r"""
        Proper initial allocation should divide the graph like

         0   1  |  2   3  |  4   5
          \ /   |   \ /   |   \ /
           6    |    7    |    8
        """
        graph, _ = self._build_chunk_dag(
            '0, 1, 2, 3, 4, 5, 6, 7, 8',
            '0 -> 6, 1 -> 6, 2 -> 7, 3 -> 7, 4 -> 8, 5 -> 8'
        )

        # assign with partial mismatch
        data_dist = {
            '0': dict(c00=WorkerMeta(chunk_size=5, workers=('w1',)),
                      c01=WorkerMeta(chunk_size=5, workers=('w2',))),
            '1': dict(c10=WorkerMeta(chunk_size=10, workers=('w1',))),
            '2': dict(c20=WorkerMeta(chunk_size=10, workers=('w3',))),
            '3': dict(c30=WorkerMeta(chunk_size=10, workers=('w3',))),
            '4': dict(c40=WorkerMeta(chunk_size=7, workers=('w3',))),
        }
        analyzer = GraphAnalyzer(graph, dict(w1=24, w2=24, w3=24))
        assignments = analyzer.calc_operand_assignments(
            analyzer.get_initial_operand_keys(),
            input_chunk_metas=data_dist
        )

        self.assertEqual(len(assignments), 6)

        # explanation of the result:
        # for '1', all data are in w1, hence assigned to w1
        # '0' assigned to w1 according to connectivity
        # '2' and '3' assigned to w3 according to connectivity
        # '4' assigned to w2 because it has fewer data, and the slots of w3 is used up

        self.assertEqual(assignments['0'], 'w1')
        self.assertEqual(assignments['1'], 'w1')
        self.assertEqual(assignments['2'], 'w3')
        self.assertEqual(assignments['3'], 'w3')
        self.assertEqual(assignments['4'], 'w2')
        self.assertEqual(assignments['5'], 'w2')

        # assign with full mismatch
        data_dist = {
            '0': dict(c00=WorkerMeta(chunk_size=5, workers=('w1',)),
                      c01=WorkerMeta(chunk_size=5, workers=('w1', 'w2',))),
            '1': dict(c10=WorkerMeta(chunk_size=10, workers=('w1',))),
            '2': dict(c20=WorkerMeta(chunk_size=10, workers=('w3',))),
            '3': dict(c30=WorkerMeta(chunk_size=10, workers=('w3',))),
            '4': dict(c40=WorkerMeta(chunk_size=7, workers=('w2',))),
            '5': dict(c50=WorkerMeta(chunk_size=7, workers=('w2',))),
        }
        analyzer = GraphAnalyzer(graph, dict(w1=24, w2=24, w3=24))
        assignments = analyzer.calc_operand_assignments(
            analyzer.get_initial_operand_keys(),
            input_chunk_metas=data_dist
        )

        self.assertEqual(len(assignments), 6)
        self.assertEqual(assignments['0'], 'w1')
        self.assertEqual(assignments['1'], 'w1')
        self.assertEqual(assignments['2'], 'w3')
        self.assertEqual(assignments['3'], 'w3')
        self.assertEqual(assignments['4'], 'w2')
        self.assertEqual(assignments['5'], 'w2')

    def testAssignsHalfway(self):
        from mars.operands import ShuffleProxy

        a = mt.ones((31, 27), chunk_size=10)
        b = a.reshape(27, 31)
        b.op.extra_params['_reshape_with_shuffle'] = True
        graph = b.build_graph(compose=False, tiled=True)

        worker_res = dict(w1=24, w2=24, w3=24)
        analyzer = GraphAnalyzer(graph, worker_res)
        assignments = analyzer.calc_operand_assignments(analyzer.get_initial_operand_keys())
        self.assertSetEqual(set(assignments.values()), set(worker_res))

        shuffle_proxy_chunk = [c for c in graph if isinstance(c.op, ShuffleProxy)][0]
        assignments = analyzer.calc_operand_assignments(
            [c.op.key for c in graph.successors(shuffle_proxy_chunk)])
        self.assertSetEqual(set(assignments.values()), set(worker_res))

    def testAssignWithExpectWorkers(self):
        r"""
        0   1   2   3   4   5
        | x |   | x |   | x |
        6   7E  8E  9E  10E 11
        | x | x | x | x | x |
        12  13  14  15  16  17
        """
        graph, str_to_chunk = self._build_chunk_dag(
            '0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17',
            '0 -> 6, 0 -> 7, 1 -> 6, 1 -> 7, 2 -> 8, 2 -> 9, 3 -> 8, 3 -> 9, '
            '4 -> 10, 4 -> 11, 5 -> 10, 5 -> 11, 6 -> 12, 6 -> 13, 7 -> 12, 7 -> 13, 7 -> 14, '
            '8 -> 13, 8 -> 14, 8 -> 15, 9 -> 14, 9 -> 15, 9 -> 16, 10 -> 15, 10 -> 16, 10 -> 17, '
            '11 -> 16, 11 -> 17'
        )

        worker_res = dict(w1=24, w2=24, w3=24, w4=24)

        expects = {str(n): f'w{n - 6}' for n in range(7, 11)}
        for key, worker in expects.items():
            str_to_chunk[key].op._expect_worker = worker

        analyzer = GraphAnalyzer(graph, worker_res)
        assignments = analyzer.calc_operand_assignments(analyzer.get_initial_operand_keys())
        self.assertSetEqual(set(assignments.values()), set(worker_res))
        self.assertLessEqual(set(expects.items()), set(assignments.items()))

    def testAssignOnWorkerAdd(self):
        r"""
        Proper initial allocation should divide the graph like

        0F  1F  2R  3R   |  4F  5F  6R  7R  |  8R  9R  10R 11R
        | x |   | x |    |  | x |   | x |   |  | x |   | x |
        12R 13R 14U 15U  |  16R 17R 18U 19U |  20U 21U 22U 23U

        U: UNSCHEDULED  F: FINISHED  R: READY
        """
        graph, str_to_chunk = self._build_chunk_dag(
            '0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, '
            '19, 20, 21, 22, 23',
            '0 -> 12, 0 -> 13, 1 -> 12, 1 -> 13, 2 -> 14, 2 -> 15, 3 -> 14, 3 -> 15, '
            '4 -> 16, 4 -> 17, 5 -> 16, 5 -> 17, 6 -> 18, 6 -> 19, 7 -> 18, 7 -> 19, '
            '8 -> 20, 8 -> 21, 9 -> 20, 9 -> 21, 10 -> 22, 10 -> 23, 11 -> 22, 11 -> 23'
        )
        inputs = [(str_to_chunk[str(i)], str_to_chunk[str(i + 1)]) for i in range(0, 12, 2)]
        results = [(str_to_chunk[str(i)], str_to_chunk[str(i + 1)]) for i in range(12, 24, 2)]

        # mark initial assigns
        fixed_assigns = dict()
        op_states = dict()
        for idx in range(2):
            for i in range(2):
                fixed_assigns[inputs[idx][i].op.key] = f'w{idx + 1}'
                op_states[results[idx][i].op.key] = OperandState.READY
                fixed_assigns[results[idx][i].op.key] = f'w{idx + 1}'

        for inp in inputs:
            for n in inp:
                if n.op.key in fixed_assigns:
                    continue
                op_states[n.op.key] = OperandState.READY

        worker_metrics = dict(w1=24, w2=24, w3=24)
        analyzer = GraphAnalyzer(graph, worker_metrics, fixed_assigns, op_states)
        assignments = analyzer.calc_operand_assignments(analyzer.get_initial_operand_keys())
        for inp in inputs:
            if any(n.op.key in fixed_assigns for n in inp):
                continue
            self.assertEqual(1, len(set(assignments[n.op.key] for n in inp)))
        worker_assigns = dict((k, 0) for k in worker_metrics)
        for w in assignments.values():
            worker_assigns[w] += 1
        self.assertEqual(2, worker_assigns['w1'])
        self.assertEqual(2, worker_assigns['w2'])
        self.assertEqual(4, worker_assigns['w3'])

    def testAssignOnWorkerLost(self):
        r"""
        Proper initial allocation should divide the graph like

        0FL 1FL 2F  3F  4R  5R  | 6FL 7FL 8F  9F  10R 11R
        | x |   | x |   | x |   | | x |   | x |   | x |
        12R 13R 14R 15R 16U 17U | 18R 19R 20R 21R 22U 23U

        U: UNSCHEDULED  F: FINISHED  R: READY  L: LOST
        """
        op_states = dict()
        graph, str_to_chunk = self._build_chunk_dag(
            '0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, '
            '19, 20, 21, 22, 23',
            '0 -> 12, 0 -> 13, 1 -> 12, 1 -> 13, 2 -> 14, 2 -> 15, 3 -> 14, 3 -> 15, '
            '4 -> 16, 4 -> 17, 5 -> 16, 5 -> 17, 6 -> 18, 6 -> 19, 7 -> 18, 7 -> 19, '
            '8 -> 20, 8 -> 21, 9 -> 20, 9 -> 21, 10 -> 22, 10 -> 23, 11 -> 22, 11 -> 23'
        )
        inputs = [(str_to_chunk[str(i)], str_to_chunk[str(i + 1)]) for i in range(0, 12, 2)]
        results = [(str_to_chunk[str(i)], str_to_chunk[str(i + 1)]) for i in range(12, 24, 2)]

        fixed_assigns = dict()
        for idx in range(4):
            for i in range(2):
                fixed_assigns[inputs[idx][i].op.key] = f'w{idx % 2 + 1}'
                op_states[inputs[idx][i].op.key] = OperandState.FINISHED
                fixed_assigns[results[idx][i].op.key] = f'w{idx % 2 + 1}'
                op_states[results[idx][i].op.key] = OperandState.READY

        for inp in inputs:
            for n in inp:
                if n.op.key in fixed_assigns:
                    continue
                op_states[n.op.key] = OperandState.READY

        lost_chunks = [c.key for inp in (inputs[0], inputs[2]) for c in inp]

        worker_metrics = dict(w2=24, w3=24)
        analyzer = GraphAnalyzer(graph, worker_metrics, fixed_assigns, op_states, lost_chunks)
        changed_states = analyzer.analyze_state_changes()

        self.assertEqual(len(changed_states), 8)
        self.assertTrue(all(changed_states[c.op.key] == OperandState.READY
                            for inp in (inputs[0], inputs[2]) for c in inp))
        self.assertTrue(all(changed_states[c.op.key] == OperandState.UNSCHEDULED
                            for res in (results[0], results[2]) for c in res))

        assignments = analyzer.calc_operand_assignments(analyzer.get_initial_operand_keys())
        for inp in inputs:
            if any(n.op.key in fixed_assigns for n in inp):
                continue
            self.assertEqual(1, len(set(assignments[n.op.key] for n in inp)))
        worker_assigns = dict((k, 0) for k in worker_metrics)
        for w in assignments.values():
            worker_assigns[w] += 1
        self.assertEqual(2, worker_assigns['w2'])
        self.assertEqual(6, worker_assigns['w3'])
