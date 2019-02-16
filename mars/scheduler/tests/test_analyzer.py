# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

import mars.tensor as mt
from mars.scheduler.analyzer import GraphAnalyzer
from mars.graph import DAG


class Test(unittest.TestCase):
    def testDepths(self):
        from mars.tensor.expressions.arithmetic import TensorAdd
        from mars.tensor.expressions.base import TensorSplit
        from mars.tensor.expressions.datasource import TensorOnes

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

    def testFullInitialAssign(self):
        import numpy as np
        from mars.tensor.expressions.random import TensorRandint
        from mars.tensor.expressions.arithmetic import TensorTreeAdd

        graph = DAG()

        r"""
        Proper initial allocation should divide the graph like
        
        U   U U   U  |  U   U U   U  |  U   U U   U 
         \ /   \ /   |   \ /   \ /   |   \ /   \ /  
          U     U    |    U     U    |    U     U   
        """

        inputs = [
            tuple(TensorRandint(dtype=np.float32()).new_chunk(None, (10, 10)) for _ in range(2))
            for _ in range(6)
        ]
        results = [TensorTreeAdd(dtype=np.float32()).new_chunk(None, (10, 10)) for _ in range(6)]
        for inp, r in zip(inputs, results):
            r.op._inputs = list(inp)

            graph.add_node(r)
            for n in inp:
                graph.add_node(n)
                graph.add_edge(n, r)

        analyzer = GraphAnalyzer(graph, dict(w1=24, w2=24, w3=24))
        assignments = analyzer.calc_initial_assignments()
        for inp in inputs:
            self.assertEqual(1, len(set(assignments[n.op.key] for n in inp)))

    def testSameKeyAssign(self):
        import numpy as np
        from mars.tensor.expressions.random import TensorRandint
        from mars.tensor.expressions.arithmetic import TensorTreeAdd

        graph = DAG()

        r"""
        Proper initial allocation should divide the graph like
        
         U   U   |   U   U   |   U   U  
        | | | |  |  | | | |  |  | | | | 
         U   U   |   U   U   |   U   U  
        """

        inputs = [
            tuple(TensorRandint(_key=str(i), dtype=np.float32()).new_chunk(None, (10, 10)) for _ in range(2))
            for i in range(6)
        ]
        results = [TensorTreeAdd(dtype=np.float32()).new_chunk(None, (10, 10)) for _ in range(6)]
        for inp, r in zip(inputs, results):
            r.op._inputs = list(inp)

            graph.add_node(r)
            for n in inp:
                graph.add_node(n)
                graph.add_edge(n, r)

        analyzer = GraphAnalyzer(graph, dict(w1=24, w2=24, w3=24))
        assignments = analyzer.calc_initial_assignments()
        self.assertEqual(len(assignments), 6)

    def testAssignOnWorkerAdd(self):
        import numpy as np
        from mars.scheduler import OperandState
        from mars.tensor.expressions.random import TensorRandint
        from mars.tensor.expressions.arithmetic import TensorTreeAdd

        graph = DAG()

        r"""
        Proper initial allocation should divide the graph like
        
        F   F R   R  |  F   F R   R  |  R   R R   R 
        | x | | x |  |  | x | | x |  |  | x | | x |
        R   R U   U  |  R   R U   U  |  U   U U   U
        
        U: UNSCHEDULED  F: FINISHED  R: READY
        """

        inputs = [
            tuple(TensorRandint(dtype=np.float32()).new_chunk(None, (10, 10)) for _ in range(2))
            for _ in range(6)
        ]
        results = [
            tuple(TensorTreeAdd(_key='%d_%d' % (i, j), dtype=np.float32()).new_chunk(None, (10, 10))
                  for j in range(2)) for i in range(6)
        ]
        for inp, outp in zip(inputs, results):
            for o in outp:
                o.op._inputs = list(inp)
                graph.add_node(o)

            for n in inp:
                graph.add_node(n)
                for o in outp:
                    graph.add_edge(n, o)

        # mark initial assigns
        fixed_assigns = dict()
        op_states = dict()
        for idx in range(2):
            for i in range(2):
                fixed_assigns[inputs[idx][i].op.key] = 'w%d' % (idx + 1)
                op_states[results[idx][i].op.key] = OperandState.READY

        for inp in inputs:
            for n in inp:
                if n.op.key in fixed_assigns:
                    continue
                op_states[n.op.key] = OperandState.READY

        worker_metrics = dict(w1=24, w2=24, w3=24)
        analyzer = GraphAnalyzer(graph, worker_metrics, fixed_assigns, op_states)
        assignments = analyzer.calc_initial_assignments()
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
        import numpy as np
        from mars.scheduler import OperandState
        from mars.tensor.expressions.random import TensorRandint
        from mars.tensor.expressions.arithmetic import TensorTreeAdd

        graph = DAG()

        r"""
        Proper initial allocation should divide the graph like
        
        FL  FL F   F R   R  |  FL  FL F   F R   R
        | x |  | x | | x |  |  | x |  | x | | x |
        R   R  R   R U   U  |  R   R  R   R U   U
        
        U: UNSCHEDULED  F: FINISHED  R: READY  L: LOST
        """

        op_states = dict()
        inputs = [
            tuple(TensorRandint(dtype=np.float32()).new_chunk(None, (10, 10)) for _ in range(2))
            for _ in range(6)
        ]
        results = [
            tuple(TensorTreeAdd(_key='%d_%d' % (i, j), dtype=np.float32()).new_chunk(None, (10, 10))
                  for j in range(2)) for i in range(6)
        ]
        for inp, outp in zip(inputs, results):
            for o in outp:
                o.op._inputs = list(inp)
                op_states[o.op.key] = OperandState.UNSCHEDULED
                graph.add_node(o)

            for n in inp:
                op_states[n.op.key] = OperandState.UNSCHEDULED
                graph.add_node(n)
                for o in outp:
                    graph.add_edge(n, o)

        fixed_assigns = dict()
        for idx in range(4):
            for i in range(2):
                fixed_assigns[inputs[idx][i].op.key] = 'w%d' % (idx % 2 + 1)
                op_states[inputs[idx][i].op.key] = OperandState.FINISHED
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

        assignments = analyzer.calc_initial_assignments()
        for inp in inputs:
            if any(n.op.key in fixed_assigns for n in inp):
                continue
            self.assertEqual(1, len(set(assignments[n.op.key] for n in inp)))
        worker_assigns = dict((k, 0) for k in worker_metrics)
        for w in assignments.values():
            worker_assigns[w] += 1
        self.assertEqual(2, worker_assigns['w2'])
        self.assertEqual(6, worker_assigns['w3'])
