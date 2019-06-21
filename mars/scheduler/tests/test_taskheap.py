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

import queue
import unittest

from mars.scheduler.taskheap import TaskHeap


class Test(unittest.TestCase):
    def _assert_queues_consistent(self, opheap):
        dump = opheap.dump()
        for qid, q in enumerate(dump.queues):
            for idx, e in enumerate(q):
                edump = e.dump()

                # check heap condition satisfied
                idx_c = (idx << 1) + 1
                if idx_c < len(q):
                    self.assertGreaterEqual(edump.store_item.priority,
                                            q[idx_c].dump().store_item.priority)
                idx_c += 1
                if idx_c < len(q):
                    self.assertGreaterEqual(edump.store_item.priority,
                                            q[idx_c].dump().store_item.priority)

                # check position pointers
                self.assertEqual(edump.store_item.qids[edump.position_idx], qid)
                self.assertEqual(edump.store_item.positions[edump.position_idx], idx)

    def testTaskHeap(self):
        taskheap = TaskHeap()
        taskheap.add_group('g1')
        taskheap.add_group('g2')
        self.assertEqual(len(taskheap.dump().queues), 2)

        with self.assertRaises(queue.Empty):
            taskheap.pop_group_task('g1')

        taskheap.add_task('1', 1, ['g1'])
        self._assert_queues_consistent(taskheap)
        taskheap.add_task('2', 2, ['g1', 'g2'])
        self._assert_queues_consistent(taskheap)
        taskheap.add_task('9', 9, ['g2'])
        self._assert_queues_consistent(taskheap)
        taskheap.add_task('7', 7, ['g1', 'g2'])
        self._assert_queues_consistent(taskheap)
        taskheap.add_task('3', 3, ['g1', 'g2'])
        self._assert_queues_consistent(taskheap)
        taskheap.add_task('5', 5, ['g1'])
        self._assert_queues_consistent(taskheap)
        taskheap.add_task('4', 4, ['g2'])
        self._assert_queues_consistent(taskheap)

        # now that
        # g1 has 1, 2, 7, 3, 5
        # g2 has 2, 9, 7, 3, 4

        dump = taskheap.dump()
        self.assertEqual(len(dump.store_items), 7)
        self.assertEqual(len(dump.queues[0]), 5)
        self.assertEqual(len(dump.queues[1]), 5)

        taskheap.update_priority('2', 20)
        self._assert_queues_consistent(taskheap)
        taskheap.update_priority('9', 6)
        self._assert_queues_consistent(taskheap)

        item = taskheap.pop_group_task('g1')
        self._assert_queues_consistent(taskheap)

        dump = taskheap.dump()
        self.assertEqual(item.key, '2')
        self.assertEqual(len(dump.store_items), 6)
        self.assertEqual(len(dump.queues[0]), 4)
        self.assertEqual(len(dump.queues[1]), 4)

        self.assertIsNone(taskheap.pop_group_task('g2', 10))

        # now that
        # g1 has 1, 7, 3, 5
        # g2 has 9(6), 7, 3, 4

        item = taskheap.pop_group_task('g2')
        self._assert_queues_consistent(taskheap)

        dump = taskheap.dump()
        self.assertEqual(len(dump.store_items), 5)
        self.assertEqual(item.key, '7')

        # now that
        # g1 has 1, 3, 5
        # g2 has 9(6), 3, 4

        taskheap.remove_group('g2')
        self._assert_queues_consistent(taskheap)

        dump = taskheap.dump()
        self.assertEqual(len(dump.group_to_queues), 1)
        self.assertEqual(len(dump.store_items), 3)

        # now that
        # g1 has 1, 3, 5

        taskheap.update_priority('3', 6)
        self._assert_queues_consistent(taskheap)
        item = taskheap.pop_group_task('g1')
        self._assert_queues_consistent(taskheap)
        self.assertEqual(item.key, '3')
        self.assertEqual(len(dump.store_items), 2)
        self.assertEqual(len(dump.queues[0]), 2)

        taskheap.add_group('g2')
        dump = taskheap.dump()
        self.assertEqual(len(dump.group_to_queues), 2)
        self.assertEqual(len(dump.queues), 2)

        # now that
        # g1 has 1, 5
        # g2 has nothing

        taskheap.add_task('1', 1, ['g1', 'g2'])
        self._assert_queues_consistent(taskheap)
        dump = taskheap.dump()
        self.assertEqual(len(dump.queues[0]), 2)
        self.assertEqual(len(dump.queues[1]), 1)
