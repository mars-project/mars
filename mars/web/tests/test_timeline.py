# -*- coding: utf-8 -*-
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

import time
import unittest
from datetime import datetime

from mars.worker.events import WorkerEvent, EventCategory, EventLevel, \
    ProcedureEventType
from mars.web.worker_pages import EventUpdater


class Test(unittest.TestCase):
    def testEventUpdate(self):
        updater = EventUpdater()
        self.assertIsNone(updater.x_range, None)
        self.assertIsNone(updater.y_range, None)

        base_time = int(time.time()) - 20
        events = [
            WorkerEvent(EventCategory.PROCEDURE, EventLevel.NORMAL, ProcedureEventType.CPU_CALC,
                        'owner1', base_time),
            WorkerEvent(EventCategory.PROCEDURE, EventLevel.NORMAL, ProcedureEventType.CPU_CALC,
                        'owner2', base_time),
            WorkerEvent(EventCategory.PROCEDURE, EventLevel.NORMAL, ProcedureEventType.CPU_CALC,
                        'owner3', base_time, base_time + 1),
        ]
        range_dfs, patches = updater.update_events(events, base_time + 10)
        df = range_dfs[ProcedureEventType.CPU_CALC]

        self.assertEqual(updater.x_range, (datetime.fromtimestamp(base_time),
                                           datetime.fromtimestamp(base_time + 10)))
        # three horizontal bars with ranges (0.5, 1.5), (1.5, 2.5), (2.5, 3.5) respectively
        self.assertEqual(updater.y_range, (0.5, 3.5))

        self.assertEqual(df.iloc[0].right, datetime.fromtimestamp(base_time + 10))
        self.assertEqual(df.iloc[1].right, datetime.fromtimestamp(base_time + 10))
        self.assertEqual(df.iloc[2].right, datetime.fromtimestamp(base_time + 1))
        self.assertEqual(len(patches), 0)

        events2 = [
            WorkerEvent(EventCategory.PROCEDURE, EventLevel.NORMAL, ProcedureEventType.CPU_CALC,
                        'owner2', base_time, base_time + 10, event_id=events[1].event_id),
        ]
        range_dfs, patches = updater.update_events(events2, base_time + 20)
        df = range_dfs[ProcedureEventType.CPU_CALC]
        patch = sorted(patches[ProcedureEventType.CPU_CALC])
        self.assertEqual(len(df), 0)
        self.assertEqual(patch[0], (0, datetime.fromtimestamp(base_time + 20)))
        self.assertEqual(patch[1], (1, datetime.fromtimestamp(base_time + 10)))
