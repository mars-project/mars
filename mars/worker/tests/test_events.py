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

import asyncio
import pickle

from mars.tests.core import aio_case, create_actor_pool
from mars.config import options
from mars.utils import get_next_port
from mars.worker.tests.base import WorkerCase
from mars.worker.events import EventsActor, EventContext, EventCategory,\
    EventLevel, ResourceEventType, ProcedureEventType


@aio_case
class Test(WorkerCase):
    def setUp(self):
        self._old_event_preserve_time = options.worker.event_preserve_time
        options.worker.event_preserve_time = 0.5

    def tearDown(self):
        options.worker.event_preserve_time = self._old_event_preserve_time

    async def testEvents(self, *_):
        mock_scheduler_addr = '127.0.0.1:%d' % get_next_port()
        async with create_actor_pool(n_process=1, address=mock_scheduler_addr) as pool:
            events_ref = await pool.create_actor(EventsActor)
            event1 = await events_ref.add_single_event(
                EventCategory.RESOURCE, EventLevel.WARNING,
                ResourceEventType.MEM_HIGH, 'test_owner'
            )
            self.assertIsNotNone(event1)

            event2 = await events_ref.add_open_event(
                EventCategory.PROCEDURE, EventLevel.NORMAL,
                ProcedureEventType.CPU_CALC, 'test_owner2'
            )
            self.assertIsNotNone(event2)

            await asyncio.sleep(1)

            proc_events = await events_ref.query_by_time(EventCategory.RESOURCE)
            self.assertEqual(len(proc_events), 0)
            proc_events = await events_ref.query_by_time(EventCategory.PROCEDURE)
            self.assertEqual(len(proc_events), 1)

            await events_ref.close_event(event2)
            proc_events = await events_ref.query_by_time(EventCategory.PROCEDURE)
            self.assertGreater(proc_events[0].time_end, proc_events[0].time_start)

            # repeated closing shall not cause any problems
            await events_ref.close_event(event2)

            reloaded = pickle.loads(pickle.dumps(proc_events[0]))
            self.assertEqual(reloaded.event_id, proc_events[0].event_id)

            async with EventContext(events_ref, EventCategory.PROCEDURE, EventLevel.NORMAL,
                                    ProcedureEventType.CPU_CALC, 'test_owner3'):
                proc_events = await events_ref.query_by_time(EventCategory.PROCEDURE)
                self.assertIsNone(proc_events[-1].time_end)
            self.assertIsNotNone(proc_events[-1].time_end)
