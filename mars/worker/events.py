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

import bisect
import itertools
import time
import uuid
from collections import defaultdict, deque
from enum import Enum

from ..config import options
from ..utils import tokenize
from .utils import WorkerActor


class EventCategory(Enum):
    RESOURCE = 0
    PROCEDURE = 1


class EventLevel(Enum):
    NORMAL = 0
    WARNING = 1
    ERROR = 2
    CRITICAL = 3


class ResourceEventType(Enum):
    MEM_HIGH = 0


class ProcedureEventType(Enum):
    __order__ = 'CPU_CALC GPU_CALC DISK_IO NETWORK'

    CPU_CALC = 0
    GPU_CALC = 1
    DISK_IO = 2
    NETWORK = 3


class WorkerEvent(object):
    __slots__ = 'event_id', 'category', 'level', 'event_type', 'owner', 'time_start', 'time_end'

    def __init__(self, category=None, level=None, event_type=None, owner=None,
                 time_start=None, time_end=None, event_id=None):
        self.category = category
        self.level = level or EventLevel.NORMAL
        self.event_type = event_type
        self.owner = owner
        self.time_start = time_start
        self.time_end = time_end

        self.event_id = event_id or tokenize(
            uuid.getnode(), time.time(), category, level, event_type, owner)


class EventsActor(WorkerActor):
    def __init__(self):
        super().__init__()
        self._event_timelines = defaultdict(deque)
        self._id_to_open_event = dict()

    def add_single_event(self, category, level, event_type, owner=None):
        event_obj = WorkerEvent(
            category=category, level=level, event_type=event_type, owner=owner,
            time_start=time.time(), time_end=time.time())
        self._event_timelines[category].append((time.time(), event_obj))

        self._purge_old_events(category)
        return event_obj.event_id

    def add_open_event(self, category, level, event_type, owner=None):
        event_obj = WorkerEvent(
            category=category, level=level, event_type=event_type, owner=owner,
            time_start=time.time(), time_end=None)
        self._event_timelines[category].append((time.time(), event_obj))
        self._id_to_open_event[event_obj.event_id] = event_obj

        self._purge_old_events(category)
        return event_obj.event_id

    def close_event(self, event_id):
        try:
            event_obj = self._id_to_open_event.pop(event_id)
        except KeyError:
            return
        event_obj.time_end = time.time()
        self._event_timelines[event_obj.category].append((time.time(), event_obj))
        self._purge_old_events(event_obj.category)

    def query_by_time(self, category, time_start=None, time_end=None):
        self._purge_old_events(category)

        class ItemWrapper(object):
            def __init__(self, tl):
                self._l = tl

            def __getitem__(self, item):
                return self._l[item][0]

            def __len__(self):
                return len(self._l)

        timeline = self._event_timelines[category]
        left_pos = 0 if time_start is None \
            else bisect.bisect_left(ItemWrapper(timeline), time_start)
        right_pos = len(timeline) if time_end is None \
            else bisect.bisect_right(ItemWrapper(timeline), time_end)
        return [it[1] for it in itertools.islice(timeline, left_pos, right_pos)]

    def _purge_old_events(self, category):
        check_time = time.time()
        min_accept_time = check_time - options.worker.event_preserve_time

        timeline = self._event_timelines[category]
        while timeline and timeline[0][1].time_end is not None \
                and timeline[0][1].time_end < min_accept_time:
            timeline.popleft()


class EventContext(object):
    def __init__(self, events_ref, category, level, event_type, owner=None):
        self._events_ref = events_ref
        if events_ref is not None:
            self._event_id = events_ref.add_open_event(category, level, event_type, owner)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._events_ref is not None:
            self._events_ref.close_event(self._event_id)
