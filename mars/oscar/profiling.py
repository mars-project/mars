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
import os
import asyncio
import copy
import json
import heapq
import logging
import operator
from collections import Counter
from .backends.message import SendMessage, TellMessage


logger = logging.getLogger(__name__)


class DummyOperator:
    @staticmethod
    def set(key, value):
        pass

    @staticmethod
    def inc(key, value):
        pass

    @staticmethod
    def nest(key):
        return DummyOperator

    @staticmethod
    def values():
        return []

    @staticmethod
    def empty():
        return True


class ProfilingDataOperator:
    __slots__ = ("_target",)

    def __init__(self, target):
        self._target = target

    def set(self, key, value):
        self._target[key] = value

    def inc(self, key, value):
        old = self._target.get(key, 0)
        self._target[key] = old + value

    def nest(self, key):
        v = self._target.setdefault(key, {})
        if not isinstance(v, dict):
            raise TypeError(
                f"The value type of key {key} is {type(v)}, but a dict is expected."
            )
        return ProfilingDataOperator(v)

    def values(self):
        return self._target.values()

    def empty(self):
        return len(self._target) == 0


class _ActorCallStats:
    def __init__(self):
        self._call_counter = Counter()
        self._slow_calls = []

    def collect(self, message, duration):
        key = (message.actor_ref.uid, message.content[0])
        self._call_counter[key] += 1
        key = (
            duration,
            message.actor_ref.uid,
            message.actor_ref.address,
            message.content,
        )
        try:
            if len(self._slow_calls) < 10:
                heapq.heappush(self._slow_calls, key)
            else:
                heapq.heapreplace(self._slow_calls, key)
        except TypeError:
            pass

    def to_dict(self):
        most_calls = {}
        for name_tuple, count in self._call_counter.most_common(10):
            uid, method_name = name_tuple
            most_calls[f"{uid.decode('utf-8')}.{method_name}"] = count
        slowest_calls = {}
        for duration, uid, address, content in sorted(
            self._slow_calls, key=operator.itemgetter(0), reverse=True
        ):
            method_name, batch, args, kwargs = content
            slowest_calls[
                f"[{address}]{uid.decode('utf-8')}.{method_name}(args={args}, kwargs={kwargs})"
            ] = duration
        return {"most_calls": most_calls, "slowest_calls": slowest_calls}


class _SubtaskStats:
    def __init__(self):
        self._band_counter = Counter()
        self._slow_subtasks = []

    def collect(self, subtask, band, duration):
        band_address = band[0]
        self._band_counter[band_address] += 1
        key = (duration, band_address, subtask)
        try:
            if len(self._slow_subtasks) < 10:
                heapq.heappush(self._slow_subtasks, key)
            else:
                heapq.heapreplace(self._slow_subtasks, key)
        except TypeError:
            pass

    def to_dict(self):
        band_subtasks = {}
        key = operator.itemgetter(1)
        if len(self._band_counter) > 10:
            items = self._band_counter.items()
            band_subtasks.update(heapq.nlargest(5, items, key=key))
            band_subtasks.update(reversed(heapq.nsmallest(5, items, key=key)))
        else:
            band_subtasks.update(
                sorted(self._band_counter.items(), key=key, reverse=True)
            )
        slow_subtasks = {}
        for duration, band, subtask in sorted(
            self._slow_subtasks, key=operator.itemgetter(0), reverse=True
        ):
            slow_subtasks[f"[{band}]{subtask}"] = duration
        return {"band_subtasks": band_subtasks, "slowest_subtasks": slow_subtasks}


class _ProfilingData:
    def __init__(self):
        self._data = {}
        self._actor_call_stats = {}
        self._subtask_stats = {}
        self._debug_task = {}

    def init(self, task_id: str, debug_interval_seconds=None):
        logger.info(
            "Init profiling data for task %s with debug interval seconds %s.",
            task_id,
            debug_interval_seconds,
        )
        self._data[task_id] = {
            "general": {},
            "serialization": {},
            "most_calls": {},
            "slowest_calls": {},
            "band_subtasks": {},
            "slowest_subtasks": {},
        }
        self._actor_call_stats[task_id] = _ActorCallStats()
        self._subtask_stats[task_id] = _SubtaskStats()

        async def _debug_profiling_log():
            while True:
                try:
                    r = self._data.get(task_id, None)
                    if r is None:
                        logger.info("Profiling debug log break.")
                        break
                    r = copy.copy(r)  # shadow copy is enough.
                    r and r.update(self._actor_call_stats.get(task_id).to_dict())
                    r and r.update(self._subtask_stats.get(task_id).to_dict())
                    logger.warning("Profiling debug:\n%s", json.dumps(r, indent=4))
                except Exception:
                    logger.exception("Profiling debug log failed.")
                finally:
                    await asyncio.sleep(debug_interval_seconds)

        if debug_interval_seconds is not None:
            logger.info(
                "Profiling debug log interval second: %s", debug_interval_seconds
            )
            self._debug_task[task_id] = asyncio.create_task(_debug_profiling_log())

    def pop(self, task_id: str):
        logger.info("Pop profiling data of task %s.", task_id)
        debug_task = self._debug_task.pop(task_id, None)
        if debug_task is not None:
            debug_task.cancel()
        r = self._data.pop(task_id, None)
        r and r.update(self._actor_call_stats.pop(task_id).to_dict())
        r and r.update(self._subtask_stats.pop(task_id).to_dict())
        return r

    def collect_actor_call(self, message, duration):
        if self._actor_call_stats:
            message_type = type(message)
            if message_type is SendMessage or message_type is TellMessage:
                for stats in self._actor_call_stats.values():
                    stats.collect(message, duration)

    def collect_subtask(self, subtask, band, duration):
        if self._subtask_stats:
            stats = self._subtask_stats.get(subtask.task_id)
            if stats is not None:
                stats.collect(subtask, band, duration)

    def __getitem__(self, item):
        key = item if isinstance(item, tuple) else (item,)
        v = None
        d = self._data
        for k in key:
            v = d.get(k, None)
            if v is None:
                break
            else:
                d = v
        return DummyOperator if v is None else ProfilingDataOperator(v)


ProfilingData = _ProfilingData()

MARS_ENABLE_PROFILING = bool(os.environ.get("MARS_ENABLE_PROFILING", 0))
MARS_DEBUG_PROFILING_INTERVAL = os.environ.get("MARS_DEBUG_PROFILING_INTERVAL")
MARS_DEBUG_PROFILING_INTERVAL = MARS_DEBUG_PROFILING_INTERVAL and int(
    MARS_DEBUG_PROFILING_INTERVAL
)
