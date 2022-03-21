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
from collections.abc import Mapping
from .backends.message import SendMessage, TellMessage
from ..typing import BandType


logger = logging.getLogger(__name__)

MARS_ENABLE_PROFILING = int(os.environ.get("MARS_ENABLE_PROFILING", 0))


class _ProfilingOptionDescriptor:
    def __init__(self, _type, default):
        self._name = None
        self._type = _type
        self._default = default

    def __get__(self, obj, cls):
        if obj is None:
            return self
        v = obj._options.get(self._name)
        if v is None:
            v = os.environ.get(f"MARS_PROFILING_{self._name.upper()}", self._default)
        if v is not None:
            v = self._type(v)
        # Cache the value.
        obj.__dict__[self._name] = v
        return v

    def set_name(self, name: str):
        self._name = name


class _ProfilingOptionsMeta(type):
    def __init__(cls, name, bases, classdict):
        super(_ProfilingOptionsMeta, cls).__init__(name, bases, classdict)
        for k, v in classdict.items():
            if isinstance(v, _ProfilingOptionDescriptor):
                v.set_name(k)


class _ProfilingOptions(metaclass=_ProfilingOptionsMeta):
    debug_interval_seconds = _ProfilingOptionDescriptor(float, default=None)
    slow_calls_duration_threshold = _ProfilingOptionDescriptor(int, default=1)
    slow_subtasks_duration_threshold = _ProfilingOptionDescriptor(int, default=10)

    def __init__(self, options):
        if isinstance(options, Mapping):
            invalid_keys = options.keys() - type(self).__dict__.keys()
            if invalid_keys:
                raise ValueError(f"Invalid profiling options: {invalid_keys}")
            self._options = options
        elif options in (True, False, None):
            self._options = {}
        else:
            raise ValueError(f"Invalid profiling options: {options}")


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


class _CallStats:
    def __init__(self, options: _ProfilingOptions):
        self._options = options
        self._call_counter = Counter()
        self._slow_calls = []

    def collect(self, message, duration: float):
        key = (message.actor_ref.uid, message.content[0])
        self._call_counter[key] += 1
        if duration < self._options.slow_calls_duration_threshold:
            return
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

    def to_dict(self) -> dict:
        most_calls = {}
        for name_tuple, count in self._call_counter.most_common(10):
            uid, method_name = name_tuple
            most_calls[f"{uid.decode('utf-8')}.{method_name}"] = count
        slow_calls = {}
        for duration, uid, address, content in sorted(
            self._slow_calls, key=operator.itemgetter(0), reverse=True
        ):
            method_name, _batch, args, kwargs = content
            slow_calls[
                f"[{address}]{uid.decode('utf-8')}.{method_name}(args={args}, kwargs={kwargs})"
            ] = duration
        return {"most_calls": most_calls, "slow_calls": slow_calls}


class _SubtaskStats:
    def __init__(self, options: _ProfilingOptions):
        self._options = options
        self._band_counter = Counter()
        self._slow_subtasks = []

    def collect(self, subtask, band: BandType, duration: float):
        band_address = band[0]
        self._band_counter[band_address] += 1
        if duration < self._options.slow_subtasks_duration_threshold:
            return
        key = (duration, band_address, subtask)
        try:
            if len(self._slow_subtasks) < 10:
                heapq.heappush(self._slow_subtasks, key)
            else:
                heapq.heapreplace(self._slow_subtasks, key)
        except TypeError:
            pass

    def to_dict(self) -> dict:
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
        return {"band_subtasks": band_subtasks, "slow_subtasks": slow_subtasks}


class _ProfilingData:
    def __init__(self):
        self._data = {}
        self._call_stats = {}
        self._subtask_stats = {}
        self._debug_task = {}

    def init(self, task_id: str, options=None):
        options = _ProfilingOptions(options)
        logger.info(
            "Init profiling data for task %s with debug interval seconds %s.",
            task_id,
            options.debug_interval_seconds,
        )
        self._data[task_id] = {
            "general": {},
            "serialization": {},
            "most_calls": {},
            "slow_calls": {},
            "band_subtasks": {},
            "slow_subtasks": {},
        }
        self._call_stats[task_id] = _CallStats(options)
        self._subtask_stats[task_id] = _SubtaskStats(options)

        async def _debug_profiling_log():
            while True:
                try:
                    r = self._data.get(task_id, None)
                    if r is None:
                        logger.info("Profiling debug log break.")
                        break
                    r = copy.copy(r)  # shadow copy is enough.
                    r.update(self._call_stats.get(task_id).to_dict())
                    r.update(self._subtask_stats.get(task_id).to_dict())
                    logger.warning("Profiling debug:\n%s", json.dumps(r, indent=4))
                except Exception:
                    logger.exception("Profiling debug log failed.")
                await asyncio.sleep(options.debug_interval_seconds)

        if options.debug_interval_seconds is not None:
            self._debug_task[task_id] = task = asyncio.create_task(
                _debug_profiling_log()
            )
            task.add_done_callback(lambda _: self._debug_task.pop(task_id, None))

    def pop(self, task_id: str):
        logger.info("Pop profiling data of task %s.", task_id)
        debug_task = self._debug_task.pop(task_id, None)
        if debug_task is not None:
            debug_task.cancel()
        r = self._data.pop(task_id, None)
        if r is not None:
            r.update(self._call_stats.pop(task_id).to_dict())
            r.update(self._subtask_stats.pop(task_id).to_dict())
        return r

    def collect_actor_call(self, message, duration: float):
        if self._call_stats:
            message_type = type(message)
            if message_type is SendMessage or message_type is TellMessage:
                for stats in self._call_stats.values():
                    stats.collect(message, duration)

    def collect_subtask(self, subtask, band: BandType, duration: float):
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
