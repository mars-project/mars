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


class _ProfilingData:
    def __init__(self):
        self._data = {}

    def init(self, task_id: str):
        self._data[task_id] = {
            "general": {},
            "serialization": {},
        }

    def pop(self, task_id: str):
        return self._data.pop(task_id, None)

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
