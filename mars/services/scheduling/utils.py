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

import asyncio
import contextlib
import sys
import time
from collections import OrderedDict
from typing import Dict, Mapping, Optional, TypeVar, Iterator

from ... import oscar as mo
from ...lib.aio import alru_cache
from ..subtask import SubtaskResult, SubtaskStatus
from ..task import TaskAPI


@alru_cache
async def _get_task_api(actor: mo.Actor):
    return await TaskAPI.create(getattr(actor, "_session_id"), actor.address)


@contextlib.asynccontextmanager
async def redirect_subtask_errors(actor: mo.Actor, subtasks):
    try:
        yield
    except:  # noqa: E722  # pylint: disable=bare-except
        _, error, traceback = sys.exc_info()
        status = (
            SubtaskStatus.cancelled
            if isinstance(error, asyncio.CancelledError)
            else SubtaskStatus.errored
        )
        task_api = await _get_task_api(actor)
        coros = []
        for subtask in subtasks:
            if subtask is None:  # pragma: no cover
                continue
            coros.append(
                task_api.set_subtask_result(
                    SubtaskResult(
                        subtask_id=subtask.subtask_id,
                        session_id=subtask.session_id,
                        task_id=subtask.task_id,
                        progress=1.0,
                        status=status,
                        error=error,
                        traceback=traceback,
                    )
                )
            )
        await asyncio.wait(coros)
        raise


ResultType = TypeVar("ResultType")


class ResultCache(Mapping[str, ResultType]):
    _cache: Dict[str, ResultType]
    _cache_time: Dict[str, float]
    _duration: float

    def __init__(self, duration: float = 120):
        self._cache = dict()
        self._cache_time = OrderedDict()
        self._duration = duration

    def __getitem__(self, item: str):
        self._del_expired_items()
        return self._cache[item]

    def get(
        self, key: str, default: Optional[ResultType] = None
    ) -> Optional[ResultType]:
        self._del_expired_items()
        return self._cache.get(key, default)

    def _del_expired_items(self):
        keys = []
        expire_time = time.time() - self._duration
        for key, store_time in self._cache_time.items():
            if store_time < expire_time:
                break
            keys.append(key)
        for key in keys:
            self._delitem(key)

    def __setitem__(self, key: str, value):
        self._del_expired_items()
        self._cache[key] = value
        self._cache_time[key] = time.time()

    def _delitem(self, key: str):
        del self._cache[key]
        self._cache_time.pop(key, None)

    def __delitem__(self, key: str):
        self._delitem(key)
        self._del_expired_items()

    def __contains__(self, item: str):
        self._del_expired_items()
        return item in self._cache

    def __len__(self) -> int:
        self._del_expired_items()
        return len(self._cache)

    def __iter__(self) -> Iterator[str]:
        self._del_expired_items()
        return iter(self._cache)

