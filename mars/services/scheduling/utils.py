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

from ... import oscar as mo
from ...lib.aio import alru_cache
from ..subtask import SubtaskResult, SubtaskStatus
from ..task import TaskAPI


@alru_cache
async def _get_task_api(actor: mo.Actor):
    return await TaskAPI.create(getattr(actor, '_session_id'), actor.address)


@contextlib.asynccontextmanager
async def redirect_subtask_errors(actor: mo.Actor, subtasks):
    try:
        yield
    except:  # noqa: E722  # pylint: disable=bare-except
        _, error, traceback = sys.exc_info()
        task_api = await _get_task_api(actor)
        coros = []
        for subtask in subtasks:
            if subtask is None:  # pragma: no cover
                continue
            coros.append(task_api.set_subtask_result(SubtaskResult(
                subtask_id=subtask.subtask_id,
                session_id=subtask.session_id,
                task_id=subtask.task_id,
                progress=1.0,
                status=SubtaskStatus.errored,
                error=error, traceback=traceback
            )))
        await asyncio.wait(coros)
        raise
