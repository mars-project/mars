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
from typing import Iterable

from ... import oscar as mo
from ..subtask import Subtask, SubtaskResult, SubtaskStatus
from ..task import TaskAPI


@contextlib.asynccontextmanager
async def redirect_subtask_errors(
    actor: mo.Actor, subtasks: Iterable[Subtask], reraise: bool = True
):
    try:
        yield
    except:  # noqa: E722  # pylint: disable=bare-except
        _, error, traceback = sys.exc_info()
        status = (
            SubtaskStatus.cancelled
            if isinstance(error, asyncio.CancelledError)
            else SubtaskStatus.errored
        )
        task_api = await TaskAPI.create(getattr(actor, "_session_id"), actor.address)
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
                        stage_id=subtask.stage_id,
                        progress=1.0,
                        status=status,
                        error=error,
                        traceback=traceback,
                    )
                )
            )
        tasks = [asyncio.ensure_future(coro) for coro in coros]
        await asyncio.wait(tasks)
        if reraise:
            raise
