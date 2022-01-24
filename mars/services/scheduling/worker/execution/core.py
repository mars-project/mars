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
import logging
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .....core.base import MarsError
from .....utils import dataslots
from ....subtask import Subtask, SubtaskResult

logger = logging.getLogger(__name__)

# the default times to run subtask.
DEFAULT_SUBTASK_MAX_RETRIES = 0


@dataslots
@dataclass
class SubtaskExecutionInfo:
    subtask: Subtask
    priority: Tuple
    supervisor_address: str
    band_name: str
    aio_tasks: List[asyncio.Task] = field(default_factory=list)
    result: SubtaskResult = field(default_factory=SubtaskResult)
    cancelling: bool = False
    max_retries: int = field(default=DEFAULT_SUBTASK_MAX_RETRIES)
    num_retries: int = 0
    finish_future: asyncio.Future = field(default_factory=asyncio.Future)
    quota_request: Optional[Dict] = None
    slot_id: Optional[int] = None
    kill_timeout: Optional[int] = None
    # if True, when a subtask stops execution, its successors
    # will be forwarded as soon as possible
    forward_successors: bool = False


async def call_with_retry(
    async_fun: Callable, max_retries: int, error_callback: Callable
):
    max_retries = max_retries or 1

    for trial in range(max_retries):
        try:
            return await async_fun()
        except (OSError, MarsError):
            exc_info_raw = sys.exc_info()
            exc_info = error_callback(trial=trial, exc_info=exc_info_raw, retry=True)
            exc_info = exc_info or exc_info_raw

            if trial >= max_retries - 1:
                raise exc_info[1].with_traceback(exc_info[-1])
        except asyncio.CancelledError:
            raise
        except:  # noqa: E722  # nosec  # pylint: disable=bare-except
            exc_info_raw = sys.exc_info()
            exc_info = error_callback(trial=trial, exc_info=exc_info_raw, retry=False)
            exc_info = exc_info or exc_info_raw

            raise exc_info[1].with_traceback(exc_info[-1])
