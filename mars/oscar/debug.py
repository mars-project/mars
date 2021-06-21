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
import json
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

from ..utils import dataslots

logger: logging.Logger = logging.getLogger(__name__)


@dataslots
@dataclass
class DebugOptions:
    actor_call_timeout: int = 10
    log_unhandled_errors: bool = True


_debug_opts: Optional[DebugOptions] = None


def get_debug_options() -> Optional[DebugOptions]:
    return _debug_opts


def set_debug_options(options: Optional[DebugOptions]):
    global _debug_opts
    _debug_opts = options

    from .core import set_debug_options as core_set_debug_options
    core_set_debug_options(options)


def reload_debug_opts_from_env():
    if 'DEBUG_OSCAR' not in os.environ:
        set_debug_options(None)
        return
    config_str = os.environ['DEBUG_OSCAR']
    config_json = {} if config_str == '1' else json.loads(config_str)
    set_debug_options(DebugOptions(**config_json))


async def log_actor_timeout(timeout, msg, *args, **kwargs):
    repeat = kwargs.pop('repeat', True)
    while repeat:
        await asyncio.sleep(timeout)
        logger.warning(msg, *args, **kwargs)


@contextmanager
def debug_actor_timeout(option_name: str, msg, *args, **kwargs):
    if _debug_opts is None:
        yield
    else:
        timeout_val = getattr(_debug_opts, option_name, -1)
        timeout_task = None
        if timeout_val and timeout_val > 0:
            timeout_task = asyncio.create_task(log_actor_timeout(
                timeout_val, msg, *args, **kwargs
            ))

        try:
            yield
        finally:
            if timeout_task is not None:
                timeout_task.cancel()


reload_debug_opts_from_env()
