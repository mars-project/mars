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

import asyncio.tasks
import contextvars
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional  # noqa: F401

from ..utils import dataslots

logger: logging.Logger = logging.getLogger(__name__)


@dataslots
@dataclass
class DebugOptions:
    actor_call_timeout: int = 10
    process_message_timeout: int = 30
    actor_lock_timeout: int = 30
    ray_object_retrieval_timeout: int = 10
    log_unhandled_errors: bool = True
    log_cycle_send: bool = True


_debug_opts: Optional[DebugOptions] = None


def get_debug_options() -> Optional[DebugOptions]:
    return _debug_opts


def set_debug_options(options: Optional[DebugOptions]):
    global _debug_opts
    _debug_opts = options

    # deliver debug config to native codes for optimization
    from .core import set_debug_options as core_set_debug_options
    core_set_debug_options(options)


def reload_debug_opts_from_env():
    config_str = os.environ.get('DEBUG_OSCAR', '0')
    if config_str == '0':
        set_debug_options(None)
        return
    config_str = os.environ['DEBUG_OSCAR']
    config_json = {} if config_str == '1' else json.loads(config_str)
    set_debug_options(DebugOptions(**config_json))


async def _log_timeout(timeout, msg, *args, **kwargs):
    start_time, rnd = time.time(), 1
    while True:
        await asyncio.sleep(timeout * rnd)
        rnd += 1
        logger.warning(msg + '(timeout for %.4f seconds).', *args, time.time() - start_time, **kwargs)


@contextmanager
def debug_async_timeout(option_name: str, msg, *args, **kwargs):
    if _debug_opts is None:
        yield
    else:
        timeout_val = getattr(_debug_opts, option_name, -1)
        timeout_task = None
        if timeout_val and timeout_val > 0:
            timeout_task = asyncio.create_task(_log_timeout(
                timeout_val, msg, *args, **kwargs
            ))

        try:
            yield
        finally:
            if timeout_task is not None:
                timeout_task.cancel()


_message_trace_var = contextvars.ContextVar('_message_trace_var')


@contextmanager
def record_message_trace(message):
    if _debug_opts is None or not _debug_opts.log_cycle_send:
        yield
    else:
        from .backends.message import MessageTraceItem
        msg_trace = list(message.message_trace or [])
        msg_trace.append(MessageTraceItem(
            uid=message.actor_ref.uid, address=message.actor_ref.address,
            method=message.content[0],
        ))
        _message_trace_var.set(msg_trace)
        try:
            yield
        finally:
            _message_trace_var.set(None)


def detect_cycle_send(message, wait_response: bool = True):
    if _debug_opts is None or not _debug_opts.log_cycle_send or not wait_response:
        return

    from .backends.message import MessageTraceItem

    cur_trace = _message_trace_var.get(None) or []  # type: List[MessageTraceItem]
    message.message_trace = cur_trace

    ref_key = (message.actor_ref.uid, message.actor_ref.address)
    traced_ref_keys = set((item.uid, item.address) for item in cur_trace)
    if ref_key in traced_ref_keys:
        looped_trace = cur_trace + [MessageTraceItem(
            uid=message.actor_ref.uid, address=message.actor_ref.address,
            method=message.content[0],
        )]

        formatted_trace = '\n    '.join(
            f'Calling {t.method!r} in actor {t.uid} at {t.address}'
            for t in looped_trace)
        logger.warning('Call cycle detected when sending to actor %s at %s, the trace is\n'
                       '    %s', message.actor_ref.uid, message.actor_ref.address,
                       formatted_trace)


@contextmanager
def no_message_trace():
    if _debug_opts is None or not _debug_opts.log_cycle_send:
        yield
    else:
        trace = pop_message_trace()
        yield
        set_message_trace(trace)


def pop_message_trace():
    trace = _message_trace_var.get(None)
    _message_trace_var.set(None)
    return trace


def set_message_trace(message_trace):
    _message_trace_var.set(message_trace)


reload_debug_opts_from_env()
