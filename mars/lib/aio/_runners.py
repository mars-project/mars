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


"""
Backport of the asyncio.runners module from Python 3.7.
"""
# Source:
# https://github.com/python/cpython/blob/a4afcdfa55ddffa4b9ae3b0cf101628c7bff4102/Lib/asyncio/runners.py

# Modifications:
# * removed relative imports of .coroutines, .events, .tasks
# * replaced `coroutines`, `events`, `tasks` with `asyncio`.
# * replaced `tasks.all_tasks` with `asyncio.Task.all_tasks` because it is
#   backwards compatible.
# * Use private function `asyncio.events._get_running_loop` directly in
#   Python 3.6

import asyncio
import weakref
from typing import Any, Awaitable, Coroutine, TypeVar, Union


try:
    from asyncio import get_running_loop  # noqa Python >=3.7
except ImportError:  # pragma: no cover
    from asyncio.events import _get_running_loop as get_running_loop  # pragma: no cover

__all__ = ("run", "get_running_loop")
_T = TypeVar("_T")


def _patch_loop(loop):
    """
    This function is designed to work around https://bugs.python.org/issue36607

    It's job is to keep a thread safe variable tasks up to date with any tasks that
    are created for the given loop. This then lets you cancel them as _all_tasks
    was intended for.

    We also need to patch the {get,set}_task_factory functions because we can't allow
    Other users of it to overwrite our factory function. This function will pretend
    like there is no factory set but in reality our factory is always set and we will
    call the provided one set
    """
    tasks = weakref.WeakSet()

    task_factory = [None]

    def _set_task_factory(factory):
        task_factory[0] = factory

    def _get_task_factory():
        return task_factory[0]

    def _safe_task_factory(loop, coro):
        if task_factory[0] is None:
            # These lines are copied from the standard library because they don't have
            # this inside a default factory function for me to call.
            # https://github.com/python/cpython/blob/3.6/Lib/asyncio/base_events.py#L304
            task = asyncio.Task(coro, loop=loop)
            if task._source_traceback:
                del task._source_traceback[-1]  # pragma: no cover
        else:
            task = task_factory[0](loop, coro)
        tasks.add(task)
        return task

    loop.set_task_factory(_safe_task_factory)
    loop.set_task_factory = _set_task_factory
    loop.get_task_factory = _get_task_factory

    return tasks


def run(
        main: Union[Coroutine[Any, None, _T], Awaitable[_T]], *, debug: bool = False
) -> _T:
    """Run a coroutine.

    This function runs the passed coroutine, taking care of
    managing the asyncio event loop and finalizing asynchronous
    generators.

    This function cannot be called when another asyncio event loop is
    running in the same thread.

    If debug is True, the event loop will be run in debug mode.

    This function always creates a new event loop and closes it at the end.
    It should be used as a main entry point for asyncio programs, and should
    ideally only be called once.

    Example:

        async def main():
            await asyncio.sleep(1)
            print('hello')

        asyncio.run(main())
    """
    # Python 3.7+ raises RuntimeError while <3.6 returns None
    try:
        loop = get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        raise RuntimeError("asyncio.run() cannot be called from a running event loop")

    if not asyncio.iscoroutine(main):
        raise ValueError("a coroutine was expected, got {!r}".format(main))

    loop = asyncio.new_event_loop()
    tasks = _patch_loop(loop)

    try:
        asyncio.set_event_loop(loop)
        loop.set_debug(debug)
        return loop.run_until_complete(main)
    finally:
        try:
            _cancel_all_tasks(loop, tasks)
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            asyncio.set_event_loop(None)  # type: ignore
            loop.close()


def _cancel_all_tasks(loop, tasks):
    to_cancel = [task for task in tasks if not task.done()]

    if not to_cancel:
        return

    for task in to_cancel:
        task.cancel()

    loop.run_until_complete(
        asyncio.gather(*to_cancel, loop=loop, return_exceptions=True)
    )

    for task in to_cancel:
        if task.cancelled():
            continue
        if task.exception() is not None:
            loop.call_exception_handler(
                {
                    "message": "unhandled exception during asyncio.run() shutdown",
                    "exception": task.exception(),
                    "task": task,
                }
            )
