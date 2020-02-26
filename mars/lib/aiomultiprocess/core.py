# Copyright 2018 John Reese
# Licensed under the MIT license

import asyncio
import logging
import multiprocessing
import multiprocessing.managers
import os
import sys
from typing import Any, Callable, Dict, Optional, Sequence

from .types import Context, R, Unit

DEFAULT_START_METHOD = "spawn"

# shared context for all multiprocessing primitives, for platform compatibility
# "spawn" is default/required on windows and mac, but can't execute non-global functions
# see https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
context = multiprocessing.get_context(DEFAULT_START_METHOD)
_manager = None

log = logging.getLogger(__name__)


def get_manager() -> multiprocessing.managers.SyncManager:
    """Return a singleton shared manager."""
    global _manager
    if _manager is None:
        _manager = context.Manager()

    return _manager


def set_start_method(method: Optional[str] = DEFAULT_START_METHOD) -> None:
    """
    Set the start method and context used for future processes/pools.

    When given no parameters (`set_context()`), will default to using the "spawn" method
    as this provides a predictable set of features and compatibility across all major
    platforms, and trades a small cost on process startup for potentially large savings
    on memory usage of child processes.

    Passing an explicit string (eg, "fork") will force aiomultiprocess to use the given
    start method instead of "spawn".

    Passing an explicit `None` value will force aiomultiprocess to use CPython's default
    start method for the current platform rather than defaulting to "spawn".

    See the official multiprocessing documentation for details on start methods:
    https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    """
    global context
    context = multiprocessing.get_context(method)


def get_context() -> Context:
    """Get the current active global context."""
    global context
    return context


def set_context(method: Optional[str] = None) -> None:
    """
    Set the start method and context used for future processes/pools. [DEPRECATED]

    Retained for backwards compatibility, and to retain prior behavior of "no parameter"
    resulting in selection of the platform's default start method.
    """
    return set_start_method(method)


async def not_implemented(*args: Any, **kwargs: Any) -> None:
    """Default function to call when none given."""
    raise NotImplementedError()


class Process:
    """Execute a coroutine on a separate process."""

    def __init__(
        self,
        group: None = None,
        target: Callable = None,
        name: str = None,
        args: Sequence[Any] = None,
        kwargs: Dict[str, Any] = None,
        *,
        daemon: bool = None,
        initializer: Optional[Callable] = None,
        initargs: Sequence[Any] = (),
        process_target: Optional[Callable] = None,
    ) -> None:
        # if target is not None and not asyncio.iscoroutinefunction(target):
        #     raise ValueError(f"target must be coroutine function")

        if initializer is not None and asyncio.iscoroutinefunction(initializer):
            raise ValueError(f"initializer must be synchronous function")

        self.unit = Unit(
            target=target or not_implemented,
            args=args or (),
            kwargs=kwargs or {},
            namespace=get_manager().Namespace(),
            initializer=initializer,
            initargs=initargs,
        )
        self.aio_process = context.Process(
            group=group,
            target=process_target or Process.run_async,
            args=(self.unit,),
            name=name,
            daemon=daemon,
        )

    def __await__(self) -> Any:
        """Enable awaiting of the process result by chaining to `start()` & `join()`."""
        if not self.is_alive() and self.exitcode is None:
            self.start()

        return self.join().__await__()

    @staticmethod
    def run_async(unit: Unit) -> R:
        """Initialize the child process and event loop, then execute the coroutine."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            if unit.initializer:
                unit.initializer(*unit.initargs)

            result: R = loop.run_until_complete(unit.target(*unit.args, **unit.kwargs))

            return result

        except BaseException:
            log.exception(f"aio process {os.getpid()} failed")
            raise

    def start(self) -> None:
        """Start the child process."""
        return self.aio_process.start()

    async def join(self, timeout: int = None) -> None:
        """Wait for the process to finish execution without blocking the main thread."""
        if not self.is_alive() and self.exitcode is None:
            raise ValueError("must start process before joining it")

        if timeout is not None:
            return await asyncio.wait_for(self.join(), timeout)

        while self.exitcode is None:
            await asyncio.sleep(0.005)

    @property
    def name(self) -> str:
        """Child process name."""
        return self.aio_process.name

    def is_alive(self) -> bool:
        """Is child process running."""
        return self.aio_process.is_alive()

    @property
    def daemon(self) -> bool:
        """Should child process be daemon."""
        return self.aio_process.daemon

    @daemon.setter
    def daemon(self, value: bool) -> None:
        """Should child process be daemon."""
        self.aio_process.daemon = value

    @property
    def pid(self) -> Optional[int]:
        """Process ID of child, or None if not started."""
        return self.aio_process.pid

    @property
    def exitcode(self) -> Optional[int]:
        """Exit code from child process, or None if still running."""
        return self.aio_process.exitcode

    def terminate(self) -> None:
        """Send SIGTERM to child process."""
        return self.aio_process.terminate()

    # multiprocessing.Process methods added in 3.7
    if sys.version_info >= (3, 7):

        def kill(self) -> None:
            """Send SIGKILL to child process."""
            return self.aio_process.kill()

        def close(self) -> None:
            """Clean up child process once finished."""
            return self.aio_process.close()


class Worker(Process):
    """Execute a coroutine on a separate process and return the result."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, process_target=Worker.run_async, **kwargs)
        self.unit.namespace.result = None

    @staticmethod
    def run_async(unit: Unit) -> R:
        """Initialize the child process and event loop, then execute the coroutine."""
        try:
            result: R = Process.run_async(unit)
            unit.namespace.result = result
            return result

        except BaseException as e:
            unit.namespace.result = e
            raise

    async def join(self, timeout: int = None) -> Any:
        """Wait for the worker to finish, and return the final result."""
        await super().join(timeout)
        return self.result

    @property
    def result(self) -> R:
        """Easy access to the resulting value from the coroutine."""
        if self.exitcode is None:
            raise ValueError("coroutine not completed")

        return self.unit.namespace.result
