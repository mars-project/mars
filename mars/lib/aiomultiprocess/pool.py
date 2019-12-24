# Copyright 2019 John Reese
# Licensed under the MIT license

import asyncio
import logging
import os
import queue
import traceback
from typing import Any, Awaitable, Callable, Dict, Optional, Sequence, Tuple

from .core import Process, get_context
from .scheduler import RoundRobin, Scheduler
from .types import PoolTask, ProxyException, Queue, QueueID, R, T, TaskID, TracebackStr

MAX_TASKS_PER_CHILD = 0  # number of tasks to execute before recycling a child process
CHILD_CONCURRENCY = 16  # number of tasks to execute simultaneously per child process

log = logging.getLogger(__name__)


class PoolWorker(Process):
    """Individual worker process for the async pool."""

    def __init__(
        self,
        tx: Queue,
        rx: Queue,
        ttl: int = MAX_TASKS_PER_CHILD,
        concurrency: int = CHILD_CONCURRENCY,
        *,
        initializer: Optional[Callable] = None,
        initargs: Sequence[Any] = (),
    ) -> None:
        super().__init__(target=self.run, initializer=initializer, initargs=initargs)
        self.concurrency = max(1, concurrency)
        self.ttl = max(0, ttl)
        self.tx = tx
        self.rx = rx

    async def run(self) -> None:
        """Pick up work, execute work, return results, rinse, repeat."""
        pending: Dict[asyncio.Future, TaskID] = {}
        completed = 0
        running = True
        while running or pending:
            # TTL, Tasks To Live, determines how many tasks to execute before dying
            if self.ttl and completed >= self.ttl:
                running = False

            # pick up new work as long as we're "running" and we have open slots
            while running and len(pending) < self.concurrency:
                try:
                    task: PoolTask = self.tx.get_nowait()
                except queue.Empty:
                    break

                if task is None:
                    running = False
                    break

                tid, func, args, kwargs = task
                future = asyncio.ensure_future(func(*args, **kwargs))
                pending[future] = tid

            if not pending:
                await asyncio.sleep(0.005)
                continue

            # return results and/or exceptions when completed
            done, _ = await asyncio.wait(
                pending.keys(), timeout=0.05, return_when=asyncio.FIRST_COMPLETED
            )
            for future in done:
                tid = pending.pop(future)

                result = None
                tb = None
                try:
                    result = future.result()
                except BaseException:
                    tb = traceback.format_exc()

                self.rx.put_nowait((tid, result, tb))
                completed += 1


class Pool:
    """Execute coroutines on a pool of child processes."""

    def __init__(
        self,
        processes: int = None,
        initializer: Callable[..., None] = None,
        initargs: Sequence[Any] = (),
        maxtasksperchild: int = MAX_TASKS_PER_CHILD,
        childconcurrency: int = CHILD_CONCURRENCY,
        queuecount: Optional[int] = None,
        scheduler: Scheduler = None,
    ) -> None:
        self.context = get_context()

        self.scheduler = scheduler or RoundRobin()
        self.process_count = max(1, processes or os.cpu_count() or 2)
        self.queue_count = max(1, queuecount or 1)

        if self.queue_count > self.process_count:
            raise ValueError("queue count must be <= process count")

        self.initializer = initializer
        self.initargs = initargs
        self.maxtasksperchild = max(0, maxtasksperchild)
        self.childconcurrency = max(1, childconcurrency)

        self.processes: Dict[Process, QueueID] = {}
        self.queues: Dict[QueueID, Tuple[Queue, Queue]] = {}

        self.running = True
        self.last_id = 0
        self._results: Dict[TaskID, Tuple[Any, Optional[TracebackStr]]] = {}

        self.init()
        self._loop = asyncio.ensure_future(self.loop())

    async def __aenter__(self) -> "Pool":
        """Enable `async with Pool() as pool` usage."""
        return self

    async def __aexit__(self, *args) -> None:
        """Automatically terminate the pool when falling out of scope."""
        self.terminate()
        await self.join()

    def init(self) -> None:
        """Create the initial mapping of processes and queues."""
        for _ in range(self.queue_count):
            tx = self.context.Queue()
            rx = self.context.Queue()
            qid = self.scheduler.register_queue(tx)

            self.queues[qid] = (tx, rx)

        qids = list(self.queues.keys())
        for i in range(self.process_count):
            qid = qids[i % self.queue_count]
            self.processes[self.create_worker(qid)] = qid
            self.scheduler.register_process(qid)

    async def loop(self) -> None:
        """Maintain the pool of workers while open."""
        while self.processes or self.running:
            # clean up workers that reached TTL
            for process in list(self.processes):
                if not process.is_alive():
                    qid = self.processes.pop(process)
                    if self.running:
                        self.processes[self.create_worker(qid)] = qid

            # pull results into a shared dictionary for later retrieval
            for _, rx in self.queues.values():
                while True:
                    try:
                        task_id, value, tb = rx.get_nowait()
                        self.finish_work(task_id, value, tb)

                    except queue.Empty:
                        break

            # let someone else do some work for once
            await asyncio.sleep(0.005)

    def create_worker(self, qid: QueueID) -> Process:
        """Create a worker process attached to the given transmit and receive queues."""
        tx, rx = self.queues[qid]
        process = PoolWorker(
            tx,
            rx,
            self.maxtasksperchild,
            self.childconcurrency,
            initializer=self.initializer,
            initargs=self.initargs,
        )
        process.start()
        return process

    def queue_work(
        self,
        func: Callable[..., Awaitable[R]],
        args: Sequence[Any],
        kwargs: Dict[str, Any],
    ) -> TaskID:
        """Add a new work item to the outgoing queue."""
        self.last_id += 1
        task_id = TaskID(self.last_id)

        qid = self.scheduler.schedule_task(task_id, func, args, kwargs)
        tx, _ = self.queues[qid]
        tx.put_nowait((task_id, func, args, kwargs))
        return task_id

    def finish_work(
        self, task_id: TaskID, value: Any, tb: Optional[TracebackStr]
    ) -> None:
        self._results[task_id] = value, tb
        self.scheduler.complete_task(task_id)

    async def results(self, tids: Sequence[TaskID]) -> Sequence[R]:
        """Wait for all tasks to complete, and return results, preserving order."""
        pending = set(tids)
        ready: Dict[TaskID, R] = {}

        while pending:
            for tid in pending.copy():
                if tid in self._results:
                    result, tb = self._results.pop(tid)
                    if tb is not None:
                        raise ProxyException(tb)
                    ready[tid] = result
                    pending.remove(tid)

            await asyncio.sleep(0.005)

        return [ready[tid] for tid in tids]

    async def apply(
        self,
        func: Callable[..., Awaitable[R]],
        args: Sequence[Any] = None,
        kwds: Dict[str, Any] = None,
    ) -> R:
        """Run a single coroutine on the pool."""
        if not self.running:
            raise RuntimeError(f"pool is closed")

        args = args or ()
        kwds = kwds or {}

        tid = self.queue_work(func, args, kwds)
        results: Sequence[R] = await self.results([tid])
        return results[0]

    async def map(
        self,
        func: Callable[[T], Awaitable[R]],
        iterable: Sequence[T],
        # chunksize: int = None,  # todo: implement chunking maybe
    ) -> Sequence[R]:
        """Run a coroutine once for each item in the iterable."""
        if not self.running:
            raise RuntimeError(f"pool is closed")

        tids = [self.queue_work(func, (item,), {}) for item in iterable]
        return await self.results(tids)

    async def starmap(
        self,
        func: Callable[..., Awaitable[R]],
        iterable: Sequence[Sequence[T]],
        # chunksize: int = None,  # todo: implement chunking maybe
    ) -> Sequence[R]:
        """Run a coroutine once for each sequence of items in the iterable."""
        if not self.running:
            raise RuntimeError(f"pool is closed")

        tids = [self.queue_work(func, args, {}) for args in iterable]
        return await self.results(tids)

    def close(self) -> None:
        """Close the pool to new visitors."""
        self.running = False
        for qid in self.processes.values():
            tx, _ = self.queues[qid]
            tx.put_nowait(None)

    def terminate(self) -> None:
        """No running by the pool!"""
        if self.running:
            self.close()

        for process in self.processes:
            process.terminate()

    async def join(self) -> None:
        """Wait for the pool to finish gracefully."""
        if self.running:
            raise RuntimeError(f"pool is still open")

        await self._loop
