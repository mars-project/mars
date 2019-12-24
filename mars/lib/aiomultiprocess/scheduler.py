# Copyright 2019 John Reese
# Licensed under the MIT license

import itertools
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, Iterator, List, Sequence

from .types import Queue, QueueID, R, TaskID


class Scheduler(ABC):
    @abstractmethod
    def register_queue(self, tx: Queue) -> QueueID:
        """
        Notify the scheduler when the pool creates a new transmit queue.
        """

    @abstractmethod
    def register_process(self, qid: QueueID) -> None:
        """
        Notify the scheduler when a process is assigned to a queue.

        This should be used for determining weights for the scheduler.
        It will only be called during initial process mapping.
        """

    @abstractmethod
    def schedule_task(
        self,
        task_id: TaskID,
        func: Callable[..., Awaitable[R]],
        args: Sequence[Any],
        kwargs: Dict[str, Any],
    ) -> QueueID:
        """
        Given a task, return a queue ID that it should be sent to.

        `func`, `args` and `kwargs` are just the exact same arguments
        that `queue_work` takes, not every scheduler would be benefit from this.
        Example that they would be useful, highly customized schedule may want
        to schedule according to function/arguments weights.
        """

    @abstractmethod
    def complete_task(self, task_id: TaskID) -> None:
        """
        Notify the scheduler that a task has been completed.
        """


class RoundRobin(Scheduler):
    """
    The default scheduling algorithm that assigns tasks to queues randomly.

    When multiple processes are assigned to the same queue, this will weight tasks
    accordingly.  For example, 12 processes over 8 queues should result in four queues
    receiving double the number tasks of
    """

    def __init__(self) -> None:
        super().__init__()
        self.qids: List[QueueID] = []
        self.next_id = itertools.count()
        self.cycler: Iterator[QueueID] = itertools.cycle([])

    def register_queue(self, tx: Queue) -> QueueID:
        return QueueID(next(self.next_id))

    def register_process(self, qid: QueueID) -> None:
        self.qids.append(qid)
        self.cycler = itertools.cycle(self.qids)

    def schedule_task(
        self,
        _task_id: TaskID,
        _func: Callable[..., Awaitable[R]],
        _args: Sequence[Any],
        _kwargs: Dict[str, Any],
    ) -> QueueID:
        return next(self.cycler)

    def complete_task(self, _task_id: TaskID) -> None:
        pass
