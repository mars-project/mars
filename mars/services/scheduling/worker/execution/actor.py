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
import functools
import logging
import sys
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from ..... import oscar as mo
from .....core.graph import DAG
from .....core.operand import Fetch
from .....lib.aio import alru_cache
from .....oscar.errors import MarsError
from ....cluster import ClusterAPI
from ....core import ActorCallback
from ....subtask import Subtask, SubtaskAPI, SubtaskResult, SubtaskStatus
from ..queues import SubtaskPrepareQueueActor, SubtaskExecutionQueueActor
from ..quota import QuotaActor
from ..slotmanager import SlotManagerActor
from .core import call_with_retry, SubtaskExecutionInfo
from .prepare import SubtaskPreparer, PrepareFastFailed

logger = logging.getLogger(__name__)

# the default times to run subtask.
DEFAULT_SUBTASK_MAX_RETRIES = 0


class SubtaskExecutionActor(mo.Actor):
    _pred_key_mapping_dag: DAG
    _subtask_caches: Dict[str, SubtaskExecutionInfo]
    _subtask_executions: Dict[str, SubtaskExecutionInfo]

    _prepare_queue_ref: Union[SubtaskPrepareQueueActor, mo.ActorRef, None]
    _execution_queue_ref: Union[SubtaskExecutionQueueActor, mo.ActorRef, None]
    _slot_manager_ref: Union[SlotManagerActor, mo.ActorRef, None]

    _subtask_api: SubtaskAPI
    _subtask_preparer: SubtaskPreparer

    def __init__(
        self,
        subtask_max_retries: int = None,
        enable_kill_slot: bool = True,
    ):
        self._pred_key_mapping_dag = DAG()
        self._subtask_caches = dict()
        self._subtask_executions = dict()
        self._prepare_queue_ref = None
        self._execution_queue_ref = None

        self._subtask_max_retries = subtask_max_retries or DEFAULT_SUBTASK_MAX_RETRIES
        self._enable_kill_slot = enable_kill_slot

    async def __post_create__(self):
        self._prepare_queue_ref = await mo.actor_ref(
            SubtaskPrepareQueueActor.default_uid(), address=self.address
        )
        self._execution_queue_ref = await mo.actor_ref(
            SubtaskExecutionQueueActor.default_uid(), address=self.address
        )
        self._slot_manager_ref = await mo.actor_ref(
            SlotManagerActor.default_uid(), address=self.address
        )
        self._subtask_api = await SubtaskAPI.create(self.address)
        self._subtask_preparer = SubtaskPreparer(self.address)

        cluster_api = await ClusterAPI.create(self.address)

        self._prepare_process_tasks = dict()
        self._execution_process_tasks = dict()
        for band in await cluster_api.get_bands():
            self._prepare_process_tasks[band] = asyncio.create_task(
                self.handle_prepare_queue(band[1])
            )
            self._execution_process_tasks[band] = asyncio.create_task(
                self.handle_execute_queue(band[1])
            )

    async def __pre_destroy__(self):
        for prepare_task in self._prepare_process_tasks.values():
            prepare_task.cancel()
        for exec_task in self._execution_process_tasks.values():
            exec_task.cancel()

    @alru_cache(cache_exceptions=False)
    async def _get_band_quota_ref(
        self, band_name: str
    ) -> Union[mo.ActorRef, QuotaActor]:
        return await mo.actor_ref(QuotaActor.gen_uid(band_name), address=self.address)

    @staticmethod
    @alru_cache(cache_exceptions=False)
    async def _get_manager_ref(session_id: str, supervisor_address: str):
        from ...supervisor.manager import SubtaskManagerActor

        return await mo.actor_ref(
            uid=SubtaskManagerActor.gen_uid(session_id),
            address=supervisor_address,
        )

    def _build_subtask_info(
        self,
        subtask: Subtask,
        priority: Tuple,
        supervisor_address: str,
        band_name: str,
    ) -> SubtaskExecutionInfo:
        subtask_max_retries = (
            subtask.extra_config.get("subtask_max_retries")
            if subtask.extra_config
            else None
        )
        if subtask_max_retries is None:
            subtask_max_retries = self._subtask_max_retries

        subtask_info = SubtaskExecutionInfo(
            subtask,
            priority,
            supervisor_address=supervisor_address,
            band_name=band_name,
            max_retries=subtask_max_retries,
        )
        subtask_info.result = SubtaskResult(
            subtask_id=subtask.subtask_id,
            session_id=subtask.session_id,
            task_id=subtask.task_id,
            status=SubtaskStatus.pending,
        )
        return subtask_info

    async def cache_subtasks(
        self,
        subtasks: List[Subtask],
        priorities: List[Tuple],
        supervisor_address: str,
        band_name: str,
    ):
        for subtask, priority in zip(subtasks, priorities):
            if subtask.subtask_id in self._subtask_executions:
                continue
            self._subtask_caches[subtask.subtask_id] = self._build_subtask_info(
                subtask,
                priority,
                supervisor_address=supervisor_address,
                band_name=band_name,
            )
            mapping_dag = self._pred_key_mapping_dag
            for chunk in subtask.chunk_graph:
                if isinstance(chunk.op, Fetch):
                    mapping_dag.add_node(chunk.key)
                    mapping_dag.add_node(subtask.subtask_id)
                    mapping_dag.add_edge(chunk.key, subtask.subtask_id)

    def uncache_subtasks(self, subtask_ids: List[str]):
        for subtask_id in subtask_ids:
            subtask_info = self._subtask_caches.pop(subtask_id, None)
            if subtask_info is None:
                continue
            subtask = subtask_info.subtask

            try:
                self._pred_key_mapping_dag.remove_node(subtask.subtask_id)
            except KeyError:
                continue

            for chunk in subtask.chunk_graph:
                if not isinstance(chunk.op, Fetch):
                    continue
                try:
                    if self._pred_key_mapping_dag.count_successors(chunk.key) == 0:
                        self._pred_key_mapping_dag.remove_node(chunk.key)
                except KeyError:
                    continue

    async def update_subtask_priorities(
        self, subtask_ids: List[str], priorities: List[Tuple]
    ):
        assert len(subtask_ids) == len(priorities)

        prepare_delays = []
        execution_delays = []
        for subtask_id, priority in zip(subtask_ids, priorities):
            try:
                subtask_info = self._subtask_caches[subtask_id]
                subtask_info.priority = priority
            except KeyError:
                pass

            try:
                subtask_info = self._subtask_executions[subtask_id]
            except KeyError:
                continue
            if subtask_info.quota_request is not None:
                self._execution_queue_ref.update_priority.delay(
                    subtask_id, subtask_info.band_name, priority
                )
            else:
                self._prepare_queue_ref.update_priority.delay(
                    subtask_id, subtask_info.band_name, priority
                )
        if prepare_delays:
            await self._prepare_queue_ref.update_priority.batch(*prepare_delays)
        if execution_delays:
            await self._execution_queue_ref.update_priority.batch(*execution_delays)

    async def submit_subtasks(
        self,
        subtasks: List[Union[str, Subtask]],
        priorities: List[Tuple],
        supervisor_address: str,
        band_name: str,
    ):
        assert len(subtasks) == len(priorities)
        logger.debug("%d subtasks submitted to SubtaskExecutionActor", len(subtasks))

        put_delays = []
        for subtask, priority in zip(subtasks, priorities):
            if isinstance(subtask, str):
                try:
                    subtask = self._subtask_caches[subtask].subtask
                except KeyError:
                    subtask = self._subtask_executions[subtask].subtask
            try:
                info = self._subtask_executions[subtask.subtask_id]
                if info.result.status not in (
                    SubtaskStatus.cancelled,
                    SubtaskStatus.errored,
                ):
                    continue
            except KeyError:
                pass

            subtask_info = self._build_subtask_info(
                subtask,
                priority,
                supervisor_address=supervisor_address,
                band_name=band_name,
            )
            self._subtask_caches.pop(subtask.subtask_id, None)
            self._subtask_executions[subtask.subtask_id] = subtask_info
            put_delays.append(
                self._prepare_queue_ref.put.delay(
                    subtask.subtask_id, band_name, priority
                )
            )

        if put_delays:
            await self._prepare_queue_ref.put.batch(*put_delays)

    async def _dequeue_subtask_ids(self, queue_ref, subtask_ids: List[str]):
        removes = [queue_ref.remove.delay(subtask_id) for subtask_id in subtask_ids]
        removed_subtask_ids = await queue_ref.remove.batch(*removes)

        infos_to_report = []
        for subtask_id in removed_subtask_ids:
            if subtask_id is None:
                continue
            subtask_info = self._subtask_caches.get(subtask_id)
            if subtask_info is None:
                subtask_info = self._subtask_executions[subtask_id]
            if not subtask_info.finish_future.done():
                self._fill_result_with_exc(subtask_info, exc_cls=asyncio.CancelledError)
                infos_to_report.append(subtask_info)
        await self._report_subtask_results(infos_to_report)

    async def _report_subtask_results(self, subtask_infos: List[SubtaskExecutionInfo]):
        if not subtask_infos:
            return
        try:
            manager_ref = await self._get_manager_ref(
                subtask_infos[0].result.session_id, subtask_infos[0].supervisor_address
            )
        except mo.ActorNotExist:
            return
        await manager_ref.set_subtask_results.tell(
            [info.result for info in subtask_infos],
            [(self.address, info.band_name) for info in subtask_infos],
        )

    async def cancel_subtasks(
        self, subtask_ids: List[str], kill_timeout: Optional[int] = 5
    ):
        aio_tasks = []
        for subtask_id in subtask_ids:
            try:
                subtask_info = self._subtask_executions[subtask_id]
            except KeyError:
                continue

            subtask_info.kill_timeout = kill_timeout
            subtask_info.cancelling = True
            aio_tasks.extend(subtask_info.aio_tasks)
            for aio_task in subtask_info.aio_tasks:
                aio_task.cancel()

        await self._dequeue_subtask_ids(self._prepare_queue_ref, subtask_ids)
        await self._dequeue_subtask_ids(self._execution_queue_ref, subtask_ids)

        if aio_tasks:
            yield asyncio.wait(aio_tasks)

        self.uncache_subtasks(subtask_ids)

        infos_to_report = []
        for subtask_id in subtask_ids:
            try:
                subtask_info = self._subtask_executions[subtask_id]
            except KeyError:
                continue
            if not subtask_info.result.status.is_done:
                self._fill_result_with_exc(subtask_info, exc_cls=asyncio.CancelledError)
                infos_to_report.append(subtask_info)
        await self._report_subtask_results(infos_to_report)

    async def wait_subtasks(self, subtask_ids: List[str]):
        infos = [
            self._subtask_executions[stid]
            for stid in subtask_ids
            if stid in self._subtask_executions
        ]
        if infos:
            yield asyncio.wait([info.finish_future for info in infos])
        raise mo.Return([info.result for info in infos])

    def _create_subtask_with_exception(self, subtask_id, coro):
        info = self._subtask_executions[subtask_id]

        async def _run_with_exception_handling():
            try:
                return await coro
            except:  # noqa: E722  # nosec  # pylint: disable=bare-except
                self._fill_result_with_exc(info)
                await self._report_subtask_results([info])
                await self._prepare_queue_ref.release_slot(
                    info.subtask.subtask_id, errors="ignore"
                )
                await self._execution_queue_ref.release_slot(
                    info.subtask.subtask_id, errors="ignore"
                )
                for aio_task in info.aio_tasks:
                    if aio_task is not asyncio.current_task():
                        aio_task.cancel()

        task = asyncio.create_task(_run_with_exception_handling())
        info.aio_tasks.append(task)

    async def handle_prepare_queue(self, band_name: str):
        while True:
            try:
                subtask_id, _ = await self._prepare_queue_ref.get(band_name)
            except asyncio.CancelledError:
                break
            except:  # pragma: no cover
                logger.exception("Errored when waiting for prepare queue")
                continue

            subtask_info = self._subtask_executions[subtask_id]
            if subtask_info.cancelling:
                continue

            logger.debug(f"Obtained subtask {subtask_id} from prepare queue")
            self._create_subtask_with_exception(
                subtask_id, self._prepare_subtask_with_retry(subtask_info)
            )

    async def handle_execute_queue(self, band_name: str):
        while True:
            try:
                subtask_id, slot_id = await self._execution_queue_ref.get(band_name)
            except asyncio.CancelledError:
                break
            except:  # pragma: no cover
                logger.exception("Errored when waiting for execution queue")
                continue

            if subtask_id not in self._subtask_executions:
                continue

            subtask_info = self._subtask_executions[subtask_id]
            if subtask_info.cancelling:
                continue

            logger.debug(f"Obtained subtask {subtask_id} from execution queue")
            await self._prepare_queue_ref.release_slot(
                subtask_info.subtask.subtask_id, errors="ignore"
            )
            subtask_info.band_name = band_name
            subtask_info.slot_id = slot_id
            # if any successors already cached, it must be scheduled
            # as soon as possible, thus fast-forward is disabled
            subtask_info.forward_successors = any(
                c.key in self._pred_key_mapping_dag
                for c in subtask_info.subtask.chunk_graph.result_chunks
            )
            self._create_subtask_with_exception(
                subtask_id, self._execute_subtask_with_retry(subtask_info)
            )

    async def _prepare_subtask_once(self, subtask_info: SubtaskExecutionInfo):
        return await self._subtask_preparer.run(subtask_info, fail_fast=False)

    async def _prepare_subtask_with_retry(self, subtask_info: SubtaskExecutionInfo):
        try:
            await self._call_with_retry(self._prepare_subtask_once, subtask_info)
        except:  # noqa: E722  # nosec  # pylint: disable=bare-except
            self._fill_result_with_exc(subtask_info)
            await self._report_subtask_results([subtask_info])
            await self._prepare_queue_ref.release_slot(
                subtask_info.subtask.subtask_id, errors="ignore"
            )
        else:
            await self._execution_queue_ref.put(
                subtask_info.subtask.subtask_id,
                subtask_info.band_name,
                subtask_info.priority,
            )

    async def _execute_subtask_once(self, subtask_info: SubtaskExecutionInfo):
        subtask = subtask_info.subtask
        band_name = subtask_info.band_name
        slot_id = subtask_info.slot_id

        finish_callback = ActorCallback(
            self.ref().schedule_next, args=(subtask.subtask_id,)
        )
        run_task = asyncio.create_task(
            self._subtask_api.run_subtask_in_slot(
                band_name,
                slot_id,
                subtask,
                forward_successors=subtask_info.forward_successors,
                finish_callback=finish_callback,
            )
        )
        try:
            return await asyncio.shield(run_task)
        except asyncio.CancelledError as ex:
            subtask_info.cancelling = True
            cancel_coro = self._subtask_api.cancel_subtask_in_slot(band_name, slot_id)
            try:
                kill_timeout = (
                    subtask_info.kill_timeout if self._enable_kill_slot else None
                )
                await asyncio.wait_for(asyncio.shield(cancel_coro), kill_timeout)
            except asyncio.TimeoutError:
                logger.debug(
                    "Wait for subtask to cancel timed out (%s). "
                    "Start killing slot %d",
                    subtask_info.kill_timeout,
                    subtask_info.slot_id,
                )
                await self._slot_manager_ref.kill_slot(band_name, slot_id)
            except:  # pragma: no cover
                logger.exception("Unexpected errors raised when handling cancel")
                raise
            raise ex
        except (OSError, MarsError) as ex:
            if slot_id is not None:
                # may encounter subprocess memory error
                sub_pool_address = await self._slot_manager_ref.get_slot_address(
                    band_name, slot_id
                )
                await mo.wait_actor_pool_recovered(sub_pool_address, self.address)
            raise ex
        finally:
            logger.debug(
                "Subtask %s running ended, slot_id=%r", subtask.subtask_id, slot_id
            )

    async def _execute_subtask_with_retry(self, subtask_info: SubtaskExecutionInfo):
        subtask = subtask_info.subtask
        try:
            subtask_info.result = await self._call_with_retry(
                self._execute_subtask_once,
                subtask_info,
                max_retries=subtask_info.max_retries if subtask.retryable else 0,
            )
        except Exception as ex:  # noqa: E722  # nosec  # pylint: disable=bare-except
            if not subtask.retryable:
                unretryable_op = [
                    chunk.op
                    for chunk in subtask.chunk_graph
                    if not getattr(chunk.op, "retryable", True)
                ]
                logger.exception(
                    "Run subtask failed due to %r, the subtask %s is "
                    "not retryable, it contains unretryable op: %r",
                    ex,
                    subtask.subtask_id,
                    unretryable_op,
                )
            self._fill_result_with_exc(subtask_info)
        finally:
            self._subtask_executions.pop(subtask.subtask_id, None)
        if not subtask_info.finish_future.done():
            subtask_info.finish_future.set_result(None)

        await self._report_subtask_results([subtask_info])
        await self._execution_queue_ref.release_slot(
            subtask.subtask_id, errors="ignore"
        )
        quota_ref = await self._get_band_quota_ref(subtask_info.band_name)
        await quota_ref.release_quotas(tuple(subtask_info.quota_request.keys()))
        return subtask_info.result

    @classmethod
    def _log_subtask_retry(
        cls,
        subtask_info: SubtaskExecutionInfo,
        target_func: Callable,
        trial: int,
        exc_info: Tuple,
        retry: bool = True,
    ):
        subtask = subtask_info.subtask
        max_retries = subtask_info.max_retries
        subtask_info.num_retries = trial
        if retry:
            if trial < max_retries - 1:
                logger.error(
                    "Rerun %s of subtask %s at attempt %d due to %s",
                    target_func,
                    subtask.subtask_id,
                    trial + 1,
                    exc_info[1],
                )
            else:
                logger.exception(
                    "Exceed max rerun (%s / %s): %s of subtask %s due to %s",
                    trial + 1,
                    max_retries,
                    target_func,
                    subtask.subtask_id,
                    exc_info[1],
                    exc_info=exc_info,
                )
        else:
            logger.exception(
                "Failed to rerun %s of subtask %s due to unhandled exception: %s",
                target_func,
                subtask.subtask_id,
                exc_info[1],
                exc_info=exc_info,
            )

    @classmethod
    async def _call_with_retry(
        cls,
        target_func: Callable,
        subtask_info: SubtaskExecutionInfo,
        max_retries: Optional[int] = None,
    ):
        subtask_info.max_retries = max_retries or subtask_info.max_retries

        if subtask_info.max_retries <= 1:
            return await target_func(subtask_info)
        else:
            err_callback = functools.partial(
                cls._log_subtask_retry, subtask_info, target_func
            )
            return await call_with_retry(
                functools.partial(target_func, subtask_info),
                max_retries=max_retries,
                error_callback=err_callback,
            )

    @classmethod
    def _fill_result_with_exc(
        cls,
        subtask_info: SubtaskExecutionInfo,
        exc_info: Optional[Tuple] = None,
        exc_cls: Type[Exception] = None,
    ):
        subtask = subtask_info.subtask
        if exc_cls is not None:
            try:
                raise exc_cls
            except:
                exc_info = sys.exc_info()
        exc_type, exc, tb = exc_info or sys.exc_info()

        if isinstance(exc, asyncio.CancelledError):
            status = SubtaskStatus.cancelled
            log_str = "Cancel"
        else:
            status = SubtaskStatus.errored
            log_str = "Failed to"

        logger.exception(
            "%s run subtask %s on band %s",
            log_str,
            subtask.subtask_id,
            subtask_info.band_name,
            exc_info=(exc_type, exc, tb),
        )

        subtask_info.result = SubtaskResult(
            subtask_id=subtask.subtask_id,
            session_id=subtask.session_id,
            task_id=subtask.task_id,
            progress=1.0,
            status=status,
            error=exc,
            traceback=tb,
        )

        if not subtask_info.finish_future.done():
            subtask_info.finish_future.set_result(None)

    async def schedule_next(self, pred_subtask_id: str):
        subtask_info = self._subtask_executions[pred_subtask_id]
        succ_ids = set()
        for result_chunk in subtask_info.subtask.chunk_graph.result_chunks:
            try:
                succ_ids.update(self._pred_key_mapping_dag.successors(result_chunk.key))
            except KeyError:
                pass

        enqueue_tasks = []
        for succ_subtask_id in succ_ids:
            try:
                succ_subtask_info = self._subtask_caches[succ_subtask_id]
            except KeyError:
                continue
            enqueue_task = asyncio.create_task(
                self._forward_subtask_info(succ_subtask_info)
            )
            enqueue_tasks.append(enqueue_task)
            succ_subtask_info.aio_tasks.append(enqueue_task)

        if enqueue_tasks:
            yield asyncio.wait(enqueue_tasks)

        await self._execution_queue_ref.release_slot(pred_subtask_id, errors="ignore")

    async def _forward_subtask_info(self, subtask_info: SubtaskExecutionInfo):
        self._subtask_executions[subtask_info.subtask.subtask_id] = subtask_info
        subtask_id = subtask_info.subtask.subtask_id
        try:
            await self._subtask_preparer.run(subtask_info, fail_fast=True)
            await self._execution_queue_ref.put(
                subtask_id, subtask_info.band_name, subtask_info.priority
            )
            self.uncache_subtasks([subtask_id])
        except PrepareFastFailed:
            self._subtask_executions.pop(subtask_id, None)
