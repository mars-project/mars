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
import copy
import itertools
import logging
import json
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from numbers import Integral
from urllib.parse import urlparse
from weakref import WeakKeyDictionary, WeakSet
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ... import oscar as mo
from ...utils import Timer, build_fetch, copy_tileables
from ...core import ChunkType, TileableType, enter_mode, TileableGraph
from ...core.operand import Fetch
from ...lib.aio import alru_cache
from ...metrics import Metrics
from ...services.cluster import AbstractClusterAPI, ClusterAPI
from ...services.lifecycle import AbstractLifecycleAPI, LifecycleAPI
from ...services.meta import MetaAPI, AbstractMetaAPI
from ...services.session import AbstractSessionAPI, SessionAPI
from ...services.mutable import MutableAPI
from ...services.storage import StorageAPI
from ...services.task import AbstractTaskAPI, TaskAPI, TaskResult
from ...services.task.execution.api import Fetcher
from ...services.web import OscarWebAPI
from ...session import (
    AbstractAsyncSession,
    AbstractSession,
    IsolatedAsyncSession,
    ExecutionInfo,
    Progress,
    Profiling,
)
from ...typing import ClientType, BandType
from ...utils import (
    implements,
    merge_chunks,
    merged_chunk_as_tileable_type,
    register_asyncio_task_timeout_detector,
)

logger = logging.getLogger(__name__)


@dataclass
class ChunkFetchInfo:
    tileable: TileableType
    chunk: ChunkType
    indexes: List[Union[int, slice]]
    data: Any = None


_submitted_tileables = WeakSet()


@enter_mode(build=True, kernel=True)
def gen_submit_tileable_graph(
    session: AbstractSession,
    result_tileables: List[TileableType],
    warn_duplicated_execution: bool = False,
) -> Tuple[TileableGraph, List[TileableType]]:
    tileable_to_copied = dict()
    indexer = itertools.count()
    result_to_index = {t: i for t, i in zip(result_tileables, indexer)}
    result = list()
    to_execute_tileables = list()
    graph = TileableGraph(result)

    q = list(result_tileables)
    while q:
        tileable = q.pop()
        if tileable in tileable_to_copied:
            continue
        if tileable.cache and tileable not in result_to_index:
            result_to_index[tileable] = next(indexer)
        outputs = tileable.op.outputs
        inputs = tileable.inputs if session not in tileable._executed_sessions else []
        new_inputs = []
        all_inputs_processed = True
        for inp in inputs:
            if inp in tileable_to_copied:
                new_inputs.append(tileable_to_copied[inp])
            elif session in inp._executed_sessions:
                # executed, gen fetch
                fetch_input = build_fetch(inp).data
                tileable_to_copied[inp] = fetch_input
                graph.add_node(fetch_input)
                new_inputs.append(fetch_input)
            else:
                # some input not processed before
                all_inputs_processed = False
                # put back tileable
                q.append(tileable)
                q.append(inp)
                break
        if all_inputs_processed:
            if isinstance(tileable.op, Fetch):
                new_outputs = [tileable]
            elif session in tileable._executed_sessions:
                new_outputs = []
                for out in outputs:
                    fetch_out = tileable_to_copied.get(out, build_fetch(out).data)
                    new_outputs.append(fetch_out)
            else:
                new_outputs = [
                    t.data for t in copy_tileables(outputs, inputs=new_inputs)
                ]
            for out, new_out in zip(outputs, new_outputs):
                tileable_to_copied[out] = new_out
                graph.add_node(new_out)
                for new_inp in new_inputs:
                    graph.add_edge(new_inp, new_out)

    # process results
    result.extend([None] * len(result_to_index))
    for t, i in result_to_index.items():
        result[i] = tileable_to_copied[t]
        to_execute_tileables.append(t)

    if warn_duplicated_execution:
        for n, c in tileable_to_copied.items():
            if not isinstance(c.op, Fetch) and n in _submitted_tileables:
                warnings.warn(
                    f"Tileable {repr(n)} has been submitted before", RuntimeWarning
                )
        # add all nodes into submitted tileables
        _submitted_tileables.update(
            n for n, c in tileable_to_copied.items() if not isinstance(c.op, Fetch)
        )

    return graph, to_execute_tileables


class _IsolatedSession(IsolatedAsyncSession):
    schemes = [None, "ray"]

    def __init__(
        self,
        address: str,
        session_id: str,
        backend: str,
        session_api: AbstractSessionAPI,
        meta_api: AbstractMetaAPI,
        lifecycle_api: AbstractLifecycleAPI,
        task_api: AbstractTaskAPI,
        mutable_api: MutableAPI,
        cluster_api: AbstractClusterAPI,
        web_api: Optional[OscarWebAPI],
        client: ClientType = None,
        timeout: float = None,
        request_rewriter: Callable = None,
    ):
        super().__init__(address, session_id)
        self._backend = backend
        self._session_api = session_api
        self._task_api = task_api
        self._meta_api = meta_api
        self._lifecycle_api = lifecycle_api
        self._mutable_api = mutable_api
        self._cluster_api = cluster_api
        self._web_api = web_api
        self.client = client
        self.timeout = timeout
        self._request_rewriter = request_rewriter

        self._tileable_to_fetch = WeakKeyDictionary()
        self._asyncio_task_timeout_detector_task = (
            register_asyncio_task_timeout_detector()
        )

        # add metrics
        self._tileable_graph_gen_time = Metrics.gauge(
            "mars.tileable_graph_gen_time_secs",
            "Time consuming in seconds to generate a tileable graph",
            ("address", "session_id"),
        )

    @classmethod
    async def _init(
        cls,
        address: str,
        session_id: str,
        backend: str,
        new: bool = True,
        timeout: float = None,
        **kwargs,
    ):
        if kwargs:  # pragma: no cover
            unexpected_keys = ", ".join(list(kwargs.keys()))
            raise TypeError(
                f"Oscar session got unexpected arguments: {unexpected_keys}"
            )

        session_api = await SessionAPI.create(address)
        if new:
            # create new session
            session_address = await session_api.create_session(session_id)
        else:
            session_address = await session_api.get_session_address(session_id)
        lifecycle_api = await LifecycleAPI.create(session_id, session_address)
        meta_api = await MetaAPI.create(session_id, session_address)
        task_api = await TaskAPI.create(session_id, session_address)
        mutable_api = await MutableAPI.create(session_id, session_address)
        cluster_api = await ClusterAPI.create(session_address)
        try:
            web_api = await OscarWebAPI.create(session_address)
        except mo.ActorNotExist:
            web_api = None
        return cls(
            address,
            session_id,
            backend,
            session_api,
            meta_api,
            lifecycle_api,
            task_api,
            mutable_api,
            cluster_api,
            web_api,
            timeout=timeout,
        )

    @classmethod
    @implements(AbstractAsyncSession.init)
    async def init(
        cls,
        address: str,
        session_id: str,
        backend: str,
        new: bool = True,
        timeout: float = None,
        **kwargs,
    ) -> "AbstractAsyncSession":
        init_local = kwargs.pop("init_local", False)
        if init_local:
            from .local import new_cluster_in_isolation

            return (
                await new_cluster_in_isolation(
                    address, timeout=timeout, backend=backend, **kwargs
                )
            ).session

        return await cls._init(
            address, session_id, backend, new=new, timeout=timeout, **kwargs
        )

    async def _update_progress(self, task_id: str, progress: Progress):
        zero_acc_time = 0
        delay = 0.5
        while True:
            try:
                last_progress_value = progress.value
                progress.value = await self._task_api.get_task_progress(task_id)
                if abs(progress.value - last_progress_value) < 1e-4:
                    # if percentage does not change, we add delay time by 0.5 seconds every time
                    zero_acc_time = min(5, zero_acc_time + 0.5)
                    delay = zero_acc_time
                else:
                    # percentage changes, we use percentage speed to calc progress time
                    zero_acc_time = 0
                    speed = abs(progress.value - last_progress_value) / delay
                    # one percent for one second
                    delay = 0.01 / speed
                delay = max(0.5, min(delay, 5.0))
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                break

    async def _run_in_background(
        self,
        tileables: list,
        task_id: str,
        progress: Progress,
        profiling: Profiling,
    ):
        with enter_mode(build=True, kernel=True):
            # wait for task to finish
            cancelled = False
            progress_task = asyncio.create_task(
                self._update_progress(task_id, progress)
            )
            start_time = time.time()
            task_result: Optional[TaskResult] = None
            try:
                if self.timeout is None:
                    check_interval = 30
                else:
                    elapsed = time.time() - start_time
                    check_interval = min(self.timeout - elapsed, 30)

                while True:
                    task_result = await self._task_api.wait_task(
                        task_id, timeout=check_interval
                    )
                    if task_result is not None:
                        break
                    elif (
                        self.timeout is not None
                        and time.time() - start_time > self.timeout
                    ):
                        raise TimeoutError(
                            f"Task({task_id}) running time > {self.timeout}"
                        )
            except asyncio.CancelledError:
                # cancelled
                cancelled = True
                await self._task_api.cancel_task(task_id)
            finally:
                progress_task.cancel()
                if task_result is not None:
                    progress.value = 1.0
                else:
                    # not finished, set progress
                    progress.value = await self._task_api.get_task_progress(task_id)
            if task_result is not None:
                profiling.result = task_result.profiling
                if task_result.profiling:
                    logger.warning(
                        "Profile task %s execution result:\n%s",
                        task_id,
                        json.dumps(task_result.profiling, indent=4),
                    )
                if task_result.error:
                    raise task_result.error.with_traceback(task_result.traceback)
            if cancelled:
                return
            fetch_tileables = await self._task_api.get_fetch_tileables(task_id)
            assert len(tileables) == len(fetch_tileables)

            for tileable, fetch_tileable in zip(tileables, fetch_tileables):
                self._tileable_to_fetch[tileable] = fetch_tileable
                # update meta, e.g. unknown shape
                tileable.params = fetch_tileable.params

    async def execute(self, *tileables, **kwargs) -> ExecutionInfo:
        if self._closed:
            raise RuntimeError("Session closed already")
        kwargs = copy.deepcopy(kwargs)
        fuse_enabled: bool = kwargs.pop("fuse_enabled", None)
        extra_config: dict = kwargs.pop("extra_config", None)
        warn_duplicated_execution: bool = kwargs.pop("warn_duplicated_execution", False)
        if kwargs:  # pragma: no cover
            raise TypeError(f"run got unexpected key arguments {list(kwargs)!r}")

        tileables = [
            tileable.data if hasattr(tileable, "data") else tileable
            for tileable in tileables
        ]

        # build tileable graph
        with Timer() as timer:
            tileable_graph, to_execute_tileables = gen_submit_tileable_graph(
                self, tileables, warn_duplicated_execution=warn_duplicated_execution
            )

        logger.info(
            "Time consuming to generate a tileable graph is %ss with address %s, session id %s",
            timer.duration,
            self.address,
            self._session_id,
        )
        self._tileable_graph_gen_time.record(
            timer.duration, {"address": self.address, "session_id": self._session_id}
        )

        # submit task
        task_id = await self._task_api.submit_tileable_graph(
            tileable_graph,
            fuse_enabled=fuse_enabled,
            extra_config=extra_config,
        )

        progress = Progress()
        profiling = Profiling()
        # create asyncio.Task
        aio_task = asyncio.create_task(
            self._run_in_background(to_execute_tileables, task_id, progress, profiling)
        )
        return ExecutionInfo(
            aio_task,
            progress,
            profiling,
            asyncio.get_running_loop(),
            to_execute_tileables,
        )

    def _get_to_fetch_tileable(
        self, tileable: TileableType
    ) -> Tuple[TileableType, List[Union[slice, Integral]]]:
        from ...tensor.indexing import TensorIndex
        from ...dataframe.indexing.iloc import DataFrameIlocGetItem, SeriesIlocGetItem

        slice_op_types = TensorIndex, DataFrameIlocGetItem, SeriesIlocGetItem

        if hasattr(tileable, "data"):
            tileable = tileable.data

        indexes = None
        while tileable not in self._tileable_to_fetch:
            # if tileable's op is slice, try to check input
            if isinstance(tileable.op, slice_op_types):
                indexes = tileable.op.indexes
                tileable = tileable.inputs[0]
                if not all(isinstance(index, (slice, Integral)) for index in indexes):
                    raise ValueError("Only support fetch data slices")
            elif isinstance(tileable.op, Fetch):
                break
            else:
                raise ValueError(f"Cannot fetch unexecuted tileable: {tileable!r}")

        if isinstance(tileable.op, Fetch):
            return tileable, indexes
        else:
            return self._tileable_to_fetch[tileable], indexes

    @classmethod
    def _calc_chunk_indexes(
        cls, fetch_tileable: TileableType, indexes: List[Union[slice, Integral]]
    ) -> Dict[ChunkType, List[Union[slice, int]]]:
        from ...tensor.utils import slice_split

        axis_to_slices = {
            axis: slice_split(ind, fetch_tileable.nsplits[axis])
            for axis, ind in enumerate(indexes)
        }
        result = dict()
        for chunk_index in itertools.product(
            *[v.keys() for v in axis_to_slices.values()]
        ):
            # slice_obj: use tuple, since numpy complains
            #
            # FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use
            # `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array
            # index, `arr[np.array(seq)]`, which will result either in an error or a different result.
            slice_obj = [
                axis_to_slices[axis][chunk_idx]
                for axis, chunk_idx in enumerate(chunk_index)
            ]
            chunk = fetch_tileable.cix[chunk_index]
            result[chunk] = slice_obj
        return result

    def _process_result(self, tileable, result):  # pylint: disable=no-self-use
        return result

    @alru_cache(cache_exceptions=False)
    async def _get_storage_api(self, band: BandType):
        if urlparse(self.address).scheme == "http":
            from ...services.storage.api import WebStorageAPI

            storage_api = WebStorageAPI(
                self._session_id, self.address, band[1], self._request_rewriter
            )
        else:
            storage_api = await StorageAPI.create(self._session_id, band[0], band[1])
        return storage_api

    async def fetch(self, *tileables, **kwargs) -> list:
        if kwargs:  # pragma: no cover
            unexpected_keys = ", ".join(list(kwargs.keys()))
            raise TypeError(f"`fetch` got unexpected arguments: {unexpected_keys}")

        fetcher = Fetcher.create(self._backend, get_storage_api=self._get_storage_api)

        with enter_mode(build=True):
            chunks = []
            get_chunk_metas = []
            fetch_infos_list = []
            for tileable in tileables:
                fetch_tileable, indexes = self._get_to_fetch_tileable(tileable)
                chunk_to_slice = None
                if indexes is not None:
                    chunk_to_slice = self._calc_chunk_indexes(fetch_tileable, indexes)
                fetch_infos = []
                for chunk in fetch_tileable.chunks:
                    if indexes and chunk not in chunk_to_slice:
                        continue
                    chunks.append(chunk)
                    get_chunk_metas.append(
                        self._meta_api.get_chunk_meta.delay(
                            chunk.key,
                            fields=fetcher.required_meta_keys,
                        )
                    )
                    indexes = (
                        chunk_to_slice[chunk] if chunk_to_slice is not None else None
                    )
                    fetch_infos.append(
                        ChunkFetchInfo(tileable=tileable, chunk=chunk, indexes=indexes)
                    )
                fetch_infos_list.append(fetch_infos)

            chunk_metas = await self._meta_api.get_chunk_meta.batch(*get_chunk_metas)
            for chunk, meta, fetch_info in zip(
                chunks, chunk_metas, itertools.chain(*fetch_infos_list)
            ):
                await fetcher.append(chunk.key, meta, fetch_info.indexes)
            fetched_data = await fetcher.get()
            for fetch_info, data in zip(
                itertools.chain(*fetch_infos_list), fetched_data
            ):
                fetch_info.data = data

            result = []
            for tileable, fetch_infos in zip(tileables, fetch_infos_list):
                index_to_data = [
                    (fetch_info.chunk.index, fetch_info.data)
                    for fetch_info in fetch_infos
                ]
                merged = merge_chunks(index_to_data)
                merged = merged_chunk_as_tileable_type(merged, tileable)
                result.append(self._process_result(tileable, merged))
            return result

    async def fetch_infos(self, *tileables, fields, **kwargs) -> list:
        available_fields = {
            "data_key",
            "object_id",
            "object_refs",
            "level",
            "memory_size",
            "store_size",
            "bands",
        }
        if fields is None:
            fields = available_fields
        else:
            for field_name in fields:
                if field_name not in available_fields:  # pragma: no cover
                    raise TypeError(
                        f"`fetch_infos` got unexpected field name: {field_name}"
                    )
            fields = set(fields)

        if kwargs:  # pragma: no cover
            unexpected_keys = ", ".join(list(kwargs.keys()))
            raise TypeError(f"`fetch` got unexpected arguments: {unexpected_keys}")
        # following fields needs to access storage API to get the meta.
        _need_query_storage_fields = {"level", "memory_size", "store_size"}
        _need_query_storage = bool(_need_query_storage_fields & fields)
        with enter_mode(build=True):
            chunk_to_bands, fetch_infos_list, result = await self._query_meta_service(
                tileables, fields, _need_query_storage
            )
            if not _need_query_storage:
                assert result is not None
                return result
            storage_api_to_gets = defaultdict(list)
            storage_api_to_fetch_infos = defaultdict(list)
            for fetch_info in itertools.chain(*fetch_infos_list):
                chunk = fetch_info.chunk
                bands = chunk_to_bands[chunk]
                storage_api = await self._get_storage_api(bands[0])
                storage_api_to_gets[storage_api].append(
                    storage_api.get_infos.delay(chunk.key)
                )
                storage_api_to_fetch_infos[storage_api].append(fetch_info)
            for storage_api in storage_api_to_gets:
                fetched_data = await storage_api.get_infos.batch(
                    *storage_api_to_gets[storage_api]
                )
                infos = storage_api_to_fetch_infos[storage_api]
                for info, data in zip(infos, fetched_data):
                    info.data = data

            result = []
            for fetch_infos in fetch_infos_list:
                fetched = defaultdict(list)
                for fetch_info in fetch_infos:
                    bands = chunk_to_bands[fetch_info.chunk]
                    # Currently there's only one item in the returned List from storage_api.get_infos()
                    data = fetch_info.data[0]
                    if "data_key" in fields:
                        fetched["data_key"].append(fetch_info.chunk.key)
                    if "object_id" in fields:
                        fetched["object_id"].append(data.object_id)
                    if "level" in fields:
                        fetched["level"].append(data.level)
                    if "memory_size" in fields:
                        fetched["memory_size"].append(data.memory_size)
                    if "store_size" in fields:
                        fetched["store_size"].append(data.store_size)
                    # data.band misses ip info, e.g. 'numa-0'
                    # while band doesn't, e.g. (address0, 'numa-0')
                    if "bands" in fields:
                        fetched["bands"].append(bands)
                result.append(fetched)

            return result

    async def _query_meta_service(self, tileables, fields, query_storage):
        chunks = []
        get_chunk_metas = []
        fetch_infos_list = []
        for tileable in tileables:
            fetch_tileable, _ = self._get_to_fetch_tileable(tileable)
            fetch_infos = []
            for chunk in fetch_tileable.chunks:
                chunks.append(chunk)
                get_chunk_metas.append(
                    self._meta_api.get_chunk_meta.delay(
                        chunk.key,
                        fields=["bands"] if query_storage else fields - {"data_key"},
                    )
                )
                fetch_infos.append(
                    ChunkFetchInfo(tileable=tileable, chunk=chunk, indexes=None)
                )
            fetch_infos_list.append(fetch_infos)
        chunk_metas = await self._meta_api.get_chunk_meta.batch(*get_chunk_metas)
        if not query_storage:
            result = []
            chunk_to_meta = dict(zip(chunks, chunk_metas))
            for fetch_infos in fetch_infos_list:
                fetched = defaultdict(list)
                for fetch_info in fetch_infos:
                    if "data_key" in fields:
                        fetched["data_key"].append(fetch_info.chunk.key)
                    for field in fields - {"data_key"}:
                        fetched[field].append(chunk_to_meta[fetch_info.chunk][field])
                result.append(fetched)
            return {}, fetch_infos_list, result
        chunk_to_bands = {
            chunk: meta["bands"] for chunk, meta in zip(chunks, chunk_metas)
        }
        return chunk_to_bands, fetch_infos_list, None

    async def decref(self, *tileable_keys):
        logger.debug("Decref tileables on client: %s", tileable_keys)
        return await self._lifecycle_api.decref_tileables(list(tileable_keys))

    async def _get_ref_counts(self) -> Dict[str, int]:
        return await self._lifecycle_api.get_all_chunk_ref_counts()

    async def fetch_tileable_op_logs(
        self,
        tileable_op_key: str,
        offsets: Union[Dict[str, List[int]], str, int],
        sizes: Union[Dict[str, List[int]], str, int],
    ) -> Dict:
        return await self._session_api.fetch_tileable_op_logs(
            self.session_id, tileable_op_key, offsets, sizes
        )

    async def get_total_n_cpu(self):
        all_bands = await self._cluster_api.get_all_bands()
        n_cpu = 0
        for band, resource in all_bands.items():
            _, band_name = band
            if band_name.startswith("numa-"):
                n_cpu += resource.num_cpus
        return n_cpu

    async def get_cluster_versions(self) -> List[str]:
        return list(await self._cluster_api.get_mars_versions())

    async def get_web_endpoint(self) -> Optional[str]:
        if self._web_api is None:
            return None
        return await self._web_api.get_web_address()

    async def destroy(self):
        await super().destroy()
        await self._session_api.delete_session(self._session_id)
        self._tileable_to_fetch.clear()
        if self._asyncio_task_timeout_detector_task:  # pragma: no cover
            self._asyncio_task_timeout_detector_task.cancel()

    async def create_remote_object(
        self, session_id: str, name: str, object_cls, *args, **kwargs
    ):
        return await self._session_api.create_remote_object(
            session_id, name, object_cls, *args, **kwargs
        )

    async def get_remote_object(self, session_id: str, name: str):
        return await self._session_api.get_remote_object(session_id, name)

    async def destroy_remote_object(self, session_id: str, name: str):
        return await self._session_api.destroy_remote_object(session_id, name)

    async def create_mutable_tensor(
        self,
        shape: tuple,
        dtype: Union[np.dtype, str],
        name: str = None,
        default_value: Union[int, float] = 0,
        chunk_size: Union[int, Tuple] = None,
    ):
        tensor_info = await self._mutable_api.create_mutable_tensor(
            shape, dtype, name, default_value, chunk_size
        )
        return tensor_info, self._mutable_api

    async def get_mutable_tensor(self, name: str):
        tensor_info = await self._mutable_api.get_mutable_tensor(name)
        return tensor_info, self._mutable_api

    async def stop_server(self):
        if self.client:
            await self.client.stop()


class _IsolatedWebSession(_IsolatedSession):
    schemes = ["http", "https"]

    @classmethod
    async def _init(
        cls,
        address: str,
        session_id: str,
        backend: str,
        new: bool = True,
        timeout: float = None,
        **kwargs,
    ):
        from ...services.session import WebSessionAPI
        from ...services.lifecycle import WebLifecycleAPI
        from ...services.meta import WebMetaAPI
        from ...services.task import WebTaskAPI
        from ...services.mutable import WebMutableAPI
        from ...services.cluster import WebClusterAPI

        request_rewriter = kwargs.pop("request_rewriter", None)

        if kwargs:  # pragma: no cover
            unexpected_keys = ", ".join(list(kwargs.keys()))
            raise TypeError(
                f"Oscar session got unexpected arguments: {unexpected_keys}"
            )

        session_api = WebSessionAPI(address, request_rewriter)
        if new:
            # create new session
            await session_api.create_session(session_id)
        lifecycle_api = WebLifecycleAPI(session_id, address, request_rewriter)
        meta_api = WebMetaAPI(session_id, address, request_rewriter)
        task_api = WebTaskAPI(session_id, address, request_rewriter)
        mutable_api = WebMutableAPI(session_id, address, request_rewriter)
        cluster_api = WebClusterAPI(address, request_rewriter)

        return cls(
            address,
            session_id,
            backend,
            session_api,
            meta_api,
            lifecycle_api,
            task_api,
            mutable_api,
            cluster_api,
            None,
            timeout=timeout,
            request_rewriter=request_rewriter,
        )

    async def get_web_endpoint(self) -> Optional[str]:
        return self.address


def register_session_schemes(overwrite: bool = False):
    _IsolatedSession.register_schemes(overwrite)
    _IsolatedWebSession.register_schemes(overwrite)
