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
from numbers import Integral
from weakref import WeakKeyDictionary

from ...core import Tileable, TileableGraph, TileableGraphBuilder
from ...core.session import AbstractSession, register_session_cls, \
    ExecutionInfo as AbstractExectionInfo
from ...services.meta import MetaAPI
from ...services.session import SessionAPI
from ...services.storage import StorageAPI
from ...services.task import TaskAPI, TaskResult
from ...utils import implements, merge_chunks


class ExectionInfo(AbstractExectionInfo):
    def __init__(self,
                 task_id: str,
                 task_api: TaskAPI,
                 aio_task: asyncio.Task):
        super().__init__(aio_task)
        self._task_api = task_api
        self._task_id = task_id

    def progress(self) -> float:
        return 1.0 if self.done() else 0.0


@register_session_cls
class Session(AbstractSession):
    name = 'oscar'

    def __init__(self,
                 address: str,
                 session_id: str,
                 session_api: SessionAPI,
                 meta_api: MetaAPI,
                 task_api: TaskAPI):
        super().__init__(address, session_id)
        self._session_api = session_api
        self._task_api = task_api
        self._meta_api = meta_api

        self._tileable_to_fetch = WeakKeyDictionary()

    @classmethod
    @implements(AbstractSession.init)
    async def init(cls,
                   address: str,
                   session_id: str,
                   **kwargs) -> "Session":
        session_api = await SessionAPI.create(address)
        # create new session
        session_address = await session_api.create_session(session_id)
        meta_api = await MetaAPI.create(session_id, session_address)
        task_api = await TaskAPI.create(session_id, session_address)

        if kwargs:  # pragma: no cover
            unexpected_keys = ', '.join(list(kwargs.keys()))
            raise TypeError(f'Oscar session got unexpected '
                            f'arguments: {unexpected_keys}')

        return Session(address, session_id,
                       session_api, meta_api, task_api)

    async def _run_in_background(self,
                                 tileables: list,
                                 task_id: str):
        # wait for task to finish
        task_result: TaskResult = await self._task_api.wait_task(task_id)
        if task_result.error:
            raise task_result.error.with_traceback(task_result.traceback)
        fetch_tileables = await self._task_api.get_fetch_tileable(task_id)
        assert len(tileables) == len(fetch_tileables)
        for tieable, fetch_tileable in zip(tileables, fetch_tileables):
            self._tileable_to_fetch[tieable] = fetch_tileable

    async def execute(self,
                      *tileables,
                      **kwargs) -> ExectionInfo:
        fuse_enabled: bool = kwargs.pop('fuse_enabled', False)
        task_name: str = kwargs.pop('task_name', None)
        if kwargs:  # pragma: no cover
            raise TypeError(f'run got unexpected key arguments {list(kwargs)!r}')

        tileables = [tileable.data if hasattr(tileable, 'data') else tileable
                     for tileable in tileables]

        # build tileable graph
        tileable_graph = TileableGraph(tileables)
        next(TileableGraphBuilder(tileable_graph).build())

        # submit task
        task_id = await self._task_api.submit_tileable_graph(
            tileable_graph, task_name=task_name, fuse_enabled=fuse_enabled)

        # create asyncio.Task
        future = asyncio.create_task(
            self._run_in_background(tileables, task_id))
        return ExectionInfo(task_id, self._task_api, future)

    def _get_to_fetch_tileable(self, tileable: Tileable):
        from ...tensor.indexing import TensorIndex
        from ...dataframe.indexing.iloc import \
            DataFrameIlocGetItem, SeriesIlocGetItem

        slice_op_types = \
            TensorIndex, DataFrameIlocGetItem, SeriesIlocGetItem

        if hasattr(tileable, 'data'):
            tileable = tileable.data

        indexes = None
        while tileable not in self._tileable_to_fetch:
            # if tileable's op is slice, try to check input
            if isinstance(tileable.op, slice_op_types):
                indexes = tileable.op.indexes
                tileable = tileable.inputs[0]
                if not all(isinstance(index, (slice, Integral))
                           for index in indexes):
                    raise ValueError('Only support fetch data slices')
            else:
                raise ValueError(f'Cannot fetch unexecuted '
                                 f'tileable: {tileable}')

        return self._tileable_to_fetch[tileable], indexes

    async def fetch(self, *tileables):
        data = []
        for tileable in tileables:
            fetch_tileable, indexes = self._get_to_fetch_tileable(tileable)
            # TODO: support fetch slices
            assert indexes is None
            index_to_data = []
            for chunk in fetch_tileable.chunks:
                # TODO: use batch API to fetch data
                band = (await self._meta_api.get_chunk_meta(
                    chunk.key, fields=['bands']))['bands'][0]
                storage_api = await StorageAPI.create(
                    self._session_id, band[0])
                index_to_data.append(
                    (chunk.index, await storage_api.get(chunk.key)))

            data.append(merge_chunks(index_to_data))

        return data

    async def destroy(self):
        await self._session_api.delete_session(self._session_id)
