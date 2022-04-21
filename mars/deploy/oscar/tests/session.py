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
import os
import uuid
from typing import Union, List, Dict

from ....core import OBJECT_TYPE
from ....deploy.oscar.local import LocalCluster
from ....tests.core import _check_args, ObjectCheckMixin
from ..session import (
    _IsolatedSession,
    AbstractAsyncSession,
    AsyncSession,
    ensure_isolation_created,
    _ensure_sync,
)


CONFIG_FILE = os.path.join(os.path.dirname(__file__), "check_enabled_config.yml")


async def _new_test_cluster_in_isolation(
    address: str = "0.0.0.0",
    n_worker: int = 1,
    n_cpu: Union[int, str] = "auto",
    mem_bytes: Union[int, str] = "auto",
    cuda_devices: Union[List[int], str] = "auto",
    subprocess_start_method: str = None,
    backend: str = None,
    config: Union[str, Dict] = None,
    web: bool = True,
    timeout: float = None,
    n_supervisor_process: int = 0,
) -> AbstractAsyncSession:
    cluster = LocalCluster(
        address,
        n_worker,
        n_cpu,
        mem_bytes,
        cuda_devices,
        subprocess_start_method,
        config,
        web,
        n_supervisor_process,
    )
    await cluster.start()
    return await _new_test_session(
        cluster.external_address,
        backend=backend,
        default=True,
        timeout=timeout,
    )


class CheckedSession(ObjectCheckMixin, _IsolatedSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tileable_checked = dict()

        check_options = dict()
        for key in _check_args:
            check_options[key] = kwargs.get(key, True)
        self._check_options = check_options

    @classmethod
    async def init(
        cls, address: str, session_id: str, **kwargs
    ) -> "AbstractAsyncSession":
        init_local = kwargs.pop("init_local", False)
        if init_local:
            if "n_cpu" not in kwargs:
                # limit to 2 cpu each worker
                kwargs["n_cpu"] = 2 * kwargs.get("n_worker", 1)
            if "config" not in kwargs:
                # enable check for task and subtask processor
                kwargs["config"] = CONFIG_FILE

            return await _new_test_cluster_in_isolation(address, **kwargs)
        session = await super().init(address, session_id, **kwargs)
        return session

    @staticmethod
    def _extract_check_options(extra_config):
        check_options = dict()
        for key in _check_args:
            check_options[key] = extra_config.pop(key, True)
        return check_options

    def _process_result(self, tileable, result):
        if self._check_options.get("check_all", True):
            if (
                not isinstance(tileable, OBJECT_TYPE)
                and tileable.key not in self._tileable_checked
            ):
                self.assert_object_consistent(tileable, result)
        return super()._process_result(tileable, result)

    async def fetch(self, *tileables, **kwargs):
        extra_config = kwargs.pop("extra_config", dict())
        if kwargs:
            unexpected_keys = ", ".join(list(kwargs.keys()))
            raise TypeError(f"`fetch` got unexpected arguments: {unexpected_keys}")

        self._check_options = self._extract_check_options(extra_config)
        results = await super().fetch(*tileables)
        return results


async def _new_test_session(
    address: str, session_id: str = None, default: bool = False, **kwargs
) -> AsyncSession:
    if session_id is None:
        session_id = str(uuid.uuid4())

    session = AsyncSession.from_isolated_session(
        await CheckedSession.init(address, session_id=session_id, **kwargs)
    )
    if default:
        session.as_default()
    return session


def new_test_session(
    address: str = None,
    session_id: str = None,
    default: bool = False,
    backend: str = "mars",
    **kwargs,
):
    isolation = ensure_isolation_created(kwargs)
    if address is None:
        address = "127.0.0.1"
        if "init_local" not in kwargs:
            kwargs["init_local"] = True
    if "web" not in kwargs:
        kwargs["web"] = False
    coro = _new_test_session(address, session_id=session_id, backend=backend, **kwargs)
    session = _ensure_sync(
        asyncio.run_coroutine_threadsafe(coro, isolation.loop).result(120)
    )
    if default:
        session.as_default()
    return session
