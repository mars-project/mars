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
import logging
import os
import sys
from contextlib import contextmanager
from io import StringIO
from typing import List

import pytest

from ..... import oscar as mo
from ....debug import reload_debug_opts_from_env, get_debug_options


class DebugActor(mo.Actor):
    def __init__(self):
        self._log_file = None
        self._pos = 0

    @classmethod
    async def wait(cls, delay: float):
        await asyncio.sleep(delay)

    @classmethod
    async def raise_error(cls, exc):
        raise exc

    @classmethod
    async def call_chain(
        cls, chain: List, use_yield: bool = False, use_tell: bool = False
    ):
        if not chain:
            return
        ref_uid, ref_address = chain[0]
        new_ref = await mo.actor_ref(ref_uid, address=ref_address)

        if use_tell:
            call_coro = new_ref.call_chain.tell(chain[1:])
        else:
            call_coro = new_ref.call_chain(chain[1:])

        if use_yield:
            yield call_coro
        else:
            await call_coro

    async def call_self_ref(self):
        await self.ref().wait(1)


@pytest.fixture
async def actor_pool():
    start_method = (
        os.environ.get("POOL_START_METHOD", "forkserver")
        if sys.platform != "win32"
        else None
    )
    pool = await mo.create_actor_pool(
        "127.0.0.1", n_process=0, subprocess_start_method=start_method
    )
    await pool.start()
    yield pool
    await pool.stop()


@pytest.fixture
async def debug_logger():
    log_file = StringIO()
    logger = logging.getLogger("mars.oscar.debug")

    log_handler = logging.StreamHandler(log_file)
    log_handler.setLevel(logging.DEBUG)
    logger.addHandler(log_handler)

    try:
        mo.set_debug_options(
            mo.DebugOptions(
                actor_call_timeout=1,
                log_unhandled_errors=True,
                log_cycle_send=True,
            )
        )
        yield log_file
    finally:
        mo.set_debug_options(None)
        logger.removeHandler(log_handler)
        assert mo.get_debug_options() is None


@contextmanager
def cut_file_log(log_file) -> StringIO:
    dest = StringIO()
    pos = log_file.tell()
    try:
        yield dest
    finally:
        log_file.seek(pos, os.SEEK_SET)
        dest.write(log_file.read())


@pytest.mark.asyncio
async def test_error_logs(actor_pool, debug_logger):
    debug_ref = await mo.create_actor(
        DebugActor, uid=DebugActor.default_uid(), address=actor_pool.external_address
    )

    with cut_file_log(debug_logger) as log_file:
        await debug_ref.wait(0.2)
    assert log_file.getvalue() == ""

    with cut_file_log(debug_logger) as log_file:
        await debug_ref.wait(1.2)
    assert DebugActor.default_uid() in log_file.getvalue()

    with pytest.raises(ValueError), cut_file_log(debug_logger) as log_file:
        await debug_ref.raise_error(ValueError)
    assert "ValueError" in log_file.getvalue()


@pytest.mark.asyncio
async def test_cycle_logs(actor_pool, debug_logger):
    address = actor_pool.external_address
    ref1 = await mo.create_actor(DebugActor, uid="debug_ref1", address=address)
    ref2 = await mo.create_actor(DebugActor, uid="debug_ref2", address=address)

    chain = [(ref2.uid, ref2.address)]

    with cut_file_log(debug_logger) as log_file:
        task = asyncio.create_task(ref1.call_chain(chain))
        await asyncio.wait_for(task, 1)
    assert log_file.getvalue() == ""

    chain = [(ref2.uid, ref2.address), (ref1.uid, ref1.address)]

    # test cycle detection with chain
    with pytest.raises(asyncio.TimeoutError), cut_file_log(debug_logger) as log_file:
        task = asyncio.create_task(ref1.call_chain(chain))
        await asyncio.wait_for(task, 1)
    assert "cycle" in log_file.getvalue()

    # test yield call (should not produce loops)
    with cut_file_log(debug_logger) as log_file:
        task = asyncio.create_task(ref1.call_chain(chain, use_yield=True))
        await asyncio.wait_for(task, 1)
    assert log_file.getvalue() == ""

    # test tell (should not produce loops)
    with cut_file_log(debug_logger) as log_file:
        task = asyncio.create_task(ref1.call_chain(chain, use_tell=True))
        await asyncio.wait_for(task, 1)
    assert log_file.getvalue() == ""

    # test calling actor inside itself
    with pytest.raises(asyncio.TimeoutError), cut_file_log(debug_logger) as log_file:
        task = asyncio.create_task(ref1.call_self_ref())
        await asyncio.wait_for(task, 1)
    assert "cycle" in log_file.getvalue()


def test_environ():
    os.environ["DEBUG_OSCAR"] = "1"
    try:
        reload_debug_opts_from_env()
        assert get_debug_options() is not None
    finally:
        os.environ.pop("DEBUG_OSCAR")
        reload_debug_opts_from_env()
        assert get_debug_options() is None
