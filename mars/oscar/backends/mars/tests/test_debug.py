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
import logging
import os
import sys
from contextlib import contextmanager
from io import StringIO

import pytest

import mars.oscar as mo
from mars.oscar.debug import reload_debug_opts_from_env, get_debug_options


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


@pytest.fixture
async def actor_pool():
    start_method = os.environ.get('POOL_START_METHOD', 'forkserver') \
        if sys.platform != 'win32' else None
    pool = await mo.create_actor_pool('127.0.0.1', n_process=0,
                                      subprocess_start_method=start_method)
    await pool.start()
    yield pool
    await pool.stop()


@pytest.fixture
async def debug_logger():
    log_file = StringIO()
    logger = logging.getLogger('mars.oscar.debug')

    log_handler = logging.StreamHandler(log_file)
    log_handler.setLevel(logging.DEBUG)
    logger.addHandler(log_handler)

    try:
        mo.set_debug_options(mo.DebugOptions(
            actor_call_timeout=1,
            log_unhandled_errors=True,
        ))
        yield log_file
    finally:
        mo.set_debug_options(None)
        logger.removeHandler(log_handler)


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
    debug_ref = await mo.create_actor(DebugActor, uid=DebugActor.default_uid(),
                                      address=actor_pool.external_address)

    with cut_file_log(debug_logger) as log_file:
        await debug_ref.wait(0.2)
    assert log_file.getvalue() == ''

    with cut_file_log(debug_logger) as log_file:
        await debug_ref.wait(1.2)
    assert DebugActor.default_uid() in log_file.getvalue()

    with pytest.raises(ValueError), \
            cut_file_log(debug_logger) as log_file:
        await debug_ref.raise_error(ValueError)
    assert 'ValueError' in log_file.getvalue()


def test_environ():
    os.environ['DEBUG_OSCAR'] = '1'
    try:
        reload_debug_opts_from_env()
        assert get_debug_options() is not None
    finally:
        os.environ.pop('DEBUG_OSCAR')
        reload_debug_opts_from_env()
        assert get_debug_options() is None
