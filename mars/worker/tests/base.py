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

import asyncio.locks
import os
import unittest

from mars import promise
from mars.tests.core import create_actor_pool
from mars.config import options
from mars.utils import classproperty
from mars.worker.utils import WorkerActor, parse_spill_dirs


class WorkerTestActor(WorkerActor):
    def __init__(self):
        super().__init__()
        self.test_obj = None

    def set_test_object(self, test_obj):
        self.test_obj = test_obj

    def get_actor_obj(self):
        return self

    async def run_later(self, fun, *args, **kw):
        delay = kw.get('_delay')
        if not delay:
            kw.pop('_delay', None)
            return await fun(*args, **kw)
        else:
            kw['_tell'] = True
            await self.ref().run_later(fun, *args, **kw)

    def set_result(self, result, accept=True):
        self.test_obj._result_store = (result, accept)
        self.test_obj._result_event.set()


class StorageClientActor(WorkerActor):
    def __getattr__(self, item):
        if item.startswith('_'):
            return object.__getattribute__(self, item)

        async def _wrapped_call(*args, **kwargs):
            callback = kwargs.pop('callback', None)
            ret = getattr(self.storage_client, item)(*args, **kwargs)
            if asyncio.iscoroutine(ret):
                ret = await ret

            if isinstance(ret, promise.Promise):
                if callback:
                    ret.then(lambda *a, **k: self.tell_promise(callback, *a, **k))
            else:
                return ret

        return _wrapped_call


class WorkerCase(unittest.TestCase):
    plasma_storage_size = 1024 * 1024 * 10

    @classproperty
    def spill_dir(cls):
        import tempfile
        if not getattr(cls, '_base_spill_dir', None):
            cls._base_spill_dir = tempfile.mkdtemp('mars_spill_%d_%d' % (os.getpid(), id(cls)))
        return cls._base_spill_dir

    @classmethod
    def setUpClass(cls):
        import pyarrow.plasma as plasma
        from mars import kvstore

        cls._plasma_store = plasma.start_plasma_store(cls.plasma_storage_size)
        cls.plasma_socket = options.worker.plasma_socket = cls._plasma_store.__enter__()[0]

        options.worker.spill_directory = cls.spill_dir

        try:
            cls._plasma_client = plasma.connect(options.worker.plasma_socket)
        except TypeError:
            cls._plasma_client = plasma.connect(options.worker.plasma_socket, '', 0)
        cls._kv_store = kvstore.get(options.kv_store)

    @classmethod
    def tearDownClass(cls):
        cls._plasma_client.disconnect()
        cls._plasma_store.__exit__(None, None, None)

        cls.rm_spill_dirs(cls.spill_dir)

        if os.path.exists(cls.plasma_socket):
            os.unlink(cls.plasma_socket)

    def setUp(self):
        super().setUp()
        self._test_pool = None
        self._test_actor = None
        self._test_actor_ref = None
        self._result_store = None
        self._result_event = None

    def run_actor_test(self, pool):
        this = self
        self._result_event = asyncio.locks.Event()

        class _AsyncContextManager(object):
            async def __aenter__(self):
                this._test_pool = pool
                this._test_actor_ref = await pool.create_actor(WorkerTestActor)
                await this._test_actor_ref.set_test_object(this)
                this._test_actor = await this._test_actor_ref.get_actor_obj()
                return this._test_actor

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                if exc_type is not None:
                    this._result_store = ((exc_type, exc_val, exc_tb), False)
                    this._result_event.set()
                    raise exc_val.with_traceback(exc_tb) from None

        return _AsyncContextManager()

    async def waitp(self, *promises, **kw):
        timeout = kw.pop('timeout', 10)
        if len(promises) > 1:
            p = promise.all_(promises)
        else:
            p = promises[0]
        p.then(lambda *s: self._test_actor_ref.set_result(s, _tell=True),
               lambda *exc: self._test_actor_ref.set_result(exc, accept=False, _tell=True))
        return await self.get_result(timeout)

    async def get_result(self, timeout=None):
        try:
            await asyncio.wait_for(self._result_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError from None
        self._result_event.clear()
        r, accept = self._result_store
        if accept:
            return r
        else:
            raise r[1].with_traceback(r[2]) from None

    @staticmethod
    def rm_spill_dirs(spill_dirs=None):
        import shutil
        spill_dirs = spill_dirs or []
        if not isinstance(spill_dirs, list):
            spill_dirs = [spill_dirs]
        option_dirs = parse_spill_dirs(options.worker.spill_directory or '')
        spill_dirs = list(set(spill_dirs + option_dirs))
        for d in spill_dirs:
            shutil.rmtree(d, ignore_errors=True)

    @staticmethod
    def create_pool(*args, **kwargs):
        from mars.worker import SharedHolderActor
        pool = None

        class _AsyncContextManager(object):
            async def __aenter__(self):
                nonlocal pool
                pool = create_actor_pool(*args, **kwargs)
                await pool.__aenter__()
                return pool

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                shared_ref = pool.actor_ref(SharedHolderActor.default_uid())
                if await pool.has_actor(shared_ref):
                    await shared_ref.destroy()
                await pool.__aexit__(exc_type, exc_val, exc_tb)

        return _AsyncContextManager()

