# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

import contextlib
import os
import sys
import unittest

import gevent.event

from mars.config import options
from mars.compat import six
from mars.utils import classproperty
from mars.worker.utils import WorkerActor


class WorkerTestActor(WorkerActor):
    def __init__(self):
        super(WorkerTestActor, self).__init__()
        self.test_obj = None

    def set_test_object(self, test_obj):
        self.test_obj = test_obj

    def run_test(self):
        yield self
        v = yield
        del v

    def run_later(self, fun, *args, **kw):
        delay = kw.get('_delay')
        if not delay:
            kw.pop('_delay', None)
            return fun(*args, **kw)
        else:
            kw['_tell'] = True
            self.ref().run_later(fun, *args, **kw)

    def set_result(self, result, accept=True):
        self.test_obj._result_store = (result, accept)
        self.test_obj._result_event.set()


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

        cls._plasma_client = plasma.connect(options.worker.plasma_socket)
        cls._kv_store = kvstore.get(options.kv_store)

    @classmethod
    def tearDownClass(cls):
        cls._plasma_client.disconnect()
        cls._plasma_store.__exit__(None, None, None)

        cls.rm_spill_dirs(cls.spill_dir)
        cls.rm_spill_dirs(options.worker.spill_directory)

        if os.path.exists(cls.plasma_socket):
            os.unlink(cls.plasma_socket)

    def setUp(self):
        super(WorkerCase, self).setUp()
        self._test_pool = None
        self._test_actor_ref = None
        self._result_store = None
        self._result_event = gevent.event.Event()

    @contextlib.contextmanager
    def run_actor_test(self, pool):
        self._test_pool = pool
        self._test_actor_ref = pool.create_actor(WorkerTestActor)
        self._test_actor_ref.set_test_object(self)
        gen = self._test_actor_ref.run_test()
        try:
            yield next(gen)
        except:
            self._result_store = (sys.exc_info(), False)
            self._result_event.set()
            raise
        finally:
            gen.send(None)

    def get_result(self, timeout=None):
        self._result_event.wait(timeout)
        self._result_event.clear()
        r, accept = self._result_store
        if accept:
            return r
        else:
            six.reraise(*r)

    @staticmethod
    def rm_spill_dirs(spill_dirs):
        import shutil
        spill_dirs = spill_dirs or []
        if not isinstance(spill_dirs, list):
            spill_dirs = [spill_dirs]
        for d in spill_dirs:
            shutil.rmtree(d, ignore_errors=True)
