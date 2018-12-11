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

import os

from mars.config import options
from mars.compat import unittest
from mars.utils import classproperty


class WorkerCase(unittest.TestCase):
    plasma_storage_size = 1024 * 1024 * 10

    @classproperty
    def spill_dir(cls):
        import tempfile
        return os.path.join(tempfile.gettempdir(), 'mars_spill_%d_%d' % (os.getpid(), id(cls)))

    @classmethod
    def setUpClass(cls):
        import pyarrow.plasma as plasma
        from mars import kvstore

        cls._plasma_store = plasma.start_plasma_store(cls.plasma_storage_size)
        cls.plasma_socket = options.worker.plasma_socket = cls._plasma_store.__enter__()[0]

        options.worker.spill_directory = cls.spill_dir

        cls._plasma_client = plasma.connect(options.worker.plasma_socket, '', 0)
        cls._kv_store = kvstore.get(options.kv_store)

    @classmethod
    def tearDownClass(cls):
        import shutil
        cls._plasma_client.disconnect()
        cls._plasma_store.__exit__(None, None, None)
        if not isinstance(options.worker.spill_directory, list):
            options.worker.spill_directory = options.worker.spill_directory.split(os.path.pathsep)
        for p in options.worker.spill_directory:
            if os.path.exists(p):
                shutil.rmtree(p)
        if os.path.exists(cls.plasma_socket):
            os.unlink(cls.plasma_socket)
