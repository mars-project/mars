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

import os
import shutil
import tempfile

from mars.tests.core import TestBase
from mars.filesystem import _parse_from_path, LocalFileSystem, glob


class Test(TestBase):

    def testPathParser(self):
        path = 'hdfs://user:password@localhost:8080/test'
        parsed_result = _parse_from_path(path)
        self.assertEqual(parsed_result['host'], 'localhost')
        self.assertEqual(parsed_result['port'], 8080)
        self.assertEqual(parsed_result['user'], 'user')
        self.assertEqual(parsed_result['password'], 'password')

    def testLocalFileSystem(self):
        local_fs1 = LocalFileSystem.get_instance()
        local_fs2 = LocalFileSystem.get_instance()
        self.assertIs(local_fs1, local_fs2)

        tempdir = tempfile.mkdtemp()
        file_path = os.path.join(tempdir, 'test')
        try:
            with open(file_path, 'wb') as f:
                f.write(b'text for test')
            self.assertEqual(local_fs1.stat(tempdir)['type'], 'directory')
            self.assertEqual(local_fs1.stat(file_path)['type'], 'file')
            self.assertEqual(len(glob(tempdir + '*')), 1)
        finally:
            shutil.rmtree(tempdir)
