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
import unittest

from mars.filesystem import _parse_from_path, LocalFileSystem, glob, FSMap


class Test(unittest.TestCase):

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

    def testFSMap(self):
        fs = LocalFileSystem.get_instance()
        with tempfile.TemporaryDirectory() as root:
            fs_map = FSMap(root, fs, check=True)

            path = '/to/path/test_file'
            test_content = b'text for test'
            fs_map[path] = test_content
            self.assertEqual(fs_map[path], test_content)
            self.assertEqual(len(fs_map), 1)
            self.assertIn(path, fs_map)

            path2 = '/to/path2/test_file2'
            fs_map[path2] = test_content
            self.assertEqual(len(fs_map), 2)

            del fs_map[path]
            self.assertEqual(list(fs_map), ['to/path2/test_file2'])

            path3 = '/to2/path3/test_file3'
            fs_map[path3] = test_content
            self.assertEqual(fs_map.pop(path3), test_content)
            self.assertEqual(fs_map.pop(path3, 'fake_content'), 'fake_content')
            with self.assertRaises(KeyError):
                fs_map.pop('not_exist')

            fs_map.clear()
            self.assertEqual(len(fs_map), 0)

            # test root not exist
            with self.assertRaises(ValueError):
                _ = FSMap(root + '/path2', fs, check=True)

            # create root
            fs_map = FSMap(root + '/path2', fs, create=True)
            self.assertEqual(len(fs_map), 0)
