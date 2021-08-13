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

import time
from io import BytesIO

import mock

import mars.dataframe as md
from mars.lib.filesystem import oss
from mars.lib.filesystem.oss import build_oss_path


class oss_obj_info:
	def __init__(self, name):
		self.key = name


class object_meta:
	def __init__(self, key, obj_dict):
		self.headers = {}
		# Use the current time as "Last-Modified" in the test.
		self.headers["Last-Modified"] = int(time.time())
		self.headers["Content-Length"] = len(obj_dict[key].encode('utf8'))


class mock_object:
	def __init__(self, obj_dict, key, byte_range):
		self._stream = BytesIO(obj_dict[key].encode('utf8'))
		self._byte_range = byte_range
	
	def read(self):
		self._stream.seek(self._byte_range[0])
		if self._byte_range[1] is None:
			return self._stream.read()
		else:
			size = self._byte_range[1] - self._byte_range[0] + 1
			return self._stream.read(size)


class side_effect_Bucket:
	def __init__(self, *args, **kwargs):
		self.obj_dict = {'file.csv': 'id1,id2,id3\n1,2,3\n', 'dir/file1.csv': '2', 'dir/file2.csv': '3'}
	
	def get_object_meta(self, key):
		return object_meta(key, self.obj_dict)
	
	def object_exists(self, key):
		return key in self.obj_dict.keys()
	
	def get_object(self, key, byte_range):
		return mock_object(self.obj_dict, key, byte_range)


class side_effect_ObjIter:
	def __init__(self, *args, **kwargs):
		self.bucket = args[0]
		self.prefix = kwargs['prefix']
	
	def __iter__(self):
		for name, content in self.bucket.obj_dict.items():
			if name.startswith(self.prefix):
				yield oss_obj_info(name)


@mock.patch('oss2.Bucket', side_effect=side_effect_Bucket)
@mock.patch('oss2.ObjectIteratorV2', side_effect=side_effect_ObjIter)
def test_oss_filesystem(fake_obj_iter, fake_oss_bucket):
	
	from mars.deploy.oscar.tests.session import new_test_session
	
	sess = new_test_session(address='test://127.0.0.1',
	                        init_local=True,
	                        default=True)
	
	access_key_id = 'your_access_key_id'
	access_key_secret = 'your_access_key_secret'
	end_point = 'your_endpoint'
	
	file_path = f"oss://bucket/file.csv"
	dir_path = f"oss://bucket/dir/"
	fake_file_path = build_oss_path(file_path, access_key_id, access_key_secret, end_point)
	fake_dir_path = build_oss_path(dir_path, access_key_id, access_key_secret, end_point)
	
	fs = oss.OSSFileSystem.get_instance()
	
	assert fs.ls(fake_dir_path) == ['oss://bucket/dir/file1.csv', 'oss://bucket/dir/file2.csv']
	assert fs.isdir(fake_dir_path)
	assert fs.isfile(fake_file_path)
	assert fs.stat(fake_file_path)["type"] == 'file'
	assert fs.stat(fake_dir_path)["type"] == 'directory'
	assert fs.glob(fake_dir_path) == [fake_dir_path]
	with fs.open(fake_file_path) as f:
		assert f.readline() == b'id1,id2,id3\n'
		assert f.readline() == b'1,2,3\n'
	df = md.read_csv(fake_file_path).execute()
	assert df.shape == (1, 3)

