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

from mars.tests.core import mock

import mars.dataframe as md
from mars.lib.filesystem import oss
from mars.lib.filesystem.oss import build_oss_path

import pytest


class OSSObjInfo:
	def __init__(self, name, content):
		self.key = name
		# Use the current time as "Last-Modified" in the test.
		self.last_modified = int(time.time())
		self.size = len(content.encode('utf8'))
		

class ObjectMeta:
	def __init__(self, key, obj_dict):
		self.headers = {}
		self.headers["Last-Modified"] = int(time.time())
		self.headers["Content-Length"] = len(obj_dict[key].encode('utf8'))


class MockObject:
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


class SideEffectBucket:
	def __init__(self, *args, **kwargs):
		self.obj_dict = {'file.csv': 'id1,id2,id3\n1,2,3\n', 'dir/file1.csv': '2', 'dir/file2.csv': '3'}
	
	def get_object_meta(self, key):
		return ObjectMeta(key, self.obj_dict)
	
	def object_exists(self, key):
		return key in self.obj_dict.keys()
	
	def get_object(self, key, byte_range):
		return MockObject(self.obj_dict, key, byte_range)


class SideEffectObjIter:
	def __init__(self, *args, **kwargs):
		self.bucket = args[0]
		self.prefix = kwargs['prefix']
	
	def __iter__(self):
		for name, content in self.bucket.obj_dict.items():
			if name.startswith(self.prefix):
				yield OSSObjInfo(name, content)


@mock.patch('oss2.Bucket', side_effect=SideEffectBucket)
@mock.patch('oss2.ObjectIteratorV2', side_effect=SideEffectObjIter)
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
	other_scheme_path = f"scheme://netloc/path"
	not_exist_file_path = f"oss://bucket/not_exist.csv"
	
	fake_file_path = build_oss_path(file_path, access_key_id, access_key_secret, end_point)
	fake_dir_path = build_oss_path(dir_path, access_key_id, access_key_secret, end_point)
	fake_other_scheme_path = build_oss_path(other_scheme_path, access_key_id, access_key_secret, end_point)
	fake_not_exist_file_path = build_oss_path(not_exist_file_path, access_key_id, access_key_secret, end_point)
	fs = oss.OSSFileSystem.get_instance()
	
	assert fs.ls(fake_dir_path) == ['oss://bucket/dir/file1.csv', 'oss://bucket/dir/file2.csv']
	assert not fs.isfile(fake_dir_path)
	assert fs.isdir(fake_dir_path)
	assert not fs.isdir(fake_file_path)
	assert fs.isfile(fake_file_path)
	assert fs.exists(fake_file_path)
	assert not fs.exists(fake_not_exist_file_path)
	assert fs.stat(fake_file_path)["type"] == 'file'
	assert fs.stat(fake_dir_path)["type"] == 'directory'
	assert fs.glob(fake_dir_path) == [fake_dir_path]
	
	with pytest.raises(ValueError) as e:
		fs.exists(fake_other_scheme_path)
	msg1 = e.value.args[0]
	assert msg1 == f"Except scheme oss, but got scheme: schemein path: {fake_other_scheme_path}"
	
	with pytest.raises(RuntimeError) as e:
		fs.exists(file_path)
	msg2 = e.value.args[0]
	assert msg2 == "Please use build_oss_path to add OSS info"
	
	# Two files in fake_dir_path.
	assert len(fs.glob(fake_dir_path+'*', recursive=True)) == 2
	with fs.open(fake_file_path) as f:
		assert f.readline() == b'id1,id2,id3\n'
		assert f.readline() == b'1,2,3\n'
	df = md.read_csv(fake_file_path).execute()
	assert df.shape == (1, 3)
