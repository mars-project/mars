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

import base64
import json
import os

from ..base import path_type, stringify_path
from ....utils import ModulePlaceholder

try:
    import oss2
except ImportError:
    oss2 = ModulePlaceholder('oss2')

# OSS api time out
_oss_time_out = 10


class OSSFileEntry:
    def __init__(self, path, *, is_dir=None, is_file=None, stat=None, storage_options=None):
        self._path = path
        self._name = os.path.basename(path)
        self._is_file = is_file
        self._is_dir = is_dir
        self._stat = stat
        self._storage_options = storage_options

    def is_dir(self):
        if self._is_dir is None:
            self._is_dir = oss_isdir(self._path)
        return self._is_dir

    def is_file(self):
        if self._is_file is None:
            if self.is_dir() or not oss_exists(self._path):
                self._is_file = False
            else:
                self._is_file = True
        return self._is_file

    def stat(self):
        if self._stat is None:
            self._stat = oss_stat(self._path)
        return self._stat

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._path


def parse_osspath(path: path_type):
    # Extract OSS configuration from the encoded URL.
    str_path = stringify_path(path)
    parse_result = oss2.urlparse(str_path)
    if parse_result.scheme != "oss":
        raise ValueError(f"Except scheme oss, but got scheme: {parse_result.scheme}"
                         f" in path: {str_path}")
    bucket = parse_result.hostname
    if not (parse_result.username and parse_result.password):
        raise RuntimeError(r"Please use build_oss_path to add OSS info")
    param_dict = url_to_dict(parse_result.username)
    access_key_id = param_dict['access_key_id']
    access_key_secret = parse_result.password
    end_point = param_dict['end_point']
    key = parse_result.path
    key = key[1:] if key.startswith("/") else key
    return bucket, key, access_key_id, access_key_secret, end_point


def _get_oss_bucket(bucket, access_key_id, access_key_secret, end_point):
    oss_bucket = oss2.Bucket(
        auth=oss2.Auth(access_key_id=access_key_id, access_key_secret=access_key_secret),
        endpoint=end_point,
        bucket_name=bucket,
        connect_timeout=_oss_time_out)
    return oss_bucket


def oss_exists(path: path_type):
    bucket, key, access_key_id, access_key_secret, end_point = parse_osspath(path)
    oss_bucket = _get_oss_bucket(bucket, access_key_id, access_key_secret, end_point)
    return oss_bucket.object_exists(key) or oss_isdir(path)


def oss_isdir(path: path_type):
    """
    OSS has no concept of directories, but we define
    a ossurl is dir, When there is at least one object
    at the ossurl that is the prefix(end with char "/"),
    it is considered as a directory.
    """
    dirname = stringify_path(path)
    if not dirname.endswith("/"):
        dirname = dirname + "/"
    bucket, key, access_key_id, access_key_secret, end_point = parse_osspath(dirname)
    oss_bucket = _get_oss_bucket(bucket, access_key_id, access_key_secret, end_point)
    isdir = False
    for obj in oss2.ObjectIteratorV2(oss_bucket, prefix=key, max_keys=2):
        if obj.key == key:
            continue
        isdir = True
        break
    return isdir


def oss_stat(path: path_type):
    path = stringify_path(path)
    bucket, key, access_key_id, access_key_secret, end_point = parse_osspath(path)
    oss_bucket = _get_oss_bucket(bucket, access_key_id, access_key_secret, end_point)
    if oss_isdir(path):
        stat = dict(name=path, size=0, modified_time=-1)
        stat["type"] = "directory"
    else:
        meta = oss_bucket.get_object_meta(key)
        stat = dict(name=path, size=int(meta.headers["Content-Length"]),
                    modified_time=meta.headers["Last-Modified"])
        stat["type"] = "file"
    return stat


def oss_scandir(dirname: path_type):
    dirname = stringify_path(dirname)
    if not dirname.endswith("/"):
        dirname = dirname + "/"
    bucket, key, access_key_id, access_key_secret, end_point = parse_osspath(dirname)
    oss_bucket = _get_oss_bucket(bucket, access_key_id, access_key_secret, end_point)
    dirname_set = set()
    for obj in oss2.ObjectIteratorV2(oss_bucket, prefix=key):
        rel_path = obj.key[len(key):]
        try:
            inside_dirname, inside_filename = rel_path.split("/", 1)
        except ValueError:
            inside_dirname = None
            inside_filename = rel_path
        if inside_dirname is not None:
            if inside_dirname in dirname_set:
                continue
            dirname_set.add(inside_dirname)
            yield OSSFileEntry(
                os.path.join(dirname, inside_dirname),
                is_dir=True,
                is_file=False,
                stat={
                    "name": os.path.join(dirname, inside_dirname),
                    "type": "directory",
                    "size": 0,
                    "modified_time": -1,
                }
            )
        else:
            yield OSSFileEntry(
                os.path.join(dirname, inside_filename),
                is_dir=False,
                is_file=True,
                stat={
                    "name": os.path.join(dirname, inside_filename),
                    "type": "file",
                    "size": obj.size,
                    "modified_time": obj.last_modified,
                }
            )


def dict_to_url(param: dict):
    # Encode the dictionary with url-safe-base64.
    str_param = json.dumps(param)
    url_param = base64.urlsafe_b64encode(bytes(str_param, encoding='utf8'))
    return bytes.decode(url_param, encoding='utf8')


def url_to_dict(url_param: str):
    # Decode url-safe-base64 encoded string.
    bytes_param = bytes(url_param, encoding='utf8')
    str_param = bytes.decode(base64.urlsafe_b64decode(bytes_param), encoding='utf8')
    return json.loads(str_param)
