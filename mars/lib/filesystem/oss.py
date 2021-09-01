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

from typing import List, Dict, Tuple, Iterator
from urllib import parse

from ._oss_lib import common as oc
from ._oss_lib.glob import glob
from ._oss_lib.handle import OSSIOBase, dict_to_url
from .base import FileSystem, path_type
from ...utils import implements, ModulePlaceholder

try:
    import oss2
except ImportError:
    oss2 = ModulePlaceholder('oss2')

_oss_time_out = 10


class OSSFileSystem(FileSystem):
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = OSSFileSystem()
        return cls._instance

    @implements(FileSystem.cat)
    def cat(self, path: path_type):
        raise NotImplementedError

    @implements(FileSystem.ls)
    def ls(self, path: path_type) -> List[path_type]:
        file_list = []
        file_entry = oc.OSSFileEntry(path)
        if not file_entry.is_dir():
            raise OSError("ls for file is not supported")
        else:
            bucket, key, access_key_id, access_key_secret, end_point \
                = oc.parse_osspath(path)
            oss_bucket = oss2.Bucket(
                auth=oss2.Auth(access_key_id=access_key_id,
                               access_key_secret=access_key_secret),
                endpoint=end_point,
                bucket_name=bucket,
                connect_timeout=_oss_time_out)
            for obj in oss2.ObjectIteratorV2(oss_bucket, prefix=key):
                if obj.key.endswith('/'):
                    continue
                obj_path = rf"oss://{bucket}/{obj.key}"
                file_list.append(obj_path)
        return file_list

    @implements(FileSystem.delete)
    def delete(self,
               path: path_type,
               recursive: bool = False):
        raise NotImplementedError

    @implements(FileSystem.rename)
    def rename(self,
               path: path_type,
               new_path: path_type):
        raise NotImplementedError

    @implements(FileSystem.stat)
    def stat(self, path: path_type) -> Dict:
        ofe = oc.OSSFileEntry(path)
        return ofe.stat()

    @implements(FileSystem.mkdir)
    def mkdir(self,
              path: path_type,
              create_parents: bool = True):
        raise NotImplementedError

    @implements(FileSystem.isdir)
    def isdir(self, path: path_type) -> bool:
        file_entry = oc.OSSFileEntry(path)
        return file_entry.is_dir()

    @implements(FileSystem.isfile)
    def isfile(self, path: path_type) -> bool:
        file_entry = oc.OSSFileEntry(path)
        return file_entry.is_file()

    @implements(FileSystem._isfilestore)
    def _isfilestore(self) -> bool:
        raise NotImplementedError

    @implements(FileSystem.exists)
    def exists(self, path: path_type):
        return oc.oss_exists(path)

    @implements(FileSystem.open)
    def open(self,
             path: path_type,
             mode: str = 'rb') -> OSSIOBase:
        file_handle = OSSIOBase(path, mode)
        return file_handle

    @implements(FileSystem.walk)
    def walk(self, path: path_type) \
            -> Iterator[Tuple[str, List[str], List[str]]]:
        raise NotImplementedError

    @implements(FileSystem.glob)
    def glob(self,
             path: path_type,
             recursive: bool = False) -> List[path_type]:
        return glob(path, recursive=recursive)


def build_oss_path(path: path_type, access_key_id,
                   access_key_secret, end_point):
    """
    Returns a path with oss info.
    Used to register the access_key_id, access_key_secret and
    endpoint of OSS. The access_key_id and endpoint are put
    into the url with url-safe-base64 encoding.

    Parameters
    ----------
    path : path_type
        The original oss url.

    access_key_id : str
        The access key id of oss.

    access_key_secret : str
        The access key secret of oss.

    end_point : str
        The endpoint of oss.

    Returns
    -------
    path_type
        Path include the encoded access key id, end point and
        access key secret of oss.
    """
    if isinstance(path, (list, tuple)):
        path = path[0]
    param_dict = {'access_key_id': access_key_id, 'end_point': end_point}
    id_endpoint = dict_to_url(param_dict)
    password = access_key_secret
    parse_result = parse.urlparse(path)
    new_path = f'{parse_result.scheme}://{id_endpoint}:{password}' \
               f'@{parse_result.netloc}{parse_result.path}'
    return new_path
