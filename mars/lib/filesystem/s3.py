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
import os

"""
An example to read csv from s3
------------------------------
>>> import mars
>>> import mars.dataframe as md
>>>
>>> mars.new_session()
>>> # Pass endpoint_url / aws_access_key_id / aws_secret_access_key to read_csv.
>>> mdf = md.read_csv("s3://bucket/example.csv", index_col=0, storage_options={
>>>     "client_kwargs": {
>>>         "endpoint_url": "http://192.168.1.12:9000",
>>>         "aws_access_key_id": "<s3 access id>",
>>>         "aws_secret_access_key": "<s3 access key>",
>>>     }})
>>> # Export environment vars AWS_ENDPOINT_URL / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.
>>> mdf = md.read_csv("s3://bucket/example.csv", index_col=0)
>>> r = mdf.head(1000).execute()
>>> print(r)
"""

try:  # pragma: no cover
    # make sure s3fs is installed
    from s3fs import S3FileSystem as _S3FileSystem

    # make sure fsspec is installed
    from .fsspec_adapter import FsSpecAdapter

    del _S3FileSystem
except ImportError:
    FsSpecAdapter = None

if FsSpecAdapter is not None:  # pragma: no cover
    from .core import register_filesystem

    class S3FileSystem(FsSpecAdapter):
        def __init__(self, **kwargs):
            super().__init__("s3", **kwargs)

        @staticmethod
        def parse_from_path(uri: str):
            client_kwargs = {
                "endpoint_url": os.environ.get("AWS_ENDPOINT_URL"),
                "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID"),
                "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
                "aws_session_token": os.environ.get("AWS_SESSION_TOKEN"),
            }
            client_kwargs = {k: v for k, v in client_kwargs.items() if v is not None}
            return {"client_kwargs": client_kwargs}

    register_filesystem("s3", S3FileSystem)
else:
    S3FileSystem = None
