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

import pytest

from ....dataframe import read_csv
from ..core import register_filesystem
from ..s3 import S3FileSystem


class KwArgsException(Exception):
    def __init__(self, kwargs):
        self.kwargs = kwargs


if S3FileSystem is not None:

    class TestS3FileSystem(S3FileSystem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            raise KwArgsException(kwargs)

else:
    TestS3FileSystem = None


@pytest.mark.skipif(S3FileSystem is None, reason="S3 is not supported")
def test_client_kwargs():
    register_filesystem("s3", TestS3FileSystem)

    test_kwargs = {
        "endpoint_url": "http://192.168.1.12:9000",
        "aws_access_key_id": "test_id",
        "aws_secret_access_key": "test_key",
        "aws_session_token": "test_session_token",
    }

    def _assert_true():
        # Pass endpoint_url / aws_access_key_id / aws_secret_access_key / aws_session_token to read_csv.
        with pytest.raises(KwArgsException) as e:
            read_csv(
                "s3://bucket/example.csv",
                index_col=0,
                storage_options={"client_kwargs": test_kwargs},
            )
        assert e.value.kwargs == {
            "client_kwargs": {
                "endpoint_url": "http://192.168.1.12:9000",
                "aws_access_key_id": "test_id",
                "aws_secret_access_key": "test_key",
                "aws_session_token": "test_session_token",
            }
        }

    _assert_true()

    test_env = {
        "AWS_ENDPOINT_URL": "a",
        "AWS_ACCESS_KEY_ID": "b",
        "AWS_SECRET_ACCESS_KEY": "c",
        "AWS_SESSION_TOKEN": "d",
    }
    for k, v in test_env.items():
        os.environ[k] = v

    try:
        _assert_true()

        for k, v in test_kwargs.items():
            with pytest.raises(KwArgsException) as e:
                read_csv(
                    "s3://bucket/example.csv",
                    index_col=0,
                    storage_options={"client_kwargs": {k: v}},
                )
            expect = {
                "endpoint_url": "a",
                "aws_access_key_id": "b",
                "aws_secret_access_key": "c",
                "aws_session_token": "d",
            }
            expect[k] = v
            assert e.value.kwargs == {"client_kwargs": expect}
    finally:
        for k, v in test_env.items():
            os.environ.pop(k, None)
