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
import tempfile

import pytest

from mars.filesystem import LocalFileSystem
from mars.aio import AioFilesystem


@pytest.mark.asyncio
async def test_aio_filesystem():
    local_fs = LocalFileSystem.get_instance()
    aio_fs = AioFilesystem(local_fs)

    assert aio_fs.pathsep == local_fs.pathsep

    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, 'test')

        with open(file_path, 'wb') as f:
            f.write(b'text for test')

        stat = await aio_fs.stat(tempdir)
        assert stat['type'] == 'directory'
