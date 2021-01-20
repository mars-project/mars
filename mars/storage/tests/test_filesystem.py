#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import numpy as np
import pandas as pd

from mars.filesystem import LocalFileSystem
from mars.storage.core import StorageLevel
from mars.storage.filesystem import FileSystemStorage


def test_base_operations():
    with tempfile.TemporaryDirectory() as tempdir:
        fs = LocalFileSystem()
        storage = FileSystemStorage(fs, [tempdir])

        assert storage.level == StorageLevel.DISK

        data1 = np.random.rand(10, 10)
        put_info1 = storage.put(data1)
        get_data1 = storage.get(put_info1.object_id)
        np.testing.assert_array_equal(data1, get_data1)

        info1 = storage.info(put_info1.object_id)
        assert info1.size == put_info1.size

        data2 = pd.DataFrame({'col1': np.arange(10),
                              'col2': [f'str{i}' for i in range(10)],
                              'col3': np.random.rand(10)},)
        put_info2 = storage.put(data2)
        get_data2 = storage.get(put_info2.object_id)
        pd.testing.assert_frame_equal(data2, get_data2)

        info2 = storage.info(put_info2.object_id)
        assert info2.size == put_info2.size

        storage.delete(info2.object_id)
        assert not os.path.exists(os.path.join(tempdir, info2.object_id))

        # test writer and reader
        test_word1 = b'test'
        test_word2 = b'filesystem'
        with storage.create_writer() as writer:
            writer.write(test_word1)
            writer.write(test_word2)

        with storage.open_reader(writer.object_id) as reader:
            content = reader.read()

        assert content == test_word1 + test_word2


