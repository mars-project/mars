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

import pytest
import sys

import numpy as np
import pandas as pd

from mars.serialize import dataserializer
from mars.storage.core import StorageLevel
from mars.storage.plasma import PlasmaStorage


@pytest.fixture
def plasma_ctx():
    plasma_storage_size = 10 * 1024 * 1024
    if sys.platform == 'darwin':
        plasma_dir = '/tmp'
    else:
        plasma_dir = '/dev/shm'
    params = PlasmaStorage.setup(plasma_storage_size, plasma_dir)
    init_params = dict(plasma_socket=params['plasma_socket'],
                       plasma_directory=params['plasma_directory'])
    yield init_params
    PlasmaStorage.teardown(params['plasma_store'])


def test_base_operations(plasma_ctx):
    init_params = plasma_ctx
    storage = PlasmaStorage(**init_params)

    assert storage.level == StorageLevel.MEMORY

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

    # test writer and reader
    t = np.random.random(10)
    b = dataserializer.dumps(t)
    with storage.create_writer(size=len(b)) as writer:
        split = len(b) // 2
        writer.write(b[:split])
        writer.write(b[split:])

    with storage.open_reader(writer.object_id) as reader:
        content = reader.read()
        t2 = dataserializer.loads(content)

    np.testing.assert_array_equal(t, t2)
