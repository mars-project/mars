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

from mars.core.entity import output_types
import pytest
import os

import mars.tensor as mt
from mars.session import new_session
from mars.utils import lazy_import
from mars.learn.contrib.tensorflow import MarsTFDataset

tf_installed = lazy_import('tensorflow', globals=globals()) is not None

@pytest.mark.skipif(not tf_installed, reason='tensorflow not installed')
def testMarsTFDataset(setup_cluster):
    import tensorflow as tf
    from tensorflow.python.data.ops.dataset_ops import DatasetV2
    import numpy as np

    sess = setup_cluster

    data = mt.random.rand(1000, 32, dtype='f4')
    labels = mt.random.randint(0, 2, (1000, 10), dtype='f4')
    data.execute()
    labels.execute()

    dataset = MarsTFDataset(data, labels).get_tfdataset()
    # output_shapes=((32, ), (10, )), output_types=(tf.float32, tf.int32)
    print(list(dataset.take(1)))

    assert isinstance(dataset, DatasetV2)
