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
import os

from ..... import tensor as mt
from ..... import dataframe as md
from .....utils import lazy_import
from .. import gen_tensorflow_dataset, run_tensorflow_script

tf_installed = lazy_import("tensorflow") is not None


@pytest.mark.skipif(not tf_installed, reason="tensorflow not installed")
def test_mars_dataset(setup_cluster):
    import numpy as np
    import pandas as pd

    tf_dataset_ops = lazy_import("tensorflow.python.data.ops.dataset_ops")

    # Mars tensor
    data = mt.random.rand(1000, 32, dtype="f4")
    data_verify = data[:10].execute().fetch()

    dataset = gen_tensorflow_dataset(data)
    assert isinstance(dataset, tf_dataset_ops.DatasetV2)
    for _, data_1batch in enumerate(dataset.repeat().batch(10).take(1)):
        np.testing.assert_array_equal(data_1batch, data_verify)

    # Mars tensors
    data = mt.random.rand(1000, 32, dtype="f4")
    labels = mt.random.randint(0, 2, (1000, 10), dtype="f4")

    data_verify = data[:10].execute().fetch()
    labels_verify = labels[:10].execute().fetch()

    dataset = gen_tensorflow_dataset((data, labels))
    assert isinstance(dataset, tf_dataset_ops.DatasetV2)
    for _, (data_1batch, label_1batch) in enumerate(dataset.repeat().batch(10).take(1)):
        np.testing.assert_array_equal(data_1batch, data_verify)
        np.testing.assert_array_equal(label_1batch, labels_verify)

    # np ndarray
    data = np.random.rand(1000, 32)
    labels = np.random.randint(0, 2, (1000, 10))

    data_verify = data[:10]
    labels_verify = labels[:10]

    dataset = gen_tensorflow_dataset((data, labels))
    assert isinstance(dataset, tf_dataset_ops.DatasetV2)
    for _, (data_1batch, label_1batch) in enumerate(dataset.repeat().batch(10).take(1)):
        np.testing.assert_array_equal(data_1batch, data_verify)
        np.testing.assert_array_equal(label_1batch, labels_verify)

    # Mars dataframe
    data = md.DataFrame(data)
    labels = md.DataFrame(labels)

    data_verify = data.iloc[:10].execute().fetch().values
    labels_verify = labels.iloc[:10].execute().fetch().values

    dataset = gen_tensorflow_dataset(
        (data, labels), fetch_kwargs={"extra_config": {"check_series_name": False}}
    )
    assert isinstance(dataset, tf_dataset_ops.DatasetV2)
    for _, (data_1batch, label_1batch) in enumerate(dataset.repeat().batch(10).take(1)):
        np.testing.assert_array_equal(data_1batch, data_verify)
        np.testing.assert_array_equal(label_1batch, labels_verify)

    # Mars series
    label = labels[1]

    label_verify = label[:10].execute().fetch()

    dataset = gen_tensorflow_dataset(
        (data, label), fetch_kwargs={"extra_config": {"check_series_name": False}}
    )
    assert isinstance(dataset, tf_dataset_ops.DatasetV2)
    for _, (data_1batch, label_1batch) in enumerate(dataset.repeat().batch(10).take(1)):
        np.testing.assert_array_equal(data_1batch, data_verify)
        np.testing.assert_array_equal(label_1batch, label_verify)

    # pandas dataframe
    data = pd.DataFrame(np.random.rand(1000, 32))
    labels = pd.DataFrame(np.random.randint(0, 2, (1000, 10)), dtype="float32")

    data_verify = data.iloc[:10].values
    labels_verify = labels.iloc[:10].values
    dataset = gen_tensorflow_dataset((data, labels))
    assert isinstance(dataset, tf_dataset_ops.DatasetV2)
    for _, (data_1batch, label_1batch) in enumerate(dataset.repeat().batch(10).take(1)):
        np.testing.assert_array_equal(data_1batch, data_verify)
        np.testing.assert_array_equal(label_1batch, labels_verify)

    # pandas series
    label = labels[1]

    label_verify = label[:10]

    dataset = gen_tensorflow_dataset((data, label))
    assert isinstance(dataset, tf_dataset_ops.DatasetV2)
    for _, (data_1batch, label_1batch) in enumerate(dataset.repeat().batch(10).take(1)):
        np.testing.assert_array_equal(data_1batch, data_verify)
        np.testing.assert_array_equal(label_1batch, label_verify)

    # list
    label = label.tolist()

    label_verify = label[:10]

    dataset = gen_tensorflow_dataset((data, label))
    assert isinstance(dataset, tf_dataset_ops.DatasetV2)
    for _, (data_1batch, label_1batch) in enumerate(dataset.repeat().batch(10).take(1)):
        np.testing.assert_array_equal(data_1batch, data_verify)
        np.testing.assert_array_equal(label_1batch, label_verify)

    # test TypeError
    label = tuple(range(1000))

    with pytest.raises(TypeError) as e:
        dataset = gen_tensorflow_dataset((data, label))
    exec_msg = e.value.args[0]
    assert exec_msg == "Unexpected dataset type: <class 'tuple'>"


@pytest.mark.skipif(not tf_installed, reason="tensorflow not installed")
def test_mars_dataset_script(setup_cluster):
    sess = setup_cluster
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tf_dataset.py")

    data = mt.random.rand(1000, 32, dtype="f4")
    labels = mt.random.randint(0, 2, (1000, 10), dtype="f4")

    assert (
        run_tensorflow_script(
            path,
            n_workers=2,
            data={"feature_data": data, "labels": labels},
            command_argv=["multiple"],
            session=sess,
        ).fetch()["status"]
        == "ok"
    )
