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
from .. import (
    MarsDataset,
    RandomSampler,
    SequentialSampler,
    SubsetRandomSampler,
    DistributedSampler,
    run_pytorch_script,
)

torch_installed = lazy_import("torch") is not None


@pytest.mark.skipif(not torch_installed, reason="pytorch not installed")
def test_mars_dataset(setup):
    from torch.utils.data import Dataset
    import numpy as np
    import pandas as pd

    # Mars tensor
    data = mt.random.rand(1000, 32, dtype="f4")
    labels = mt.random.randint(0, 2, (1000, 10), dtype="f4")

    data_verify = data[1].execute().fetch()
    labels_verify = labels[1].execute().fetch()

    train_dataset = MarsDataset(data, labels)

    assert isinstance(train_dataset, Dataset)
    np.testing.assert_array_equal(train_dataset[1][0], data_verify)
    np.testing.assert_array_equal(train_dataset[1][1], labels_verify)
    assert len(train_dataset) == 1000

    # np ndarray
    data = np.random.rand(1000, 32)
    labels = np.random.randint(0, 2, (1000, 10))

    data_verify = data[1]
    labels.dtype = "float32"
    labels_verify = labels[1]

    train_dataset = MarsDataset(data, labels)
    np.testing.assert_array_equal(train_dataset[1][0], data_verify)
    np.testing.assert_array_equal(train_dataset[1][1], labels_verify)
    assert len(train_dataset) == 1000

    # Mars dataframe
    data = md.DataFrame(data)
    labels = md.DataFrame(labels)

    data_verify = data.iloc[1].execute().fetch().values
    labels_verify = labels.iloc[1].execute().fetch().values

    train_dataset = MarsDataset(
        data, labels, fetch_kwargs={"extra_config": {"check_series_name": False}}
    )
    np.testing.assert_array_equal(train_dataset[1][0], data_verify)
    np.testing.assert_array_equal(train_dataset[1][1], labels_verify)
    assert len(train_dataset) == 1000

    # Mars Series
    label = labels[1]

    label_verify = label[1].execute().fetch()

    train_dataset = MarsDataset(
        data, label, fetch_kwargs={"extra_config": {"check_series_name": False}}
    )
    np.testing.assert_array_equal(train_dataset[1][0], data_verify)
    assert train_dataset[1][1] == label_verify
    assert len(train_dataset) == 1000

    # pandas dataframe
    data = pd.DataFrame(np.random.rand(1000, 32))
    labels = pd.DataFrame(np.random.randint(0, 2, (1000, 10)), dtype="float32")

    data_verify = data.iloc[1].values
    labels_verify = labels.iloc[1].values

    train_dataset = MarsDataset(data, labels)
    np.testing.assert_array_equal(train_dataset[1][0], data_verify)
    np.testing.assert_array_equal(train_dataset[1][1], labels_verify)
    assert len(train_dataset) == 1000

    # pands series
    label = labels[1]
    label_verify = label[1]

    train_dataset = MarsDataset(data, label)
    np.testing.assert_array_equal(train_dataset[1][0], data_verify)
    assert train_dataset[1][1] == label_verify
    assert len(train_dataset) == 1000

    # test TypeError
    label = tuple(range(1000))

    with pytest.raises(TypeError) as e:
        train_dataset = MarsDataset(data, label)
    exec_msg = e.value.args[0]
    assert exec_msg == "Unexpected dataset type: <class 'tuple'>"


@pytest.mark.skipif(not torch_installed, reason="pytorch not installed")
def test_sequential_sampler(setup_cluster):
    import torch

    data = mt.random.rand(1000, 32, dtype="f4")
    labels = mt.random.randint(0, 2, (1000, 10), dtype="f4")

    train_dataset = MarsDataset(data, labels)
    assert len(train_dataset) == 1000

    train_sampler = SequentialSampler(train_dataset)
    assert len(train_sampler) == 1000

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=32, sampler=train_sampler
    )

    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10),
        torch.nn.Softmax(dim=1),
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion = torch.nn.BCELoss()
    for _ in range(2):
        # 2 epochs
        for _, (batch_data, batch_labels) in enumerate(train_loader):
            outputs = model(batch_data)
            loss = criterion(outputs.squeeze(), batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@pytest.mark.skipif(not torch_installed, reason="pytorch not installed")
def test_random_sampler(setup_cluster):
    import torch

    data = mt.random.rand(1000, 32, dtype="f4")
    labels = mt.random.randint(0, 2, (1000, 10), dtype="f4")

    train_dataset = MarsDataset(data, labels)

    # test __init__()
    with pytest.raises(ValueError) as e:
        train_sampler = RandomSampler(train_dataset, replacement=1)
    exec_msg = e.value.args[0]
    assert exec_msg == "replacement should be a boolean value, but got replacement=1"

    with pytest.raises(ValueError) as e:
        train_sampler = RandomSampler(train_dataset, num_samples=900)
    exec_msg = e.value.args[0]
    assert (
        exec_msg
        == "With replacement=False, num_samples should not "
        + "be specified, since a random permute will be performed."
    )

    with pytest.raises(ValueError) as e:
        train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=-1)
    exec_msg = e.value.args[0]
    assert (
        exec_msg
        == "num_samples should be a positive integer value, but got num_samples=-1"
    )

    train_sampler = RandomSampler(train_dataset)

    # test __len__ num_samples()
    assert len(train_sampler) == 1000
    assert train_sampler.num_samples == 1000

    # test __iter__
    g_cpu = torch.Generator()
    g_cpu.manual_seed(2147483647)

    train_sampler = RandomSampler(train_dataset, generator=g_cpu)
    assert len(train_sampler) == 1000
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=32, sampler=train_sampler
    )
    for _, (batch_data, batch_labels) in enumerate(train_loader):
        assert len(batch_data[0]) == 32
        assert len(batch_labels[0]) == 10

    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=900)
    assert len(train_sampler) == 900
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=32, sampler=train_sampler
    )
    for _, (batch_data, batch_labels) in enumerate(train_loader):
        assert len(batch_data[0]) == 32
        assert len(batch_labels[0]) == 10

    # torch train
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10),
        torch.nn.Softmax(dim=1),
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion = torch.nn.BCELoss()
    for _ in range(2):
        # 2 epochs
        for _, (batch_data, batch_labels) in enumerate(train_loader):
            outputs = model(batch_data)
            loss = criterion(outputs.squeeze(), batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@pytest.mark.skipif(not torch_installed, reason="pytorch not installed")
def test_subset_random_sampler(setup_cluster):
    import numpy as np
    import torch

    data = mt.random.rand(1000, 32, dtype="f4")
    labels = mt.random.randint(0, 2, (1000, 10), dtype="f4")
    data.execute()
    labels.execute()

    train_dataset = MarsDataset(data, labels)
    train_sampler = SubsetRandomSampler(
        np.random.choice(range(len(train_dataset)), len(train_dataset))
    )

    assert len(train_sampler) == 1000
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=32, sampler=train_sampler
    )
    for _, (batch_data, batch_labels) in enumerate(train_loader):
        assert len(batch_data[0]) == 32
        assert len(batch_labels[0]) == 10


@pytest.mark.skipif(not torch_installed, reason="pytorch not installed")
def test_distributed_sampler(setup_cluster):
    import torch

    data = mt.random.rand(1001, 32, dtype="f4")
    labels = mt.random.randint(0, 2, (1001, 10), dtype="f4")

    train_dataset = MarsDataset(data, labels)

    with pytest.raises(ValueError) as e:
        train_sampler = DistributedSampler(train_dataset, num_replicas=2, rank=-1)
    exec_msg = e.value.args[0]
    assert exec_msg == "Invalid rank -1, rank should be in the interval [0, 1]"

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=2, rank=0, drop_last=True, shuffle=True
    )
    assert len(train_sampler) == 500
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=32, sampler=train_sampler
    )
    for _, (batch_data, batch_labels) in enumerate(train_loader):
        assert len(batch_data[0]) == 32
        assert len(batch_labels[0]) == 10

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=2, rank=0, drop_last=False, shuffle=False
    )
    train_sampler.set_epoch(10)
    assert len(train_sampler) == 501
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=32, sampler=train_sampler
    )
    for _, (batch_data, batch_labels) in enumerate(train_loader):
        assert len(batch_data[0]) == 32
        assert len(batch_labels[0]) == 10


@pytest.mark.skipif(not torch_installed, reason="pytorch not installed")
def test_mars_dataset_script(setup_cluster):
    sess = setup_cluster
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "pytorch_dataset.py"
    )

    data = mt.random.rand(1000, 32, dtype="f4")
    labels = mt.random.randint(0, 2, (1000, 10), dtype="f4")

    assert (
        run_pytorch_script(
            path,
            n_workers=2,
            data={"feature_data": data, "labels": labels},
            command_argv=["multiple"],
            port=9945,
            session=sess,
        ).fetch()["status"]
        == "ok"
    )
