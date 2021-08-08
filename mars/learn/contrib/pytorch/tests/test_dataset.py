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

import mars.tensor as mt
import mars.dataframe as md
from mars.utils import lazy_import
from mars.learn.contrib.pytorch import MarsDataset, RandomSampler, SequentialSampler, SubsetRandomSampler

torch_installed = lazy_import('torch', globals=globals()) is not None


@pytest.mark.skipif(not torch_installed, reason='pytorch not installed')
def test_MarsDataset(setup_cluster):
    from torch.utils.data import Dataset
    import numpy as np
    import pandas as pd

    # Mars tensor
    data = mt.random.rand(1000, 32, dtype='f4')
    labels = mt.random.randint(0, 2, (1000, 10), dtype='f4')

    data_verify = data[0].execute().fetch()
    labels_verify = labels[0].execute().fetch()

    train_dataset = MarsDataset(data, labels)
    assert isinstance(train_dataset, Dataset)
    assert (train_dataset[0][0] == data_verify).all()
    assert (train_dataset[0][1] == labels_verify).all()
    assert len(train_dataset) == 1000

    # np ndarray
    data = np.random.rand(1000, 32)
    labels = np.random.randint(0, 2, (1000, 10))

    data_verify = data[0]
    labels.dtype = "float32"
    labels_verify = labels[0]

    train_dataset = MarsDataset(data, labels)
    assert (train_dataset[0][0] == data_verify).all()
    assert (train_dataset[0][1] == labels_verify).all()
    assert len(train_dataset) == 1000

    # Mars dataframe
    data = md.DataFrame(data)
    labels = md.DataFrame(labels)

    data_verify = data.iloc[0].execute().fetch().values
    labels_verify = labels.iloc[0].execute().fetch().values

    train_dataset = MarsDataset(data, labels)
    assert (train_dataset[0][0] == data_verify).all()
    assert (train_dataset[0][1] == labels_verify).all()
    assert len(train_dataset) == 1000

    # Mars Series
    label = labels[0]

    label_verify = label[0].execute().fetch()

    train_dataset = MarsDataset(data, label)
    assert (train_dataset[0][0] == data_verify).all()
    assert train_dataset[0][1] == label_verify
    assert len(train_dataset) == 1000

    # pandas dataframe
    data = pd.DataFrame(np.random.rand(1000, 32))
    labels = pd.DataFrame(np.random.randint(0, 2, (1000, 10)), dtype="float32")

    data_verify = data.iloc[0].values
    labels_verify = labels.iloc[0].values

    train_dataset = MarsDataset(data, labels)
    assert (train_dataset[0][0] == data_verify).all()
    assert (train_dataset[0][1] == labels_verify).all()
    assert len(train_dataset) == 1000

    # pands series
    label = labels[0]
    label_verify = label[0]

    train_dataset = MarsDataset(data, label)
    assert (train_dataset[0][0] == data_verify).all()
    assert train_dataset[0][1] == label_verify
    assert len(train_dataset) == 1000


@pytest.mark.skipif(not torch_installed, reason='pytorch not installed')
def test_SequentialSampler(setup_cluster):
    import torch

    data = mt.random.rand(1000, 32, dtype='f4')
    labels = mt.random.randint(0, 2, (1000, 10), dtype='f4')
    data.execute()
    labels.execute()

    train_dataset = MarsDataset(data, labels)
    assert len(train_dataset) == 1000

    train_sampler = SequentialSampler(train_dataset)
    assert len(train_sampler) == 1000

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=32,
                                                sampler=train_sampler)

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


@pytest.mark.skipif(not torch_installed, reason='pytorch not installed')
def test_RandomSampler(setup_cluster):
    import torch

    data = mt.random.rand(1000, 32, dtype='f4')
    labels = mt.random.randint(0, 2, (1000, 10), dtype='f4')
    data.execute().fetch()
    labels.execute().fetch()

    train_dataset = MarsDataset(data, labels)

    # test __init__()
    with pytest.raises(ValueError) as e:
        train_sampler = RandomSampler(train_dataset, replacement=1)
    exec_msg = e.value.args[0]
    assert exec_msg == "replacement should be a boolean value, but got replacement=1"

    with pytest.raises(ValueError) as e:
        train_sampler = RandomSampler(train_dataset, num_samples=900)
    exec_msg = e.value.args[0]
    assert exec_msg == "With replacement=False, num_samples should not be specified, since a random permute will be performed."

    with pytest.raises(ValueError) as e:
        train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=-1)
    exec_msg = e.value.args[0]
    assert exec_msg == "num_samples should be a positive integer value, but got num_samples=-1"

    train_sampler = RandomSampler(train_dataset)

    # test __len__ num_samples()
    assert len(train_sampler) == 1000
    assert train_sampler.num_samples == 1000

    # test __iter__
    g_cpu = torch.Generator()
    g_cpu.manual_seed(2147483647)

    train_sampler = RandomSampler(train_dataset, generator=g_cpu)
    assert len(train_sampler) == 1000
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=32,
                                                sampler=train_sampler)
    for _, (batch_data, batch_labels) in enumerate(train_loader):
        assert len(batch_data[0]) == 32
        assert len(batch_labels[0]) == 10

    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=900)
    assert len(train_sampler) == 900
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=32,
                                                sampler=train_sampler)
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


@pytest.mark.skipif(not torch_installed, reason='pytorch not installed')
def test_SubsetRandomSampler(setup_cluster):
    import numpy as np
    import torch

    data = mt.random.rand(1000, 32, dtype='f4')
    labels = mt.random.randint(0, 2, (1000, 10), dtype='f4')
    data.execute()
    labels.execute()

    train_dataset = MarsDataset(data, labels)
    train_sampler = SubsetRandomSampler(
                    np.random.choice(range(len(train_dataset)), len(train_dataset)))

    assert len(train_sampler) == 1000
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=32,
                                                sampler=train_sampler)
    for _, (batch_data, batch_labels) in enumerate(train_loader):
        assert len(batch_data[0]) == 32
        assert len(batch_labels[0]) == 10
