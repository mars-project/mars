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

from mars.learn.contrib.xgboost import train
import sys
sys.path.append("/mnt/d/OneDrive/专业/420实验室/开源之夏/project/mars")
import unittest

import mars.tensor as mt
from mars.session import new_session
from mars.tests import setup
from mars.utils import lazy_import
# from mars.learn.contrib.pytorch import MarsDataset, MarsRandomSampler
from mars.learn.contrib.pytorch.dataset import MarsDataset
from mars.learn.contrib.pytorch.sampler import MarsSequentialSampler, MarsRandomSampler

torch_installed = lazy_import('torch', globals=globals()) is not None

setup = setup

@unittest.skipIf(not torch_installed, 'pytorch not installed')
# class Test(unittest.TestCase):
#     # def setUp(self) -> None:
#     #     self.session = new_session().as_default()
#     #     self._old_executor = self.session._sess._executor
#     #     self.executor = self.session._sess._executor = \
#     #         ExecutorForTest('numpy', storage=self.session._sess._context)

#     # def tearDown(self) -> None:
#     #     self.session._sess._executor = self._old_executor


def testMarsDataset(setup):
    import torch
    from torch.utils.data import Dataset
    import numpy as np

    data = mt.random.rand(1000, 32, dtype='f4')
    labels = mt.random.randint(0, 2, (1000, 10), dtype='f4')
    data.execute().fetch()
    labels.execute().fetch()

    train_dataset = MarsDataset(data, labels)
    assert len(train_dataset) == 1000
    assert train_dataset[1][0].shape == (32,)
    assert train_dataset[1][1].shape == (10,)
    assert isinstance(train_dataset, Dataset)
    assert isinstance(train_dataset[1][0], np.ndarray)

def testMarsDatasetWithSampler(setup):
    import torch
    from torch.utils.data import SequentialSampler, RandomSampler

    data = mt.random.rand(1000, 32, dtype='f4')
    labels = mt.random.randint(0, 2, (1000, 10), dtype='f4')
    data.execute().fetch()
    labels.execute().fetch()

    train_dataset = MarsDataset(data, labels)

    train_sampler = RandomSampler(train_dataset)
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

def testSequentialSampler(setup):
    import torch

    data = mt.random.rand(1000, 32, dtype='f4')
    labels = mt.random.randint(0, 2, (1000, 10), dtype='f4')
    data.execute().fetch()
    labels.execute().fetch()

    train_dataset = MarsDataset(data, labels)

    train_sampler = MarsSequentialSampler(train_dataset)
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

def testRandomSampler(setup):
    import torch

    data = mt.random.rand(1000, 32, dtype='f4')
    labels = mt.random.randint(0, 2, (1000, 10), dtype='f4')
    data.execute().fetch()
    labels.execute().fetch()

    train_dataset = MarsDataset(data, labels)

    train_sampler = MarsRandomSampler(train_dataset)
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
          

# def testLocalDataset(setup):
#     import torch

#     data = mt.random.rand(1000, 32, dtype='f4')
#     labels = mt.random.randint(0, 2, (1000, 10), dtype='f4')
#     data.execute().fetch()
#     labels.execute().fetch()

#     train_dataset = MarsDataset(data, labels)

#     train_sampler = MarsRandomSampler(train_dataset)
#     train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                                 batch_size=32,
#                                                 sampler=train_sampler)

#     model = torch.nn.Sequential(
#         torch.nn.Linear(32, 64),
#         torch.nn.ReLU(),
#         torch.nn.Linear(64, 64),
#         torch.nn.ReLU(),
#         torch.nn.Linear(64, 10),
#         torch.nn.Softmax(dim=1),
#     )

#     optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
#     criterion = torch.nn.BCELoss()
#     for _ in range(2):
#         # 2 epochs
#         for _, (batch_data, batch_labels) in enumerate(train_loader):
#             outputs = model(batch_data)
#             loss = criterion(outputs.squeeze(), batch_labels)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()



# if __name__ == "__main__":
#     testLocalDataset(setup)
