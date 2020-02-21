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

import unittest

import mars.tensor as mt
from mars.utils import lazy_import
from mars.learn.contrib.pytorch import MarsDataset, MarsRandomSampler

torch_installed = lazy_import('torch', globals=globals()) is not None


@unittest.skipIf(not torch_installed, 'pytorch not installed')
class Test(unittest.TestCase):
    def testLocalDataset(self):
        import torch

        data = mt.random.rand(1000, 32, dtype='f4')
        labels = mt.random.randint(0, 2, (1000, 10), dtype='f4')
        data.execute(fetch=False)
        labels.execute(fetch=False)

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
            torch.nn.Softmax(),
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
