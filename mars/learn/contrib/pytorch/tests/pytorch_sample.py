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

import sys


def get_model():
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
        nn.Softmax(),
    )


def main():
    import torch.nn as nn
    import torch.distributed as dist
    import torch.optim as optim
    import torch.utils.data

    dist.init_process_group(backend="gloo")
    torch.manual_seed(42)

    data = torch.rand((1000, 32), dtype=torch.float32)
    labels = torch.randint(1, (1000, 10), dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(data, labels)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=32, shuffle=False, sampler=train_sampler
    )

    model = nn.parallel.DistributedDataParallel(get_model())
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.BCELoss()

    for _ in range(2):
        # 2 epochs
        for _, (batch_data, batch_labels) in enumerate(train_loader):
            outputs = model(batch_data)
            loss = criterion(outputs.squeeze(), batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    assert len(sys.argv) == 2
    assert sys.argv[1] == "multiple"
    main()
