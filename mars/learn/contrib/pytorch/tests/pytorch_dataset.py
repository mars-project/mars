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


def main(feature_data, labels):
    import torch.nn as nn
    import torch.distributed as dist
    import torch.optim as optim
    import torch.utils.data
    from mars.learn.contrib.pytorch import MarsDataset, DistributedSampler

    dist.init_process_group(backend="gloo")
    torch.manual_seed(42)

    data = feature_data
    labels = labels

    train_dataset = MarsDataset(data, labels)
    assert len(train_dataset) == 1000

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
    )

    model = nn.parallel.DistributedDataParallel(get_model())
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.BCELoss()

    for i in range(2):
        # 2 epochs
        train_sampler.set_epoch(i)
        running_loss = 0.0
        for _, (batch_data, batch_labels) in enumerate(train_loader):
            outputs = model(batch_data)
            loss = criterion(outputs.squeeze(), batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"running_loss is {loss.item()}")

    print("Done!")


if __name__ == "__main__":
    assert len(sys.argv) == 2
    assert sys.argv[1] == "multiple"
    feature_data = globals()["feature_data"]
    labels = globals()["labels"]
    main(feature_data, labels)
