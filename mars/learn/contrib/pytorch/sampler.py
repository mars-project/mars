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


import math

try:
    import torch
    from torch.utils.data import Sampler
except ImportError:  # pragma: no cover
    torch = None
    Sampler = object

from ....utils import require_not_none


@require_not_none(torch)
class MarsDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        import torch.distributed as dist

        super().__init__(dataset)
        if num_replicas is None:  # pragma: no cover
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:  # pragma: no cover
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def generate_indices(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:  # pragma: no cover
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        self.dataset.prefetch(indices)
        return indices

    def __iter__(self):
        return iter(self.generate_indices())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class MarsRandomSampler(Sampler):
    def __init__(self, data_source, replacement=False, num_samples=None):
        super().__init__(data_source)

        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples

        if not isinstance(self.replacement, bool):  # pragma: no cover
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:  # pragma: no cover
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:  # pragma: no cover
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        else:  # pragma: no cover
            return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:  # pragma: no cover
            indices = torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist()
            self.data_source.prefetch(indices)
            return iter(indices)
        else:
            indices = torch.randperm(n).tolist()
            self.data_source.prefetch(indices)
            return iter(indices)

    def __len__(self):
        return self.num_samples
