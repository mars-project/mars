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
from typing import Generator, Iterator, Sized

try:
    import torch
    from torch.utils.data import Sampler
except ImportError:  # pragma: no cover
    torch = None
    Sampler = object

from ....utils import require_not_none


@require_not_none(torch)
class MarsSequentialSampler(Sampler):
    r""""Note: recommend to use torch original API SequentialSampler. 
    Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source: Sized

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class MarsRandomSampler(Sampler):
    r""""Note: recommend to use torch original API RandomSampler. 
    Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source, replacement=False, num_samples=None, generator=None):

        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):  
            raise ValueError("replacement should be a boolean value, but got "
                             f"replacement={self.replacement}")

        if self._num_samples is not None and not replacement:  
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:  
            raise ValueError("num_samples should be a positive integer "
                             f"value, but got num_samples={self.num_samples}")

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator
        if self.replacement:  
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            yield from torch.randperm(n, generator=generator).tolist()

    def __len__(self) -> int:
        return self.num_samples
