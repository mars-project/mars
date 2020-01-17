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

try:
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    Dataset = None

from ....context import get_context
from ....tensor.indexing.core import process_index


MarsTorchDataset = None
if Dataset:
    class MarsTorchDataset(Dataset):
        def __init__(self, *tensors):
            self._context = get_context()
            self.tensors = tensors

        def __len__(self):
            return self.tensors[0].shape[0]

        def __getitem__(self, item):
            indexes = process_index(self.tensors[0].ndim, item)
            return tuple(self._context.get_tileable_data(t.key, indexes) for t in self.tensors)


