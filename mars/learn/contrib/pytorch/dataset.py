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

import os

try:
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    Dataset = object

from ....context import get_context, DistributedContext
from ....tensor.indexing.core import process_index
from ....tensor.fetch import TensorFetch
from ....utils import require_not_none


@require_not_none(Dataset)
class MarsDataset(Dataset):
    def __init__(self, *names):
        self._context = get_context()

        tensors = []
        for name in names:
            tileable_key = self._context.get_tileable_key_by_name(name)
            nsplits = self._context.get_tileable_metas([tileable_key], filter_fields=['nsplits'])[0][0]
            shape = tuple(sum(s) for s in nsplits)
            tensors.append(TensorFetch().new_tensor([], shape=shape, _key=tileable_key))
        self.tensors = tensors

    def __len__(self):
        return self.tensors[0].shape[0]

    def __getitem__(self, item):
        indexes = process_index(self.tensors[0].ndim, item)
        return tuple(self._context.get_tileable_data(t.key, indexes) for t in self.tensors)


def enter_mars_context():
    scheduler = os.environ['MARS_SCHEDULER']
    session_id = os.environ['MARS_SESSION']
    return DistributedContext(scheduler_address=scheduler, session_id=session_id)


