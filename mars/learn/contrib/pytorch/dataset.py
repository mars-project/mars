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

import numpy as np
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object

from ....context import get_context, RunningMode
from ....tensor.core import TENSOR_TYPE
from ....tensor.indexing.core import process_index
from ....dataframe.indexing.iloc import process_iloc_indexes
from ....utils import require_not_none


@require_not_none(torch)
class MarsDataset(Dataset):
    def __init__(self, *tileables):
        from ....session import Session

        self._context = get_context() or Session.default_or_local().context

        self.tileables = tileables
        self._datas = None
        self._offset = 0

    def prefetch(self, indices):
        self._datas = self._get_data(indices)
        self._offset = 0

    @staticmethod
    def _process_index(t, index):
        if isinstance(t, TENSOR_TYPE):
            return process_index(t.ndim, index)
        else:
            return process_iloc_indexes(t, index)

    def _get_data(self, item):
        if self._context.running_mode != RunningMode.distributed:
            return tuple(t[item].fetch() for t in self.tileables)
        else:
            return tuple(self._context.get_tileable_data(
                t.key, self._process_index(t, item)) for t in self.tileables)

    def __len__(self):
        return self.tileables[0].shape[0]

    def __getitem__(self, item):
        if self._datas is not None:
            ret = []
            for data in self._datas:
                if isinstance(data, np.ndarray):
                    ret.append(data[self._offset])
                else:
                    ret.append(data.iloc[self._offset])
            self._offset += 1
            return ret
        else:
            return self._get_data(item)
