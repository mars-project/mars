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

import asyncio
import os
import threading

import numpy as np
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object

from ....context import get_context, DistributedContext, RunningMode
from ....tensor.core import TENSOR_TYPE
from ....tensor.indexing.core import process_index
from ....dataframe.indexing.iloc import process_iloc_indexes
from ....utils import require_not_none, wait_results


@require_not_none(torch)
class MarsDataset(Dataset):
    _loop_local = threading.local()

    def __init__(self, *tileables):
        from ....session import Session

        self._context = get_context() or Session.default_or_local().context

        self.tileables = tileables
        self._datas = None
        self._offset = 0

    @property
    def _loop(self):
        try:
            return self._loop_local.loop
        except AttributeError:
            loop = self._loop_local.loop = asyncio.get_event_loop()
            return loop

    def prefetch(self, indices):
        self._datas = self._get_data(indices)
        self._offset = 0

    @staticmethod
    def _process_index(t, index):
        if isinstance(t, TENSOR_TYPE):
            return process_index(t.ndim, index)
        else:
            return process_iloc_indexes(t, index)

    async def _get_data_async(self, item):
        if self._context.running_mode != RunningMode.distributed:
            coros = [t[item].fetch_async() for t in self.tileables]
        else:
            coros = [self._context.get_tileable_data(
                t.key, process_index(t.ndim, item)) for t in self.tileables]
        return tuple((await wait_results(coros))[0])

    def _get_data(self, item):
        return self._loop.run_until_complete(self._get_data_async(item))

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


def enter_mars_context():
    scheduler = os.environ['MARS_SCHEDULER']
    session_id = os.environ['MARS_SESSION']
    return DistributedContext(scheduler_address=scheduler, session_id=session_id)
