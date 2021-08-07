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

from mars.core.entity.tileables import Tileable
import numpy as np
import pandas as pd
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  
    torch = None
    Dataset = object

from ....core.context import get_context
from ....tensor.core import TENSOR_TYPE
from ....dataframe.core import DATAFRAME_TYPE, SERIES_TYPE
from ....utils import require_not_none


@require_not_none(torch)
class MarsDataset(Dataset):
    def __init__(self, *tileables):

        self._context = get_context() 
        self._tileables = tileables

    def __len__(self):
        return self._tileables[0].shape[0]

    def __getitem__(self, item):
        return tuple(self.get_data(t, item) for t in self._tileables)

    def get_data(self, t, index):
        if isinstance(t, TENSOR_TYPE):
            return t[index].execute().fetch()
        elif isinstance(t, np.ndarray):
            return t[index]
        elif isinstance(t, DATAFRAME_TYPE):
            return t.iloc[index].execute().fetch().values
        elif isinstance(t, SERIES_TYPE):
            return t[index].execute().fetch()
        elif isinstance(t, pd.DataFrame):
            return t.iloc[index].values
        elif isinstance(t, pd.Series):
            return t[index]
