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
    r"""MarsDataset that inherit from torch.utils.data.Dataset.
    It converts from Mars basic datatype such as Tensor,
    DataFrame, Series. Additionally, it's constructor can receive
    np.ndarray, pd.DataFrame, pd.Series type.
    """
    def __init__(self, *tileables):

        self._context = get_context()
        self._tileables = tileables

        self._check_and_execute()
        
    def _check_and_execute(self):
        for t in self._tileables:
            if isinstance(t, TENSOR_TYPE):
                t.execute()
            elif isinstance(t, DATAFRAME_TYPE):
                t.execute()
            elif isinstance(t, SERIES_TYPE):
                t.execute()

    def __len__(self):
        return self._tileables[0].shape[0]

    def __getitem__(self, index):
        return tuple(self.get_data(t, index) for t in self._tileables)

    @staticmethod
    def get_data(t, index):
        if isinstance(t, TENSOR_TYPE):
            return t[index].fetch()
        elif isinstance(t, np.ndarray):
            return t[index]
        elif isinstance(t, DATAFRAME_TYPE):
            return t.iloc[index:index+1].fetch().values[0]
        elif isinstance(t, SERIES_TYPE):
            return t.iloc[index].fetch()
        elif isinstance(t, pd.DataFrame):
            return t.iloc[index].values
        elif isinstance(t, pd.Series):
            return t.iloc[index]
