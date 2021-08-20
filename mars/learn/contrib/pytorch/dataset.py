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

import copy
from typing import List

import numpy as np
import pandas as pd
try:
    import torch
    from torch.utils.data import Dataset
except ImportError: # pragma: no cover
    torch = None
    Dataset = object

from .... import execute
from ....core.context import get_context
from ....tensor.core import TENSOR_TYPE
from ....dataframe.core import DATAFRAME_TYPE, SERIES_TYPE
from ....utils import require_not_none


ACCEPT_TYPE = (TENSOR_TYPE, DATAFRAME_TYPE, SERIES_TYPE,
               np.ndarray, pd.DataFrame, pd.Series, List)


@require_not_none(torch)
class MarsDataset(Dataset):
    r"""MarsDataset that inherit from torch.utils.data.Dataset.
    It converts from Mars basic datatype such as Tensor,
    DataFrame, Series. Additionally, it's constructor can receive
    np.ndarray, pd.DataFrame, pd.Series type.
    """
    def __init__(self, *tileables, fetch_kwargs=None):

        self._context = get_context()
        self._tileables = tileables
        self._fetch_kwargs = fetch_kwargs or dict()
        self._executed = False
        self._check_type()

    def _check_type(self):
        for t in self._tileables:
            if not isinstance(t, ACCEPT_TYPE):
                raise TypeError(f"Unexpected dataset type: {type(t)}")

    def _execute(self):
        execute_data = [t for t in self._tileables
                        if isinstance(t, ACCEPT_TYPE[:3])]
        if len(execute_data):
            execute(execute_data)

    def __len__(self):
        return self._tileables[0].shape[0]

    def __getitem__(self, index):
        if not self._executed:
            self._execute()
            self._executed = True
        return tuple(self.get_data(t, index) for t in self._tileables)

    def get_data(self, t, index):
        fetch_kwargs = dict()
        if self._fetch_kwargs:
            fetch_kwargs = copy.deepcopy(self._fetch_kwargs)

        if isinstance(t, TENSOR_TYPE):
            return t[index].fetch(**fetch_kwargs)
        elif isinstance(t, np.ndarray):
            return t[index]
        elif isinstance(t, DATAFRAME_TYPE):
            return t.iloc[index].fetch(**fetch_kwargs).values
        elif isinstance(t, SERIES_TYPE):
            return t.iloc[index].fetch(**fetch_kwargs)
        elif isinstance(t, pd.DataFrame):
            return t.iloc[index].values
        elif isinstance(t, pd.Series):
            return t.iloc[index]
        else:
            return t[index]
