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
import pandas as pd
import numpy as np
try:
    import tensorflow as tf
except ImportError: # pragma: no cover
    tensorflow = None
    Dataset = object

from .... import execute
from ....core.context import get_context
from ....tensor.core import TENSOR_TYPE
from ....dataframe.core import DATAFRAME_TYPE, SERIES_TYPE
from ....utils import require_not_none
from typing import List


ACCEPT_TYPE = (TENSOR_TYPE, DATAFRAME_TYPE, SERIES_TYPE,
               np.ndarray, pd.DataFrame, pd.Series, List)


@require_not_none(tf)
class MarsDataset:
    def __init__(self, *tileable,
                 output_shapes = None,
                 output_types = None,
                 fetch_kwargs=None):

        self._context = get_context()
        self._tileable = tileable
        self._output_shapes = output_shapes
        self._output_types = output_types
        self._fetch_kwargs = fetch_kwargs or dict()
        self._executed = False
        self._check_and_convert()

    def _check_and_convert(self):
        for t in self._tileable:
            if not isinstance(t, ACCEPT_TYPE):
                raise TypeError(f"Unexpected dataset type: {type(t)}")

        if not self._output_shapes:
            get_shape = lambda t: tuple(()) if isinstance(t, (List, SERIES_TYPE, pd.Series)) \
                                  else t.shape[1:]
            self._output_shapes = tuple(get_shape(t) for t in self._tileable)

        if not self._output_types:
            get_type = lambda t: type(t[0]) if isinstance(t, List) else \
                                 t[0].dtype if isinstance(t, (DATAFRAME_TYPE, pd.DataFrame)) \
                                 else t.dtype
            self._output_types = tuple(tf.as_dtype(get_type(t)) for t in self._tileable)

    def _execute(self):
        execute_data = [t for t in self._tileable if isinstance(t, ACCEPT_TYPE[:3])]

        if len(execute_data) > 0:
            execute(execute_data)

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

    def to_tf(self) -> "tf.data.Dataset":
        """Get TF Dataset.

        convert into a tensorflow.data.Dataset
        """
        def make_generator():
            if not self._executed:
                self._execute()
                self._executed = True

            for i in range(len(self._tileable[0])):
                # print(f"len(self._tileable[0]): {len(self._tileable[0])}")
                # print(f"len(self._tileable): {len(self._tileable)}")
                # print(tuple(self.get_data(t, i) for t in self._tileable))
                yield tuple(self.get_data(t, i) for t in self._tileable)

        return tf.data.Dataset.from_generator(
            make_generator,
            output_types=self._output_types,
            output_shapes=self._output_shapes
        )


def get_tfdataset(*tileable,
                 output_shapes = None,
                 output_types = None,
                 fetch_kwargs=None):
    """
    convert mars data type to tf.data.Dataset. Note this is based tensorflow 2.0

    Parameters
    ----------
    tileable: Mars data type
        the data that convert to tf.data.dataset
    output_shapes:
        A (nested) structure of `tf.TensorShape` objects corresponding to
        each component of an element yielded from mars object.
    output_types:
        A (nested) structure of `tf.DType` objects corresponding to each
        component of an element yielded from mars object.
    fetch_kwargs:
        the parameters of mars object executes fetch() operation.
    """
    mars_dataset = MarsDataset(*tileable, output_shapes=output_shapes,
                               output_types=output_types, fetch_kwargs=fetch_kwargs)

    return mars_dataset.to_tf()