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
from typing import List, Tuple

import pandas as pd
import numpy as np

from .... import execute
from ....core.context import get_context
from ....tensor.core import TENSOR_TYPE
from ....dataframe.core import DATAFRAME_TYPE, SERIES_TYPE
from ....utils import require_not_none, lazy_import

tf = lazy_import("tensorflow")


ACCEPT_TYPE = (
    TENSOR_TYPE,
    DATAFRAME_TYPE,
    SERIES_TYPE,
    np.ndarray,
    pd.DataFrame,
    pd.Series,
    List,
)


@require_not_none(tf)
class MarsDataset:
    def __init__(
        self,
        tensors,
        output_shapes: Tuple[int, ...] = None,
        output_types: Tuple[np.dtype, ...] = None,
        fetch_kwargs=None,
    ):

        self._context = get_context()
        self._tensors = tensors
        self._output_shapes = output_shapes
        self._output_types = output_types
        self._fetch_kwargs = fetch_kwargs or dict()
        self._executed = False
        self._check_and_convert()

    def _check_and_convert(self):
        if not isinstance(self._tensors, Tuple):
            self._tensors = (self._tensors,)
        for t in self._tensors:
            if not isinstance(t, ACCEPT_TYPE):
                raise TypeError(f"Unexpected dataset type: {type(t)}")

        if not self._executed:
            self._execute()
            self._executed = True

        if not self._output_shapes:
            get_shape = (
                lambda t: tuple(())
                if isinstance(t, (List, SERIES_TYPE, pd.Series))
                else t.shape[1:]
            )
            self._output_shapes = (
                get_shape(self._tensors[0])
                if len(self._tensors) == 1
                else tuple(get_shape(t) for t in self._tensors)
            )

        if not self._output_types:
            get_type = (
                lambda t: type(t[0])
                if isinstance(t, List)
                else t[0].dtype
                if isinstance(t, (DATAFRAME_TYPE, pd.DataFrame))
                else t.dtype
            )
            self._output_types = (
                get_type(self._tensors[0])
                if len(self._tensors) == 1
                else tuple(tf.as_dtype(get_type(t)) for t in self._tensors)
            )

    def _execute(self):  # pragma: no cover
        execute_data = [t for t in self._tensors if isinstance(t, ACCEPT_TYPE[:3])]

        if len(execute_data) > 0:
            execute(execute_data)

    def get_data(self, t, index):  # pragma: no cover
        # coverage not included as now there is no solution to cover tensorflow methods
        # see https://github.com/tensorflow/tensorflow/issues/33759 for more details.
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

        def make_generator():  # pragma: no cover
            if not self._executed:
                self._execute()
                self._executed = True

            for i in range(len(self._tensors[0])):
                if len(self._tensors) == 1:
                    yield self.get_data(self._tensors[0], i)
                else:
                    yield tuple(self.get_data(t, i) for t in self._tensors)

        return tf.data.Dataset.from_generator(
            make_generator,
            output_types=self._output_types,
            output_shapes=self._output_shapes,
        )


def gen_tensorflow_dataset(
    tensors,
    output_shapes: Tuple[int, ...] = None,
    output_types: Tuple[np.dtype, ...] = None,
    fetch_kwargs=None,
):
    """
    convert mars data type to tf.data.Dataset. Note this is based tensorflow 2.0
    For example
    -----------
    >>> # convert a tensor to tf.data.Dataset.
    >>> data = mt.tensor([[1, 2], [3, 4]])
    >>> dataset = gen_tensorflow_dataset(data)
    >>> list(dataset.as_numpy_iterator())
    [array([1, 2]), array([3, 4])]
    >>> dataset.element_spec
    TensorSpec(shape=(2,), dtype=tf.int64, name=None)

    >>> # convert a tuple of tensors to tf.data.Dataset.
    >>> data1 = mt.tensor([1, 2]); data2 = mt.tensor([3, 4]); data3 = mt.tensor([5, 6])
    >>> dataset = gen_tensorflow_dataset((data1, data2, data3))
    >>> list(dataset.as_numpy_iterator())
    [(1, 3, 5), (2, 4, 6)]

    Parameters
    ----------
    tensors: Mars data type or a tuple consisting of Mars data type
        the data that convert to tf.data.dataset
    output_shapes:
        A (nested) structure of `tf.TensorShape` objects corresponding to
        each component of an element yielded from mars object.
    output_types:
        A (nested) structure of `tf.DType` objects corresponding to each
        component of an element yielded from mars object.
    fetch_kwargs:
        the parameters of mars object executes fetch() operation.
    Returns
    -------
        tf.data.Dataset
    """
    mars_dataset = MarsDataset(
        tensors,
        output_shapes=output_shapes,
        output_types=output_types,
        fetch_kwargs=fetch_kwargs,
    )

    return mars_dataset.to_tf()
