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

from numpy.lib.arraysetops import isin
from mars.core.entity.tileables import Tileable
from mars.core.entity import output_types
import numpy as np
try:
    import tensorflow as tf
    # from tensorflow.python.data.ops.dataset_ops import DatasetV2
    # from tensorflow.python.data.util import structure
    # from tensorflow.python.ops import gen_dataset_ops
except ImportError:  
    tensorflow = None
    Dataset = object

from ....core.context import get_context
from ....tensor.core import TENSOR_TYPE
from ....tensor.indexing.core import process_index
from ....dataframe.indexing.iloc import process_iloc_indexes
from ....utils import require_not_none
from ....core.entity import Tileable
from typing import Any, List, Optional


@require_not_none(tf)
class MarsDataset:
    """A TFMarsDataset which converted from mars tensor

    Args:
        feature_data(Mars tensor): feature data
        label_data(Mars tensor): label data
        feature_columns (List[Any]): the feature columns' name
        feature_shapes (Optional[List[tf.TensorShape]]): the shape for each
            feature. If provide, it should match the size of feature_columns
        feature_types (Optional[List[tf.DType]]): the data type for each
            feature. If provide, it should match the size of feature_columns
        label_column (Any): the label column name
        label_shape (Optional[tf.TensorShape]): the shape for the label data
        label_type (Optional[tf.DType]): the data type for the label data
    """

    def __init__(self, feature_data, label_data, 
                 feature_shapes: Optional[List[tf.TensorShape]] = None,
                 feature_types: Optional[List[tf.DType]] = None,
                 label_shape: Optional[tf.TensorShape] = None,
                 label_type: Optional[tf.DType] = None
                 ):

        self._context = get_context() 
        if isinstance(feature_data, Tileable):
            feature_data.execute()
        self._feature_data = feature_data.fetch()
        if isinstance(label_data, Tileable):
            label_data.execute()
        self._label_data = label_data.fetch()
        self._feature_shapes = feature_shapes
        self._feature_types = feature_types
        self._label_shape = label_shape
        self._label_type = label_type

        self._check_and_convert()

    def _check_and_convert(self):
        if not self._feature_shapes:
            self._feature_shapes = tf.TensorShape(self._feature_data[0].shape)
            print("self._feature_shapes", self._feature_shapes)

        if not self._feature_types:
            self._feature_types = tf.as_dtype(self._feature_data.dtype)

        if not self._label_shape:
            self._label_shape = tf.TensorShape(self._label_data[0].shape)

        if not self._label_type:
            self._label_type = tf.as_dtype(self._label_data.dtype)


    def get_tfdataset(self,
                      output_types=None,
                      output_shapes=None,
                      args=None,
                      output_signature=None
                      ) -> "tf.data.Dataset":
        
        """Get TF Dataset.

        convert into a tensorflow.data.Dataset
        """
        
        def make_generator():
            for i in range(len(self._feature_data)):
                features = self._feature_data[i]
                label = self._label_data[i]
                if len(features) > 1:
                    yield tuple(features), label
                else:
                    yield features, label

        if not output_shapes:
            output_shapes = (self._feature_shapes, self._label_shape)
            print(output_shapes)

        if not output_types:
            output_types = (self._feature_types, self._label_type)
            print(output_types)

        return tf.data.Dataset.from_generator(
            make_generator,
            output_types=output_types,
            output_shapes=output_shapes,
            args=args,
            output_signature=output_signature
        )
            







# @require_not_none(tf)
# class MarsTFDataset(DatasetV2):
#     def __init__(self, tileables):
#         self._context = get_context() 
#         self.tileables = tileables
#         element = tileables.fetch()
#         element = structure.normalize_element(element)
#         self._structure = structure.type_spec_from_value(element)
#         self._tensors = structure.to_tensor_list(self._structure, element)

#         variant_tensor = gen_dataset_ops.tensor_dataset(
#             self._tensors,
#             output_shapes=structure.get_flat_tensor_shapes(self._structure))
        
#         super(MarsTFDataset, self).__init__(variant_tensor)
    
#     def _inputs(self):
#         return []
    
#     @property
#     def element_spec(self):
#         return self._structure

#     @staticmethod
#     def from_marstensors(tensors):
#         """Creates a `Dataset` with a single element, comprising the given tensors.

#         `from_tensors` produces a dataset containing only a single element. To slice
#         the input tensor into multiple elements, use `from_tensor_slices` instead.

#         >>> dataset = tf.data.Dataset.from_tensors([1, 2, 3])
#         >>> list(dataset.as_numpy_iterator())
#         [array([1, 2, 3], dtype=int32)]
#         >>> dataset = tf.data.Dataset.from_tensors(([1, 2, 3], 'A'))
#         >>> list(dataset.as_numpy_iterator())
#         [(array([1, 2, 3], dtype=int32), b'A')]

#         >>> # You can use `from_tensors` to produce a dataset which repeats
#         >>> # the same example many times.
#         >>> example = tf.constant([1,2,3])
#         >>> dataset = tf.data.Dataset.from_tensors(example).repeat(2)
#         >>> list(dataset.as_numpy_iterator())
#         [array([1, 2, 3], dtype=int32), array([1, 2, 3], dtype=int32)]

#         Note that if `tensors` contains a NumPy array, and eager execution is not
#         enabled, the values will be embedded in the graph as one or more
#         `tf.constant` operations. For large datasets (> 1 GB), this can waste
#         memory and run into byte limits of graph serialization. If `tensors`
#         contains one or more large NumPy arrays, consider the alternative described
#         in [this
#         guide](https://tensorflow.org/guide/data#consuming_numpy_arrays).

#         Args:
#         tensors: A dataset "element". Supported values are documented
#             [here](https://www.tensorflow.org/guide/data#dataset_structure).

#         Returns:
#         Dataset: A `Dataset`.
#         """
#         return MarsTFDataset(tensors)


