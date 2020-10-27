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
    import pyproxima2 as proxima
except ImportError:  # pragma: no cover
    proxima = None

from ... import tensor as mt


available_numpy_dtypes = [
    np.dtype(np.float16),
    np.dtype(np.float32),
    np.dtype(np.int8),
    np.dtype(np.int16),
]

if proxima:
    _proxima_types = [
        proxima.IndexMeta.FT_FP16,
        proxima.IndexMeta.FT_FP32,
        proxima.IndexMeta.FT_INT8,
        proxima.IndexMeta.FT_INT16,
    ]
    assert len(_proxima_types) == len(available_numpy_dtypes)
    _type_mapping = {numpy_dtype: proxima_type
                     for numpy_dtype, proxima_type
                     in zip(available_numpy_dtypes, _proxima_types)}


def validate_tensor(tensor):
    if hasattr(tensor, 'to_tensor'):
        tensor = tensor.to_tensor()
    else:
        tensor = mt.tensor(tensor)
    if tensor.ndim != 2:
        raise ValueError('Input tensor should be 2-d')
    return tensor


def get_proxima_type(np_dtype):
    try:
        return _type_mapping[np_dtype]
    except KeyError:
        raise TypeError(f"Does not support {np_dtype}, available types include "
                        f"{', '.join(t.name for t in _type_mapping)}")
