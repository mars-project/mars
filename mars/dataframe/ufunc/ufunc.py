# Copyright 1999-2021 Alibaba Group Holding Ltd.
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

from numbers import Number

from ...tensor import tensor as astensor
from ...tensor.ufunc.ufunc import UFUNC_TO_TENSOR_FUNC
from ..core import DATAFRAME_TYPE, SERIES_TYPE


def _check_arg(arg):
    if isinstance(arg, Number):
        return True

    if isinstance(arg, (DATAFRAME_TYPE, SERIES_TYPE)):
        return True

    try:
        astensor(arg)
        return True
    except ValueError:
        return False


def _array_ufunc(_, ufunc, method, *inputs, **kwargs):
    out = kwargs.get('out', tuple())
    for x in inputs + out:
        if not _check_arg(x):
            return NotImplemented

    if method == '__call__':
        if ufunc.signature is not None:
            return NotImplemented
        if ufunc not in UFUNC_TO_TENSOR_FUNC:
            return NotImplemented

        # we delegate numpy ufunc to tensor ufunc,
        # tensor ufunc will handle Mars DataFrame properly.
        tensor_func = UFUNC_TO_TENSOR_FUNC[ufunc]
        return tensor_func(*inputs, **kwargs)

    return NotImplemented
