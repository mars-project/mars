# Copyright 1999-2018 Alibaba Group Holding Ltd.
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


def flatten(a):
    """
    Return a copy of the tensor collapsed into one dimension.

    Parameters
    ----------

    Returns
    -------
    y : Tensor
        A copy of the input tensor, flattened to one dimension.

    See Also
    --------
    ravel : Return a flattened array.
    flat : A 1-D flat iterator over the array.

    Examples
    --------

    >>> import mars.tensor as mt

    >>> a = mt.array([[1,2], [3,4]])
    >>> a.flatten().execute()
    array([1, 2, 3, 4])
    """

    from ..reshape.reshape import TensorReshape, calc_shape

    if np.isnan(sum(a.shape)):
        raise ValueError('tensor shape is unknown, {0}'.format(a.shape))

    new_shape = calc_shape(a.size, -1)
    op = TensorReshape(new_shape, dtype=a.dtype, create_view=False)
    return op(a)
