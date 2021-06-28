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

import numpy as np

from ..utils import get_order


def flatten(a, order='C'):
    """
    Return a copy of the tensor collapsed into one dimension.

    Parameters
    ----------
    order : {'C', 'F', 'A', 'K'}, optional
        'C' means to flatten in row-major (C-style) order.
        'F' means to flatten in column-major (Fortran-
        style) order. 'A' means to flatten in column-major
        order if `a` is Fortran *contiguous* in memory,
        row-major order otherwise. 'K' means to flatten
        `a` in the order the elements occur in memory.
        The default is 'C'.

    Returns
    -------
    y : Tensor
        A copy of the input tensor, flattened to one dimension.

    See Also
    --------
    ravel : Return a flattened tensor.
    flat : A 1-D flat iterator over the tensor.

    Examples
    --------

    >>> import mars.tensor as mt

    >>> a = mt.array([[1,2], [3,4]])
    >>> a.flatten().execute()
    array([1, 2, 3, 4])
    """

    from ..reshape.reshape import TensorReshape, calc_shape

    if np.isnan(sum(a.shape)):
        raise ValueError(f'tensor shape is unknown, {a.shape}')

    new_shape = calc_shape(a.size, -1)
    tensor_order = get_order(order, a.order)
    op = TensorReshape(new_shape, dtype=a.dtype, create_view=False)
    return op(a, order=tensor_order, out_shape=new_shape)
