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


def ndim(a):
    """
    Return the number of dimensions of a tensor.

    Parameters
    ----------
    a : array_like
        Input tebsir.  If it is not already a tensor, a conversion is
        attempted.

    Returns
    -------
    number_of_dimensions : int
        The number of dimensions in `a`.  Scalars are zero-dimensional.

    See Also
    --------
    ndarray.ndim : equivalent method
    shape : dimensions of tensor
    Tensor.shape : dimensions of tensor

    Examples
    --------
    >>> import mars.tensor as mt
    >>> mt.ndim([[1,2,3],[4,5,6]])
    2
    >>> mt.ndim(mt.array([[1,2,3],[4,5,6]]))
    2
    >>> mt.ndim(1)
    0

    """
    from ..datasource import asarray

    try:
        return a.ndim
    except AttributeError:
        return asarray(a).ndim
