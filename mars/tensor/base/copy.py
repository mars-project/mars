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


def copy(a, order='K'):
    """
    Return a tensor copy of the given object.

    Parameters
    ----------
    a : array_like
        Input data.
    order : {'C', 'F', 'A', 'K'}, optional
        Controls the memory layout of the copy. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible. (Note that this function and :meth:`ndarray.copy` are very
        similar, but have different default values for their order=
        arguments.)

    Returns
    -------
    arr : Tensor
        Tensor interpretation of `a`.

    Notes
    -----
    This is equivalent to:

    >>> import mars.tensor as mt

    >>> mt.array(a, copy=True)  #doctest: +SKIP

    Examples
    --------
    Create an array x, with a reference y and a copy z:

    >>> x = mt.array([1, 2, 3])
    >>> y = x
    >>> z = mt.copy(x)

    Note that, when we modify x, y changes, but not z:

    >>> x[0] = 10
    >>> (x[0] == y[0]).execute()
    True
    >>> (x[0] == z[0]).execute()
    False

    """
    from ..datasource import array

    return array(a, order=order, copy=True)
