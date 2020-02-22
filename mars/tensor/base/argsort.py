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

from .sort import _validate_sort_arguments, TensorSort


def argsort(a, axis=-1, kind=None, parallel_kind=None, psrs_kinds=None, order=None):
    """
    Returns the indices that would sort a tensor.

    Perform an indirect sort along the given axis using the algorithm specified
    by the `kind` keyword. It returns a tensor of indices of the same shape as
    `a` that index data along the given axis in sorted order.

    Parameters
    ----------
    a : array_like
        Tensor to sort.
    axis : int or None, optional
        Axis along which to sort.  The default is -1 (the last axis). If None,
        the flattened tensor is used.
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
        Sorting algorithm. The default is 'quicksort'. Note that both 'stable'
        and 'mergesort' use timsort under the covers and, in general, the
        actual implementation will vary with data type. The 'mergesort' option
        is retained for backwards compatibility.

        .. versionchanged:: 1.15.0.
           The 'stable' option was added.
    order : str or list of str, optional
        When `a` is a tensor with fields defined, this argument specifies
        which fields to compare first, second, etc.  A single field can
        be specified as a string, and not all fields need be specified,
        but unspecified fields will still be used, in the order in which
        they come up in the dtype, to break ties.

    Returns
    -------
    index_tensor : Tensor, int
        Tensor of indices that sort `a` along the specified `axis`.
        If `a` is one-dimensional, ``a[index_tensor]`` yields a sorted `a`.
        More generally, ``np.take_along_axis(a, index_tensor, axis=axis)``
        always yields the sorted `a`, irrespective of dimensionality.

    See Also
    --------
    sort : Describes sorting algorithms used.
    lexsort : Indirect stable sort with multiple keys.
    Tensor.sort : Inplace sort.
    argpartition : Indirect partial sort.

    Notes
    -----
    See `sort` for notes on the different sorting algorithms.

    Examples
    --------
    One dimensional tensor:

    >>> import mars.tensor as mt
    >>> x = mt.array([3, 1, 2])
    >>> mt.argsort(x).execute()
    array([1, 2, 0])

    Two-dimensional tensor:

    >>> x = mt.array([[0, 3], [2, 2]])
    >>> x.execute()
    array([[0, 3],
           [2, 2]])

    >>> ind = mt.argsort(x, axis=0)  # sorts along first axis (down)
    >>> ind.execute()
    array([[0, 1],
           [1, 0]])
    #>>> mt.take_along_axis(x, ind, axis=0).execute()  # same as np.sort(x, axis=0)
    #array([[0, 2],
    #       [2, 3]])

    >>> ind = mt.argsort(x, axis=1)  # sorts along last axis (across)
    >>> ind.execute()
    array([[0, 1],
           [0, 1]])
    #>>> mt.take_along_axis(x, ind, axis=1).execute()  # same as np.sort(x, axis=1)
    #array([[0, 3],
    #       [2, 2]])

    Indices of the sorted elements of a N-dimensional array:

    >>> ind = mt.unravel_index(mt.argsort(x, axis=None), x.shape)
    >>> ind.execute9)
    (array([0, 1, 1, 0]), array([0, 0, 1, 1]))
    >>> x[ind].execute()  # same as np.sort(x, axis=None)
    array([0, 2, 2, 3])

    Sorting with keys:

    >>> x = mt.array([(1, 0), (0, 1)], dtype=[('x', '<i4'), ('y', '<i4')])
    >>> x.execute()
    array([(1, 0), (0, 1)],
          dtype=[('x', '<i4'), ('y', '<i4')])

    >>> mt.argsort(x, order=('x','y')).execute()
    array([1, 0])

    >>> mt.argsort(x, order=('y','x')).execute()
    array([0, 1])

    """
    a, axis, kind, parallel_kind, psrs_kinds, order = _validate_sort_arguments(
        a, axis, kind, parallel_kind, psrs_kinds, order)

    op = TensorSort(axis=axis, kind=kind, parallel_kind=parallel_kind,
                    order=order, psrs_kinds=psrs_kinds,
                    return_value=False, return_indices=True,
                    dtype=a.dtype, gpu=a.op.gpu)
    return op(a)
