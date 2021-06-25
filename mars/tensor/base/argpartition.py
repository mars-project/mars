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

from .partition import _validate_partition_arguments, TensorPartition


def argpartition(a, kth, axis=-1, kind='introselect', order=None, **kw):
    """
    Perform an indirect partition along the given axis using the
    algorithm specified by the `kind` keyword. It returns an array of
    indices of the same shape as `a` that index data along the given
    axis in partitioned order.

    .. versionadded:: 1.8.0

    Parameters
    ----------
    a : array_like
        Tensor to sort.
    kth : int or sequence of ints
        Element index to partition by. The k-th element will be in its
        final sorted position and all smaller elements will be moved
        before it and all larger elements behind it. The order all
        elements in the partitions is undefined. If provided with a
        sequence of k-th it will partition all of them into their sorted
        position at once.
    axis : int or None, optional
        Axis along which to sort. The default is -1 (the last axis). If
        None, the flattened tensor is used.
    kind : {'introselect'}, optional
        Selection algorithm. Default is 'introselect'
    order : str or list of str, optional
        When `a` is a tensor with fields defined, this argument
        specifies which fields to compare first, second, etc. A single
        field can be specified as a string, and not all fields need be
        specified, but unspecified fields will still be used, in the
        order in which they come up in the dtype, to break ties.

    Returns
    -------
    index_tensor : Tensor, int
        Tensor of indices that partition `a` along the specified axis.
        If `a` is one-dimensional, ``a[index_tensor]`` yields a partitioned `a`.
        More generally, ``np.take_along_axis(a, index_tensor, axis=a)`` always
        yields the partitioned `a`, irrespective of dimensionality.

    See Also
    --------
    partition : Describes partition algorithms used.
    Tensor.partition : Inplace partition.
    argsort : Full indirect sort

    Notes
    -----
    See `partition` for notes on the different selection algorithms.

    Examples
    --------
    One dimensional tensor:

    >>> import mars.tensor as mt
    >>> x = mt.array([3, 4, 2, 1])
    >>> x[mt.argpartition(x, 3)].execute()
    array([2, 1, 3, 4])
    >>> x[mt.argpartition(x, (1, 3))].execute()
    array([1, 2, 3, 4])

    >>> x = [3, 4, 2, 1]
    >>> mt.array(x)[mt.argpartition(x, 3)].execute()
    array([2, 1, 3, 4])

    """
    a, kth, axis, kind, order, need_align = _validate_partition_arguments(
        a, kth, axis, kind, order, kw)
    op = TensorPartition(kth=kth, axis=axis, kind=kind, order=order,
                         need_align=need_align, return_value=False,
                         return_indices=True, dtype=a.dtype, gpu=a.op.gpu)
    return op(a, kth)
