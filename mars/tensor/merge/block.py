#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import functools
import itertools
import operator
import numpy as np

from ..datasource.empty import empty
from ..datasource.array import array

# Internal functions to eliminate the overhead of repeated dispatch in one of
# the two possible paths inside mt.block.
# Use getattr to protect against __array_function__ being disabled.
_size = getattr(np.size, '__wrapped__', np.size)
_ndim = getattr(np.ndim, '__wrapped__', np.ndim)


def _block_format_index(index):
    """
    Convert a list of indices ``[0, 1, 2]`` into ``"arrays[0][1][2]"``.
    """
    idx_str = ''.join('[{}]'.format(i) for i in index if i is not None)
    return 'arrays' + idx_str


def _block_check_depths_match(arrays, parent_index=[]):
    """
    Recursive function checking that the depths of nested lists in `arrays`
    all match. Mismatch raises a ValueError as described in the block
    docstring below.

    The entire index (rather than just the depth) needs to be calculated
    for each innermost list, in case an error needs to be raised, so that
    the index of the offending list can be printed as part of the error.

    Parameters
    ----------
    arrays : nested list of arrays
        The arrays to check
    parent_index : list of int
        The full index of `arrays` within the nested lists passed to
        `_block_check_depths_match` at the top of the recursion.

    Returns
    -------
    first_index : list of int
        The full index of an element from the bottom of the nesting in
        `arrays`. If any element at the bottom is an empty list, this will
        refer to it, and the last index along the empty axis will be None.
    max_arr_ndim : int
        The maximum of the ndims of the arrays nested in `arrays`.
    final_size: int
        The number of elements in the final array. This is used the motivate
        the choice of algorithm used using benchmarking wisdom.

    """
    if type(arrays) is tuple:
        # not strictly necessary, but saves us from:
        #  - more than one way to do things - no point treating tuples like
        #    lists
        #  - horribly confusing behaviour that results when tuples are
        #    treated like ndarray
        raise TypeError(
            '{} is a tuple. '
            'Only lists can be used to arrange blocks, and mt.block does '
            'not allow implicit conversion from tuple to ndarray.'.format(
                _block_format_index(parent_index)
            )
        )
    elif type(arrays) is list and len(arrays) > 0:
        idxs_ndims = (_block_check_depths_match(arr, parent_index + [i])
                      for i, arr in enumerate(arrays))

        first_index, max_arr_ndim, final_size = next(idxs_ndims)
        for index, ndim, size in idxs_ndims:
            final_size += size
            if ndim > max_arr_ndim:
                max_arr_ndim = ndim
            if len(index) != len(first_index):
                raise ValueError(
                    "List depths are mismatched. First element was at depth "
                    "{}, but there is an element at depth {} ({})".format(
                        len(first_index),
                        len(index),
                        _block_format_index(index)
                    )
                )
            # propagate our flag that indicates an empty list at the bottom
            if index[-1] is None:
                first_index = index

        return first_index, max_arr_ndim, final_size
    elif type(arrays) is list and len(arrays) == 0:
        # We've 'bottomed out' on an empty list
        return parent_index + [None], 0, 0
    else:
        # We've 'bottomed out' - arrays is either a scalar or an array
        size = _size(arrays)
        return parent_index, _ndim(arrays), size


def _atleast_nd(a, ndim):
    # Ensures `a` has at least `ndim` dimensions by prepending
    # ones to `a.shape` as necessary
    return array(a, ndmin=ndim, copy=False)


def _accumulate(values):
    return list(itertools.accumulate(values))


def _concatenate_shapes(shapes, axis):
    """Given array shapes, return the resulting shape and slices prefixes.
    These help in nested concatenation.

    Returns
    -------
    shape: tuple of int
        This tuple satisfies:
        ```
        shape, _ = _concatenate_shapes([arr.shape for shape in arrs], axis)
        shape == concatenate(arrs, axis).shape
        ```
    slice_prefixes: tuple of (slice(start, end), )
        For a list of arrays being concatenated, this returns the slice
        in the larger array at axis that needs to be sliced into.
        For example, the following holds:
        ```
        ret = concatenate([a, b, c], axis)
        _, (sl_a, sl_b, sl_c) = concatenate_slices([a, b, c], axis)
        ret[(slice(None),) * axis + sl_a] == a
        ret[(slice(None),) * axis + sl_b] == b
        ret[(slice(None),) * axis + sl_c] == c
        ```
        These are called slice prefixes since they are used in the recursive
        blocking algorithm to compute the left-most slices during the
        recursion. Therefore, they must be prepended to rest of the slice
        that was computed deeper in the recursion.
        These are returned as tuples to ensure that they can quickly be added
        to existing slice tuple without creating a new tuple every time.
    """
    # Cache a result that will be reused.
    shape_at_axis = [shape[axis] for shape in shapes]

    # Take a shape, any shape
    first_shape = shapes[0]
    first_shape_pre = first_shape[:axis]
    first_shape_post = first_shape[axis + 1:]

    if any(shape[:axis] != first_shape_pre or
           shape[axis + 1:] != first_shape_post for shape in shapes):
        raise ValueError(
                'Mismatched array shapes in block along axis {}.'.format(axis))

    shape = (first_shape_pre + (sum(shape_at_axis),) + first_shape[axis + 1:])

    offsets_at_axis = _accumulate(shape_at_axis)
    slice_prefixes = [(slice(start, end),)
                      for start, end in zip([0] + offsets_at_axis,
                                            offsets_at_axis)]
    return shape, slice_prefixes


def _block_info_recursion(arrays, max_depth, result_ndim, depth=0):
    """
    Returns the shape of the final array, along with a list
    of slices and a list of arrays that can be used for assignment inside the
    new array

    Parameters
    ----------
    arrays : nested list of arrays
        The arrays to check
    max_depth : list of int
        The number of nested lists
    result_ndim: int
        The number of dimensions in thefinal array.

    Returns
    -------
    shape : tuple of int
        The shape that the final array will take on.
    slices: list of tuple of slices
        The slices into the full array required for assignment. These are
        required to be prepended with ``(Ellipsis, )`` to obtain to correct
        final index.
    arrays: list of ndarray
        The data to assign to each slice of the full array

    """
    if depth < max_depth:
        shapes, slices, arrays = zip(
            *[_block_info_recursion(arr, max_depth, result_ndim, depth+1)
              for arr in arrays])

        axis = result_ndim - max_depth + depth
        shape, slice_prefixes = _concatenate_shapes(shapes, axis)

        # Prepend the slice prefix and flatten the slices
        slices = [slice_prefix + the_slice
                  for slice_prefix, inner_slices in zip(slice_prefixes, slices)
                  for the_slice in inner_slices]

        # Flatten the array list
        arrays = functools.reduce(operator.add, arrays)

        return shape, slices, arrays
    else:
        # We've 'bottomed out' - arrays is either a scalar or an array
        # type(arrays) is not list
        # Return the slice and the array inside a list to be consistent with
        # the recursive case.
        arr = _atleast_nd(arrays, result_ndim)
        return arr.shape, [()], [arr]


def _block(arrays, max_depth, result_ndim, depth=0):
    """
    Internal implementation of block based on repeated concatenation.
    `arrays` is the argument passed to
    block. `max_depth` is the depth of nested lists within `arrays` and
    `result_ndim` is the greatest of the dimensions of the arrays in
    `arrays` and the depth of the lists in `arrays` (see block docstring
    for details).
    """
    from ..merge.concatenate import concatenate

    if depth < max_depth:
        arrs = [_block(arr, max_depth, result_ndim, depth+1)
                for arr in arrays]
        return concatenate(arrs, axis=-(max_depth-depth))
    else:
        # We've 'bottomed out' - arrays is either a scalar or an array
        # type(arrays) is not list
        return _atleast_nd(arrays, result_ndim)


def block(arrays):
    """
    Assemble an nd-array from nested lists of blocks.

    Blocks in the innermost lists are concatenated (see `concatenate`) along
    the last dimension (-1), then these are concatenated along the
    second-last dimension (-2), and so on until the outermost list is reached.

    Blocks can be of any dimension, but will not be broadcasted using the normal
    rules. Instead, leading axes of size 1 are inserted, to make ``block.ndim``
    the same for all blocks. This is primarily useful for working with scalars,
    and means that code like ``mt.block([v, 1])`` is valid, where
    ``v.ndim == 1``.

    When the nested list is two levels deep, this allows block matrices to be
    constructed from their components.

    .. versionadded:: 1.13.0

    Parameters
    ----------
    arrays : nested list of array_like or scalars (but not tuples)
        If passed a single ndarray or scalar (a nested list of depth 0), this
        is returned unmodified (and not copied).

        Elements shapes must match along the appropriate axes (without
        broadcasting), but leading 1s will be prepended to the shape as
        necessary to make the dimensions match.

    Returns
    -------
    block_array : Tensor
        The array assembled from the given blocks.

        The dimensionality of the output is equal to the greatest of:
        * the dimensionality of all the inputs
        * the depth to which the input list is nested

    Raises
    ------
    ValueError
        * If list depths are mismatched - for instance, ``[[a, b], c]`` is
          illegal, and should be spelt ``[[a, b], [c]]``
        * If lists are empty - for instance, ``[[a, b], []]``

    See Also
    --------
    concatenate : Join a sequence of arrays along an existing axis.
    stack : Join a sequence of arrays along a new axis.
    vstack : Stack arrays in sequence vertically (row wise).
    hstack : Stack arrays in sequence horizontally (column wise).
    dstack : Stack arrays in sequence depth wise (along third axis).
    column_stack : Stack 1-D arrays as columns into a 2-D array.
    vsplit : Split an array into multiple sub-arrays vertically (row-wise).

    Notes
    -----

    When called with only scalars, ``mt.block`` is equivalent to an ndarray
    call. So ``mt.block([[1, 2], [3, 4]])`` is equivalent to
    ``mt.array([[1, 2], [3, 4]])``.

    This function does not enforce that the blocks lie on a fixed grid.
    ``mt.block([[a, b], [c, d]])`` is not restricted to arrays of the form::

        AAAbb
        AAAbb
        cccDD

    But is also allowed to produce, for some ``a, b, c, d``::

        AAAbb
        AAAbb
        cDDDD

    Since concatenation happens along the last axis first, `block` is _not_
    capable of producing the following directly::

        AAAbb
        cccbb
        cccDD

    Matlab's "square bracket stacking", ``[A, B, ...; p, q, ...]``, is
    equivalent to ``mt.block([[A, B, ...], [p, q, ...]])``.

    Examples
    --------
    The most common use of this function is to build a block matrix

    >>> import mars.tensor as mt
    >>> A = mt.eye(2) * 2
    >>> B = mt.eye(3) * 3
    >>> mt.block([
    ...     [A,               mt.zeros((2, 3))],
    ...     [mt.ones((3, 2)), B               ]
    ... ]).execute()
    array([[2., 0., 0., 0., 0.],
           [0., 2., 0., 0., 0.],
           [1., 1., 3., 0., 0.],
           [1., 1., 0., 3., 0.],
           [1., 1., 0., 0., 3.]])

    With a list of depth 1, `block` can be used as `hstack`

    >>> mt.block([1, 2, 3]).execute()    # hstack([1, 2, 3])
    array([1, 2, 3])

    >>> a = mt.array([1, 2, 3])
    >>> b = mt.array([2, 3, 4])
    >>> mt.block([a, b, 10]).execute()   # hstack([a, b, 10])
    array([ 1,  2,  3,  2,  3,  4, 10])

    >>> A = mt.ones((2, 2), int)
    >>> B = 2 * A
    >>> mt.block([A, B]).execute()       # hstack([A, B])
    array([[1, 1, 2, 2],
           [1, 1, 2, 2]])

    With a list of depth 2, `block` can be used in place of `vstack`:

    >>> a = mt.array([1, 2, 3])
    >>> b = mt.array([2, 3, 4])
    >>> mt.block([[a], [b]]).execute()   # vstack([a, b])
    array([[1, 2, 3],
           [2, 3, 4]])

    >>> A = mt.ones((2, 2), int)
    >>> B = 2 * A
    >>> mt.block([[A], [B]]).execute()   # vstack([A, B])
    array([[1, 1],
           [1, 1],
           [2, 2],
           [2, 2]])

    It can also be used in places of `atleast_1d` and `atleast_2d`

    >>> a = mt.array(0)
    >>> b = mt.array([1])
    >>> mt.block([a]).execute()          # atleast_1d(a)
    array([0])
    >>> mt.block([b]).execute()          # atleast_1d(b)
    array([1])

    >>> mt.block([[a]]).execute()        # atleast_2d(a)
    array([[0]])
    >>> mt.block([[b]]).execute()        # atleast_2d(b)
    array([[1]])


    """
    arrays, list_ndim, result_ndim, final_size = _block_setup(arrays)

    # It was found through benchmarking that making an array of final size
    # around 256x256 was faster by straight concatenation on a
    # i7-7700HQ processor and dual channel ram 2400MHz.
    # It didn't seem to matter heavily on the dtype used.
    #
    # A 2D array using repeated concatenation requires 2 copies of the array.
    #
    # The fastest algorithm will depend on the ratio of CPU power to memory
    # speed.
    # One can monitor the results of the benchmark
    # https://pv.github.io/numpy-bench/#bench_shape_base.Block2D.time_block2d
    # to tune this parameter until a C version of the `_block_info_recursion`
    # algorithm is implemented which would likely be faster than the python
    # version.
    if list_ndim * final_size > (2 * 512 * 512):
        return _block_slicing(arrays, list_ndim, result_ndim)
    else:
        return _block_concatenate(arrays, list_ndim, result_ndim)


# These helper functions are mostly used for testing.
# They allow us to write tests that directly call `_block_slicing`
# or `_block_concatenate` without blocking large arrays to force the wisdom
# to trigger the desired path.
def _block_setup(arrays):
    """
    Returns
    (`arrays`, list_ndim, result_ndim, final_size)
    """
    bottom_index, arr_ndim, final_size = _block_check_depths_match(arrays)
    list_ndim = len(bottom_index)
    if bottom_index and bottom_index[-1] is None:
        raise ValueError(
            'List at {} cannot be empty'.format(
                _block_format_index(bottom_index)
            )
        )
    result_ndim = max(arr_ndim, list_ndim)
    return arrays, list_ndim, result_ndim, final_size


def _block_slicing(arrays, list_ndim, result_ndim):
    shape, slices, arrays = _block_info_recursion(
        arrays, list_ndim, result_ndim)
    dtype = np.result_type(*[arr.dtype for arr in arrays])

    # Test preferring F only in the case that all input arrays are F
    F_order = all(arr.flags['F_CONTIGUOUS'] for arr in arrays)
    C_order = all(arr.flags['C_CONTIGUOUS'] for arr in arrays)
    order = 'F' if F_order and not C_order else 'C'
    result = empty(shape=shape, dtype=dtype, order=order)
    # Note: In a c implementation, the function
    # PyArray_CreateMultiSortedStridePerm could be used for more advanced
    # guessing of the desired order.

    for the_slice, arr in zip(slices, arrays):
        result[(Ellipsis,) + the_slice] = arr
    return result


def _block_concatenate(arrays, list_ndim, result_ndim):
    result = _block(arrays, list_ndim, result_ndim)
    if list_ndim == 0:
        # Catch an edge case where _block returns a view because
        # `arrays` is a single mars array and not a list of mars arrays.
        # This might copy scalars or lists twice, but this isn't a likely
        # usecase for those interested in performance
        result = result.copy()
    return result
