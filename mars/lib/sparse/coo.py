#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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
from .array import SparseNDArray
from .. import six


class COONDArray(SparseNDArray):
    __slots__ = ('indices', 'data', 'shape', 'dtype')
    """
    SparseNDArray in COO format.
    Stores indices in a 2-D array, indices, of size n * ndims, where n is the number of
    non-zero values and ndims is the number of dimensions of
    the SparseNDArray.
    For example,
    indices = [(1,3,4), (2,1,3), (3, 2, 2)],
    data = [11, 12, 13],
    shape = [4, 5, 6],
    together they represent a sparse array of size 4*5*6
    with data 11, 12, 13 at coordinate (1,3,4), (2,1,3) and (3, 2, 2).

    Attributes:
    -----------
    data: ndarray
        A 1-D array of size n, where n is the number of elements
    indices: ndarray
        A 2-D array of size n * ndims, where n is the number of non-zero values
            and ndims is the number of dimensions of the SparseNDArray.
    shape: ndarray
        A 1-D array of size ndims, where ndims is the number of dimensions
            of the sparse array in the dense form.
    """

    def __init__(self, indices=(), data=(), shape=(), dtype=np.int64):
        try:
            self.dtype = data.dtype
        except AttributeError:
            self.dtype = dtype
        self.indices = np.asarray(indices)
        self.data = np.asarray(data, dtype=self.dtype)
        self.shape = np.asarray(shape)

    # return raw components of a COONDArray instance
    @property
    def raw(self):
        return self.indices, self.data, self.shape

    @property
    def ndim(self):
        return self.indices.shape[1]

    @property
    def nnz(self):
        return self.indices.shape[0]

    # return a copy of a COONDArray instance
    def copy(self):
        return COONDArray(self.indices, self.data, self.shape)

    # Return a copy of nd.ndarray of self
    def toarray(self):
        # allocate dense_arr
        dense_arr = np.zeros(shape=self.shape, dtype=self.data.dtype, order='C')

        # Assign values to corresponding coordinates in dense array
        # Each column vector contains coordinates of one and only one dim.
        if len(self.data) == 0 and len(self.indices) == 0:
            return dense_arr
        dense_arr[tuple(self.indices.T)] = self.data
        # ( Written in plainer words:
        # dense_arr[
        #     <split n-dim coords into n column vectors>,
        #     number of dims, split along axis 1
        # ] = transform_into_column_vectors(self.values)
        # P.S. Each column vector contains coordinates of a dim. )
        return dense_arr

    # Per scipy sparse issparse / scipy matrix issparse
    def issparse(self):
        if isinstance(self, COONDArray):
            return True

    def mask_matches(self, other):
        mask = (self.indices[:, np.newaxis, :]
                == other.indices[np.newaxis, :, :]).all(axis=2)
        matches = np.where(mask == True)
        # Check if the entire row is false.
        # If so, then the coordinate at the corresponding position
        # in the self.indices does not have a matching coordinate in other.indices
        self_non_matches = np.where(~mask.any(axis=1))
        # Check if the entire column is false;.
        # If so, then the coordinate at corresponding position
        # in the other.indices does not have a matching coordinate in self.indices
        other_non_matches = np.where(~mask.any(axis=0))
        matches = np.asarray(matches)
        return matches, self_non_matches, other_non_matches

    def __add__(self, other, increment_all=False):
        """
        Perform elementwise addition of two sparse tensor.
        ind stands for index, inds stands for indices, dct stands for dictionary.

        :param other:
        :return: the sum of two SparseNDArray as an object.
        OUTPUT TYPE:
        Sparse + Sparse -> Sparse
        Sparse + Dense -> Dense
        Sparse + Constant -> Dense (following conventions of other libraries)
        """
        if isinstance(other, SparseNDArray):
            matches, self_non_matches, other_non_matches = self.mask_matches(other)

            # calculate sums of values at matching coordinates
            matching_vals = self.data[matches[0]] + other.data[matches[1]]
            # document matching coordinates
            matching_indices = self.indices[matches[0]]

            # append non-matching coordinates in self to existing indices
            new_indices = np.append(matching_indices,
                                    self.indices[self_non_matches], axis=0)
            # append non-matching coordinates in other to exisiting indices
            new_indices = np.append(new_indices,
                                    other.indices[other_non_matches], axis=0)
            # append values at non-matching coordinates in self to existing values
            new_vals = np.append(matching_vals,
                                 self.data[self_non_matches], axis=0)
            # append values at non-matching coordinates in other to existing values
            new_vals = np.append(new_vals,
                                 other.data[other_non_matches], axis=0)

            return COONDArray(new_indices, new_vals, self.shape)

        elif isinstance(other, np.ndarray):
            result = np.copy(other)
            for i in range(len(self.indices)):
                result[tuple(self.indices[i])] += self.data[i]
            return result

        elif isinstance(other, (int, float, np.int, np.float)):
            result = np.ones(shape=self.shape) * other
            for i in range(len(self.indices)):
                result[self.indices[i]] += self.data[i]
            return result

        else:
            return NotImplemented
            # raise TypeError(
            #     "The added is not (dense)np.ndarray, "
            #     "(sparse)COONDArray or (constant)int/float. ")

    def __sub__(self, other, increment_all=False):
        """
        Perform element-wise subtraction of two sparse tensor.
        （numpy function to use: setdiff1d, in1d, isin.）

        If having difficulty of understanding and/or doubts about any
        implementation decisions in this function, refer to the
        comment block in __add__.

        :param other: what to subtract
        :return: the sum of two SparseNDArray as an object.
                    Sparse - Sparse -> Sparse
                    Sparse - Dense -> Sparse
                    Sparse - Constant -> Dense

        Refer to additional comments on decision-makings of implementation
        at the corresponding location in __add__ function.
        """
        if isinstance(other, SparseNDArray):
            matches, self_non_matches, other_non_matches = self.mask_matches(other)

            # calculate sums of values at matching coordinates
            matching_vals = self.data[matches[0]] - other.data[matches[1]]
            # document matching coordinates
            matching_indices = self.indices[matches[0]]

            # append non-matching coordinates in self to existing indices
            new_indices = np.append(matching_indices,
                                    self.indices[self_non_matches], axis=0)
            # append non-matching coordinates in other to exisiting indices
            new_indices = np.append(new_indices,
                                    other.indices[other_non_matches], axis=0)
            # append values at non-matching coordinates in self to existing values
            new_vals = np.append(matching_vals,
                                 self.data[self_non_matches], axis=0)
            # append values at non-matching coordinates in other to existing values
            new_vals = np.append(new_vals,
                                 (-1) * other.data[other_non_matches], axis=0)

            return COONDArray(new_indices, new_vals, self.shape)

        elif isinstance(other, np.ndarray):
            result = np.copy(other)
            for i in range(len(self.indices)):
                result[tuple(self.indices[i])] -= self.data[i]
            return np.negative(result)

        elif isinstance(other, (int, float, np.int, np.float)):
            result = np.ones(shape=self.shape) * other
            for i in range(len(self.indices)):
                result[self.indices[i]] -= self.data[i]
            return np.negative(result)

        else:
            return NotImplemented
            # Equivalent to:
            # raise TypeError(
            #     "The subtracted is not (dense)np.ndarray, "
            #     "(sparse)COONDArray or (constant)int/float. ")

    def __mul__(self, other):
        """
        Element-wise multiplication between two matrices, self and other.
        Self(self) is always assumed to be sparse.
        Other(other) can be one of the three types: sparse COONDArray,
            dense np.ndarray or any other numerical values.

        :param other: the multiplied
        :return: element-wise product of self and other
                    (S: Sparse, D: Dense, C: Constant)
                    Sparse * Sparse -> Sparse
                    Sparse * Dense -> Sparse
                    Sparse * Constant -> Sparse

        More on my __mul__ implementation:
            When operating on operands in the COO format, one must pay attention to matches of coordinates.
            For different operators, logic on manipulating data from left and right side of operators may be
            subtly different, yet these differences are significant.
            For summations and subtractions, they are analogous to LOGIC OR, since we book-keep both sides on
            either '+' or '-' whether a coordinate
            has a match or not.
            For multiplications, it is analogous to LOGIC AND, since we book-keep only matching coordinates
            and their values; the rest of them are made irrelevant by being multiplied by zero.
            Division is a bit complicated. It shares some similarities with multiplications, but, when other
            contains 0, ZeroDivisionError will be raised and values of those elements will be np.inf.

        """
        # if isinstance(other, SparseNDArray) and self.shape == other.shape:
        if isinstance(other, SparseNDArray) and (self.shape == other.shape).all():
            mask = (self.indices[:, np.newaxis, :] == other.indices[np.newaxis, :, :]).all(axis=2)
            matches = np.where(mask == True)
            matches = np.asarray(matches)

            matching_vals = self.data[matches[0]] * other.data[matches[1]]
            matching_indices = self.indices[matches[0]]

            new_vals = matching_vals
            new_indices = matching_indices

            return COONDArray(new_indices, new_vals, self.shape)

        elif isinstance(other, np.ndarray):
            data = np.copy(self.data)
            for i in range(len(data)):
                data[i] *= other[tuple(self.indices[i])]
            return COONDArray(self.indices, data, self.shape)

        elif isinstance(other, (int, float, np.int, np.float)):
            if other == 0 or other == float(0):
                return COONDArray(indices=[], data=[], shape=self.shape)
            elif other == 1 or other == float(1):
                return COONDArray(self.indices, self.data, self.shape)
            else:
                data = list(map(lambda v: v * other, self.data))
                return COONDArray(self.indices, data, self.shape)

        else:
            return NotImplemented
            # Equivalent to:
            # raise TypeError("The multiplying is not (dense)np.ndarray, (sparse)COONDArray or (constant)int/float.")

    def __truediv__(self, other):
        """
        Element-wise division between two matrices, self and other.
        Self(self) is always assumed to be sparse.
        Other(other) can be one of these types: sparse COONDArray,
            dense np.ndarray or any other numerical values.

        :param other: the divisor
        :return: element-wise product of self and other
                    S: Sparse, D: Dense, C: Constant
                    Sparse / Sparse -> Sparse
                    Sparse / Dense -> Sparse
                    Sparse * Constant -> Sparse
        """

        if isinstance(other, COONDArray):
            if six.PY2:
                return np.true_divide(self.toarray(), other.toarray())
            else:
                return self.toarray() / other.toarray()

        elif isinstance(other, np.ndarray):
            if six.PY2:
                return np.true_divide(self.toarray(), other)
            else:
                return self.toarray() / other

        elif isinstance(other, (int, float, np.int, np.float)):
            if other == 0 or other == 0.0:
                return NotImplemented
                # Equivalent to:
                # raise ValueError("A divisor should not be zero")

            elif other == 1 or other == 1.0:
                return COONDArray(self.indices,
                                  np.asarray(self.data, dtype=type(other)), self.shape)

            else:
                if six.PY2:
                    data = np.true_divide(self.data, float(other))
                else:
                    data = np.asarray(self.data) / float(other)
                return COONDArray(self.indices, data, self.shape)

        else:
            return NotImplemented
            # Equivalent to:
            # raise TypeError("The divisor is not (dense)np.ndarray, (sparse)COONDArray or (constant)int/float. ")

    # Support function for python 2.7
    # IN THE FUTURE: __div__ returns the result of the function call to __floordiv__
    def __div__(self, other):
        return self.__truediv__(other)

    def transpose(self, axes=None):
        """
        transpose a COONDArray instance.
        (Exchange values of coordinates given axes (index of coordinate values))

        :param axes: axes whose coordinates are changed.
        :return: not returning anyting. Instead, states of the calling objecting
                    is updated. Change indices of the calling object.
        """
        # CASE: incorrect input type

        if axes is None:
            # indices = list(map(lambda c: tuple(c[-1::-1]), self.indices))
            return COONDArray(self.indices[:, -1::-1], self.data, self.shape[-1::-1])

        elif not isinstance(axes, tuple):
            # return NotImplemented
            # Equivalent to:
            raise TypeError("axes should be either None or a tuple of length 2. ")

        elif isinstance(axes, tuple) and len(axes) != 2:
            # return NotImplemented
            # Equivalent to:
            raise ValueError("If a tuple is given, its length should be 2. ")

        elif isinstance(axes, tuple) and len(axes) == 2:
            indices = np.copy(self.indices)
            shape = np.copy(self.shape)

            indices[:, [axes[0], axes[1]]] = indices[:, [axes[1], axes[0]]]
            shape[[axes[0], axes[1]]] = shape[[axes[1], axes[0]]]

            return COONDArray(indices, self.data, shape)

        else:
            # return NotImplemented
            raise TypeError("Some unknown type is given as input! ")


# Per SciPy csr_matrix issparse().
def is_coo(x):
    if isinstance(x, COONDArray):
        return True
