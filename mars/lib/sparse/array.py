#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from .core import issparse, get_array_module, get_sparse_module


class SparseNDArray(object):
    __slots__ = '__weakref__',
    __array_priority__ = 21

    def __new__(cls, *args, **kwargs):

        shape = kwargs.get('shape', None)
        if shape is not None and len(shape) == 1:
            from .vector import SparseVector

            return object.__new__(SparseVector)
        if len(args) == 1 and issparse(args[0]) and args[0].ndim == 2:
            from .matrix import SparseMatrix

            return object.__new__(SparseMatrix)

        else:
            from .coo import COONDArray
            return object.__new__(COONDArray)

    @property
    def raw(self):
        raise NotImplementedError


def call_sparse_binary_scalar(method, left, right, **kwargs):
    if isinstance(left, SparseNDArray):
        spmatrix = left.spmatrix
        xp = get_array_module(spmatrix)
        new_data = getattr(xp, method)(spmatrix.data, right, **kwargs)
        shape = left.shape
    else:
        spmatrix = right.spmatrix
        xp = get_array_module(spmatrix)
        new_data = getattr(xp, method)(left, spmatrix.data, **kwargs)
        shape = right.shape
    new_spmatrix = get_sparse_module(spmatrix).csr_matrix(
        (new_data, spmatrix.indices, spmatrix.indptr), spmatrix.shape)
    return SparseNDArray(new_spmatrix, shape=shape)


def call_sparse_unary(method, matrix, *args, **kwargs):
    spmatrix = matrix.spmatrix
    xp = get_array_module(spmatrix)
    new_data = getattr(xp, method)(spmatrix.data, *args, **kwargs)
    new_spmatrix = get_sparse_module(spmatrix).csr_matrix(
        (new_data, spmatrix.indices, spmatrix.indptr), spmatrix.shape)
    return SparseNDArray(new_spmatrix, shape=matrix.shape)
