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

from .core import issparse


class SparseNDArray(object):
    __slots__ = '__weakref__',
    __array_priority__ = 21

    def __new__(cls, *args, **kwargs):

        if len(args) == 1 and issparse(args[0]) and args[0].ndim == 2:
            from .matrix import SparseMatrix

            return object.__new__(SparseMatrix)

        else:
            from .coo import COONDArray
            return object.__new__(COONDArray)

    @property
    def raw(self):
        raise NotImplementedError
