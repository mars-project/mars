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

from ...utils import lazy_import
from .datasource import register_data_source_handler
from .datastore import register_data_store_handler
from .random import register_random_handler
from .base import register_basic_handler
from .arithmetic import register_arithmetic_handler
from .indexing import register_indexing_handler
from .reduction import register_reduction_handler
from .merge import register_merge_handler
from .fetch import register_fetch_handler
from .fft import register_fft_handler
from .linalg import register_linalg_handler
from .reshape import register_reshape_handler


ne = lazy_import('numexpr', globals=globals(), rename='ne')
cp = lazy_import('cupy', globals=globals(), rename='cp')

NUMEXPR_INSTALLED = ne is not None
CP_INSTALLED = cp is not None


def register_tensor_execution_handler():
    if NUMEXPR_INSTALLED:
        from .ne import register_numexpr_handler
        register_numexpr_handler()
    if CP_INSTALLED:
        from .cp import register_cp_handler
        register_cp_handler()
    register_data_source_handler()
    register_data_store_handler()
    register_random_handler()
    register_basic_handler()
    register_arithmetic_handler()
    register_indexing_handler()
    register_reduction_handler()
    register_merge_handler()
    register_fetch_handler()
    register_fft_handler()
    register_linalg_handler()
    register_reshape_handler()
