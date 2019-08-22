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

from .initializer import DataFrame, Series
# do imports to register operands
from .datasource.from_tensor import from_tensor
from .datasource.from_records import from_records
from .utils import concat_tileable_chunks, get_fetch_op_cls, get_fuse_op_cls
from .fetch import DataFrameFetch, DataFrameFetchShuffle

from . import arithmetic
from . import indexing
from . import merge
from . import reduction
del reduction, arithmetic, indexing, merge

del DataFrameFetch, DataFrameFetchShuffle
