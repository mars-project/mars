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

from .initializer import DataFrame, Series, Index
# do imports to register operands
from .base.cut import cut
from .base.eval import mars_eval as eval  # pylint: disable=redefined-builtin
from .base.melt import melt
from .base.qcut import qcut
from .base.to_numeric import to_numeric
from .datasource.from_tensor import dataframe_from_tensor, series_from_tensor
from .datasource.from_index import series_from_index
from .datasource.from_records import from_records
from .datasource.read_csv import read_csv
from .datasource.read_sql import read_sql, read_sql_table, read_sql_query
from .datasource.read_parquet import read_parquet
from .datasource.date_range import date_range
from .fetch import DataFrameFetch, DataFrameFetchShuffle
from .merge import concat, merge
from .missing.checkna import isna, isnull, notna, notnull
from .reduction import CustomReduction, unique
from .tseries.to_datetime import to_datetime

from . import arithmetic
from . import base
from . import indexing
from . import merge as merge_
from . import missing
from . import reduction
from . import statistics
from . import sort
from . import groupby
from . import ufunc
from . import datastore
from . import window
from . import plotting

del reduction, statistics, arithmetic, indexing, merge_, base, groupby, \
    missing, ufunc, datastore, sort, window, plotting
del DataFrameFetch, DataFrameFetchShuffle

# noinspection PyUnresolvedReferences
from ..core import ExecutableTuple
from .arrays import ArrowStringDtype, ArrowStringArray, ArrowListDtype, ArrowListArray

# noinspection PyUnresolvedReferences
from pandas import Timedelta, Timestamp, offsets, NaT, Interval
try:
    from pandas import NA, NamedAgg
except ImportError:  # pragma: no cover
    pass
