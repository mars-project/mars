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

import math
import numbers
import warnings
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

try:
    from sklearn import get_config as sklearn_get_config
except ImportError:  # pragma: no cover
    sklearn_get_config = None

from ... import options
from ...core import enter_mode
from ...typing import TileableType
from ...dataframe import DataFrame, Series
from ...dataframe.core import DATAFRAME_TYPE, SERIES_TYPE
from ...tensor import tensor as astensor
from ...utils import parse_readable_size


def convert_to_tensor_or_dataframe(item):
    if isinstance(item, (DATAFRAME_TYPE, pd.DataFrame)):
        item = DataFrame(item)
    elif isinstance(item, (SERIES_TYPE, pd.Series)):
        item = Series(item)
    else:
        item = astensor(item)
    return item


def concat_chunks(chunks):
    tileable = chunks[0].op.create_tileable_from_chunks(chunks)
    return tileable.op.concat_tileable_chunks(tileable).chunks[0]


def copy_learned_attributes(from_estimator: BaseEstimator, to_estimator: BaseEstimator):
    attrs = {k: v for k, v in vars(from_estimator).items() if k.endswith("_")}
    for k, v in attrs.items():
        setattr(to_estimator, k, v)


def is_scalar_nan(x):
    """Tests if x is NaN.

    This function is meant to overcome the issue that np.isnan does not allow
    non-numerical types as input, and that np.nan is not float('nan').

    Parameters
    ----------
    x : any type

    Returns
    -------
    boolean

    Examples
    --------
    >>> is_scalar_nan(np.nan)
    True
    >>> is_scalar_nan(float("nan"))
    True
    >>> is_scalar_nan(None)
    False
    >>> is_scalar_nan("")
    False
    >>> is_scalar_nan([np.nan])
    False
    """
    return isinstance(x, numbers.Real) and math.isnan(x)


def get_chunk_n_rows(row_bytes, max_n_rows=None, working_memory=None):
    """Calculates how many rows can be processed within working_memory

    Parameters
    ----------
    row_bytes : int
        The expected number of bytes of memory that will be consumed
        during the processing of each row.
    max_n_rows : int, optional
        The maximum return value.
    working_memory : int or float, optional
        The number of rows to fit inside this number of MiB will be returned.
        When None (default), the value of
        ``sklearn.get_config()['working_memory']`` is used.

    Returns
    -------
    int or the value of n_samples

    Warns
    -----
    Issues a UserWarning if ``row_bytes`` exceeds ``working_memory`` MiB.
    """

    if working_memory is None:  # pragma: no cover
        working_memory = options.learn.working_memory
        if working_memory is None and sklearn_get_config is not None:
            working_memory = sklearn_get_config()["working_memory"]
        elif working_memory is None:
            working_memory = 1024

    if isinstance(working_memory, int):
        working_memory *= 2**20
    else:
        working_memory = parse_readable_size(working_memory)[0]

    chunk_n_rows = int(working_memory // row_bytes)
    if max_n_rows is not None:
        chunk_n_rows = min(chunk_n_rows, max_n_rows)
    if chunk_n_rows < 1:  # pragma: no cover
        warnings.warn(
            "Could not adhere to working_memory config. "
            "Currently %.0fMiB, %.0fMiB required."
            % (working_memory, np.ceil(row_bytes * 2**-20))
        )
        chunk_n_rows = 1
    return chunk_n_rows


@enter_mode(build=True)
def sort_by(
    tensors: List[TileableType], by: TileableType, ascending: bool = True
) -> List[TileableType]:
    # sort tensors by another tensor
    i_to_tensors = {i: t for i, t in enumerate(tensors)}
    if by not in tensors:
        by_name = len(i_to_tensors)
        i_to_tensors[by_name] = by
    else:
        by_name = tensors.index(by)
    df = DataFrame(i_to_tensors)
    sorted_df = df.sort_values(by_name, ascending=ascending)
    return [sorted_df[i].to_tensor() for i in range(len(tensors))]
