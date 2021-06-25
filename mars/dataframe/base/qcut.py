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

import numpy as np
import pandas as pd
from pandas.api.types import is_integer

from ...core import ENTITY_TYPE
from ...tensor import tensor as astensor
from ...tensor.statistics.percentile import percentile
from ..core import DATAFRAME_TYPE, SERIES_TYPE
from ..initializer import DataFrame, Series
from .cut import cut


def qcut(x, q, labels=None, retbins=False, precision=3, duplicate='raise'):
    """
    Quantile-based discretization function.

    Discretize variable into equal-sized buckets based on rank or based
    on sample quantiles. For example 1000 values for 10 quantiles would
    produce a Categorical object indicating quantile membership for each data point.

    Parameters
    ----------
    x : 1d tensor or Series
    q : int or list-like of float
        Number of quantiles. 10 for deciles, 4 for quartiles, etc. Alternately
        array of quantiles, e.g. [0, .25, .5, .75, 1.] for quartiles.
    labels : array or False, default None
        Used as labels for the resulting bins. Must be of the same length as
        the resulting bins. If False, return only integer indicators of the
        bins. If True, raises an error.
    retbins : bool, optional
        Whether to return the (bins, labels) or not. Can be useful if bins
        is given as a scalar.
    precision : int, optional
        The precision at which to store and display the bins labels.
    duplicates : {default 'raise', 'drop'}, optional
        If bin edges are not unique, raise ValueError or drop non-uniques.

    Returns
    -------
    out : Categorical or Series or tensor of integers if labels is False
        The return type (Categorical or Series) depends on the input: a Series
        of type category if input is a Series else Categorical. Bins are
        represented as categories when categorical data is returned.
    bins : tensor of floats
        Returned only if `retbins` is True.

    Notes
    -----
    Out of bounds values will be NA in the resulting Categorical object

    Examples
    --------
    >>> import mars.dataframe as md
    >>> md.qcut(range(5), 4).execute()
    ... # doctest: +ELLIPSIS
    [(-0.001, 1.0], (-0.001, 1.0], (1.0, 2.0], (2.0, 3.0], (3.0, 4.0]]
    Categories (4, interval[float64]): [(-0.001, 1.0] < (1.0, 2.0] ...

    >>> md.qcut(range(5), 3, labels=["good", "medium", "bad"]).execute()
    ... # doctest: +SKIP
    [good, good, medium, bad, bad]
    Categories (3, object): [good < medium < bad]

    >>> md.qcut(range(5), 4, labels=False).execute()
    array([0, 0, 1, 2, 3])
    """
    if is_integer(q):
        q = np.linspace(0, 1, q + 1)

    if isinstance(x, (DATAFRAME_TYPE, SERIES_TYPE, pd.DataFrame, pd.Series)):
        x = DataFrame(x) if x.ndim == 2 else Series(x)
        bins = x.quantile(q)
    else:
        x = astensor(x)
        if isinstance(q, ENTITY_TYPE):
            q = q * 100
        else:
            q = [iq * 100 for iq in q]
        bins = percentile(x, q)

    return cut(x, bins, labels=labels, retbins=retbins,
               precision=precision, include_lowest=True,
               duplicates=duplicate)
