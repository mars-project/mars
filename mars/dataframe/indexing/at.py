# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

from .loc import DataFrameLoc


class DataFrameAt:
    def __init__(self, obj):
        self._obj = obj
        self._loc = DataFrameLoc(self._obj)

    def __getitem__(self, indexes):
        if not isinstance(indexes, tuple):
            indexes = (indexes,)

        for index in indexes:
            if not np.isscalar(index):
                raise ValueError('Invalid call for scalar access (getting)!')

        return self._loc[indexes]


def at(a):
    """
    Access a single value for a row/column label pair.

    Similar to ``loc``, in that both provide label-based lookups. Use
    ``at`` if you only need to get or set a single value in a DataFrame
    or Series.

    Raises
    ------
    KeyError
        If 'label' does not exist in DataFrame.

    See Also
    --------
    DataFrame.iat : Access a single value for a row/column pair by integer
        position.
    DataFrame.loc : Access a group of rows and columns by label(s).
    Series.at : Access a single value using a label.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> df = md.DataFrame([[0, 2, 3], [0, 4, 1], [10, 20, 30]],
    ...                   index=[4, 5, 6], columns=['A', 'B', 'C'])
    >>> df.execute()
        A   B   C
    4   0   2   3
    5   0   4   1
    6  10  20  30

    Get value at specified row/column pair

    >>> df.at[4, 'B'].execute()
    2

    # Set value at specified row/column pair
    #
    # >>> df.at[4, 'B'] = 10
    # >>> df.at[4, 'B']
    # 10

    Get value within a Series

    >>> df.loc[5].at['B'].execute()
    4
    """
    return DataFrameAt(a)
