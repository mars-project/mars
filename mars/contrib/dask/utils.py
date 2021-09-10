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

from typing import List

from dask import is_dask_collection
from dask.array.core import _concatenate2 as array_concat
from dask.dataframe import concat as df_concat
from dask.utils import is_arraylike, is_dataframe_like, is_series_like, is_index_like

from ...remote import spawn


def concat(objs: List):
    """
    Concat the results of partitioned dask task executions. This function guess the
        types of resulting list, then calls the corresponding native dask concat functions.

    Parameters
    ----------
    objs: List
        List of the partitioned dask task execution results, which will be concat.

    Returns
    -------
    obj:
        The concat result

    """
    if is_arraylike(objs[0]):
        res = array_concat(objs, axes=[0])  # TODO: Add concat with args support
    elif any((is_dataframe_like(objs[0]), is_series_like(objs[0]), is_index_like(objs[0]))):
        res = df_concat(objs)
    else:
        res = objs
    return res.compute() if is_dask_collection(res) else res


def reduce(objs: List[List]):
    """
    Spawn a concat task for 2d-list objects

    Parameters
    ----------
    objs: List
        2d-list of the partitioned dask task execution results, which will be concat.

    Returns
    -------
    obj:
        The spawning concat task.
    """
    return spawn(
        concat,
        args=([spawn(concat, args=(objs_d,)) for objs_d in objs],),
    )
