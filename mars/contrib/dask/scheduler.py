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

from dask.core import istask, ishashable

from typing import List, Tuple
from .utils import reduce
from ...remote import spawn


def mars_scheduler(dsk: dict, keys: List[List[str]]):
    """
    A Dask-Mars scheduler

    This scheduler is intended to be compatible with existing
    dask user interface, no callbacks are implemented.

    Parameters
    ----------
    dsk: Dict
        Dask graph, represented as a task DAG dictionary.
    keys: List[List[str]]
        2d-list of Dask graph keys whose values we wish to compute and return.

    Returns
    -------
    Object
        Computed values corresponding to the provided keys.
    """
    res = reduce(mars_dask_get(dsk, keys)).execute().fetch()
    if not isinstance(res, List):
        return [[res]]
    else:
        return res


def mars_dask_get(dsk: dict, keys: List[List]):
    """
    A Dask-Mars convert function. This function will send the dask graph layers
        to Mars Remote API, generating mars objects correspond to the provided keys.

    Parameters
    ----------
    dsk: Dict
        Dask graph, represented as a task DAG dictionary.
    keys: List[List[str]]
        2d-list of Dask graph keys whose values we wish to compute and return.

    Returns
    -------
    Object
        Spawned mars objects corresponding to the provided keys.
    """

    def _get_arg(a):
        # if arg contains layer index or callable objs, handle it
        if ishashable(a) and a in dsk.keys():
            while ishashable(a) and a in dsk.keys():
                a = dsk[a]
            return _execute_task(a)
        elif not isinstance(a, str) and hasattr(a, "__getitem__"):
            if istask(a):  # TODO:Handle `SubgraphCallable`, which may contains dsk in it
                return spawn(a[0], args=tuple(_get_arg(i) for i in a[1:]))
            elif isinstance(a, dict):
                return {k: _get_arg(v) for k, v in a.items()}
            elif isinstance(a, List) or isinstance(a, Tuple):
                return type(a)(_get_arg(i) for i in a)
        return a

    def _execute_task(task: tuple):
        if not istask(task):
            return _get_arg(task)
        return spawn(task[0], args=tuple(_get_arg(a) for a in task[1:]))

    return [[_execute_task(dsk[k]) for k in keys_d] for keys_d in keys]
