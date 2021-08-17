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

from dask import is_dask_collection, optimize
from dask.bag import Bag

from .scheduler import mars_dask_get
from .utils import reduce
from ...remote import spawn


def convert_dask_collection(dc):
    """
    Convert dask collection object into mars.core.Object via remote API

    Parameters
    ----------
    dc: dask collection
        Dask collection object to be converted.

    Returns
    -------
    Object
        Mars Object.
    """
    if not is_dask_collection(dc):
        raise TypeError(f"'{type(dc).__name__}' object is not a valid dask collection")

    dc.__dask_graph__().validate()
    dsk = optimize(dc)[0].__dask_graph__()

    first_key = next(iter(dsk.keys()))
    if isinstance(first_key, str):
        key = [first_key]
    elif isinstance(first_key, tuple):
        key = sorted([i for i in dsk.keys() if i[0] == first_key[0]], key=lambda x: x[1])
    else:
        raise ValueError(
            f"Dask collection object seems be broken, with unexpected key type:'{type(first_key).__name__}'")
    res = reduce(mars_dask_get(dsk, [key]))
    if isinstance(dc, Bag):
        return spawn(lambda x: list(x[0][0]), args=(res,))
    else:
        return res
