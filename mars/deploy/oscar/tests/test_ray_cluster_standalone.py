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

import mars
from .... import tensor as mt
from .... import dataframe as md
from ....tests.core import require_ray
from ....utils import lazy_import
from ..ray import (
    new_cluster_in_ray,
    new_ray_session,
)

ray = lazy_import("ray")


@require_ray
def test_new_cluster_in_ray(stop_ray):
    cluster = new_cluster_in_ray(worker_num=2)
    mt.random.RandomState(0).rand(100, 5).sum().execute()
    cluster.session.execute(mt.random.RandomState(0).rand(100, 5).sum())
    mars.execute(mt.random.RandomState(0).rand(100, 5).sum())
    session = new_ray_session(address=cluster.address, session_id="abcd", default=True)
    session.execute(mt.random.RandomState(0).rand(100, 5).sum())
    mars.execute(mt.random.RandomState(0).rand(100, 5).sum())
    cluster.stop()


@require_ray
def test_new_ray_session(stop_ray):
    new_ray_session_test()


def new_ray_session_test():
    session = new_ray_session(session_id="abc", worker_num=2)
    mt.random.RandomState(0).rand(100, 5).sum().execute()
    session.execute(mt.random.RandomState(0).rand(100, 5).sum())
    mars.execute(mt.random.RandomState(0).rand(100, 5).sum())
    session = new_ray_session(session_id="abcd", worker_num=2, default=True)
    session.execute(mt.random.RandomState(0).rand(100, 5).sum())
    mars.execute(mt.random.RandomState(0).rand(100, 5).sum())
    df = md.DataFrame(mt.random.rand(100, 4), columns=list("abcd"))
    # Convert mars dataframe to ray dataset
    ds = md.to_ray_dataset(df)
    print(ds.schema(), ds.count())
    ds.filter(lambda row: row["a"] > 0.5).show(5)
    # Convert ray dataset to mars dataframe
    df2 = md.read_ray_dataset(ds)
    print(df2.head(5).execute())
    # Test ray cluster exists after session got gc.
    del session
    import gc

    gc.collect()
    mars.execute(mt.random.RandomState(0).rand(100, 5).sum())
