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

import os
import numpy as np
import pandas as pd
import pytest

import mars.dataframe as md
from mars.deploy.oscar.dataset import RayMLDataset
from mars.deploy.oscar.ray import new_cluster, _load_config
from mars.deploy.oscar.session import new_session
from mars.tests.conftest import *  # noqa
from mars.tests.core import require_ray
from mars.utils import lazy_import
from ray.util.data import MLDataset


ray = lazy_import('ray')
xgboost_ray = lazy_import('ray')


def require_xgboost_ray(func):
    if pytest:
        func = pytest.mark.xgboost_ray(func)
    func = pytest.mark.skipif(xgboost_ray is None, reason='xgboost_ray not installed')(func)
    return func


@pytest.fixture
async def create_cluster(request):
    param = getattr(request, "param", {})
    ray_config = _load_config()
    ray_config.update(param.get('config', {}))
    client = await new_cluster('test_cluster',
                               worker_num=4,
                               worker_cpu=2,
                               worker_mem=1 * 1024 ** 3,
                               config=ray_config)
    async with client:
        yield client


@require_ray
@pytest.mark.asyncio
async def test_convert_to_mldataset(ray_large_cluster, create_cluster):
    assert create_cluster.session
    session = new_session(address=create_cluster.address, backend='oscar', default=True)
    with session:
        value = np.random.rand(10, 10)
        df: md.DataFrame = md.DataFrame(value, chunk_size=5)
        df.execute()

        ds = RayMLDataset.from_mars(df, num_shards=4)
        assert isinstance(ds, MLDataset)


@require_ray
@pytest.mark.asyncio
async def test_mars_with_xgboost(ray_large_cluster, create_cluster):
    from sklearn.datasets import load_breast_cancer
    from xgboost_ray import RayDMatrix, RayParams, train

    assert create_cluster.session
    session = new_session(address=create_cluster.address, backend='oscar', default=True)
    with session:
        train_x, train_y = load_breast_cancer(return_X_y=True, as_frame=True)
        pd_df = pd.concat([train_x, train_y], axis=1)
        df: md.DataFrame = md.DataFrame(pd_df)

        num_shards = 4
        ds = RayMLDataset.from_mars(df, num_shards=num_shards)
        assert isinstance(ds, MLDataset)

        # train
        train_set = RayDMatrix(ds, "target")
        evals_result = {}
        bst = train(
            {
                "objective": "binary:logistic",
                "eval_metric": ["logloss", "error"],
            },
            train_set,
            evals_result=evals_result,
            evals=[(train_set, "train")],
            verbose_eval=False,
            ray_params=RayParams(
                num_actors=num_shards,  # Number of remote actors
                cpus_per_actor=1)
            )
        bst.save_model("model.xgb")
        assert os.path.exists("model.xgb")
        os.remove("model.xgb")
        print("Final training error: {:.4f}".format(
            evals_result["train"]["error"][-1]))
