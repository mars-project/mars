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

from ..... import dataframe as md
from .....deploy.oscar.ray import new_cluster
from .....deploy.oscar.session import new_session
from .....tests.core import require_ray
from .....utils import lazy_import
from ....contrib import raydataset as mdd


ray = lazy_import("ray")
# Ray Datasets is available in early preview at ray.data with Ray 1.6+
# (and ray.experimental.data in Ray 1.5)
ray_dataset = lazy_import("ray.data")
try:
    import xgboost_ray
except ImportError:  # pragma: no cover
    xgboost_ray = None
try:
    import sklearn
except ImportError:  # pragma: no cover
    sklearn = None


@pytest.fixture
async def create_cluster(request):
    client = await new_cluster(
        "test_cluster",
        supervisor_mem=1 * 1024**3,
        worker_num=4,
        worker_cpu=2,
        worker_mem=1 * 1024**3,
    )
    async with client:
        yield client


@require_ray
@pytest.mark.asyncio
@pytest.mark.parametrize("test_option", [[3, 3], [3, 2], [None, None]])
async def test_convert_to_ray_dataset(ray_large_cluster, create_cluster, test_option):
    assert create_cluster.session
    session = new_session(address=create_cluster.address, backend="oscar", default=True)
    with session:
        value = np.random.rand(10, 10)
        chunk_size, num_shards = test_option
        df: md.DataFrame = md.DataFrame(value, chunk_size=chunk_size)
        df.execute()

        ds = mdd.to_ray_dataset(df, num_shards=num_shards)
        assert isinstance(ds, ray_dataset.Dataset)


@require_ray
@pytest.mark.asyncio
@pytest.mark.skipif(xgboost_ray is None, reason="xgboost_ray not installed")
async def test_mars_with_xgboost(ray_large_cluster, create_cluster):
    from xgboost_ray import RayDMatrix, RayParams, train
    from sklearn.datasets import load_breast_cancer

    assert create_cluster.session
    session = new_session(address=create_cluster.address, backend="oscar", default=True)
    with session:
        train_x, train_y = load_breast_cancer(return_X_y=True, as_frame=True)
        pd_df = pd.concat([train_x, train_y], axis=1)
        df: md.DataFrame = md.DataFrame(pd_df)
        df.execute()

        num_shards = 4
        ds = md.to_ray_dataset(df, num_shards=num_shards)
        assert isinstance(ds, ray_dataset.Dataset)

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
                num_actors=num_shards, cpus_per_actor=1  # Number of remote actors
            ),
        )
        bst.save_model("model.xgb")
        assert os.path.exists("model.xgb")
        os.remove("model.xgb")
        print("Final training error: {:.4f}".format(evals_result["train"]["error"][-1]))


@require_ray
@pytest.mark.parametrize(
    "ray_large_cluster", [{"num_nodes": 3, "num_cpus": 16}], indirect=True
)
@pytest.mark.asyncio
@pytest.mark.skipif(sklearn is None, reason="sklearn not installed")
@pytest.mark.skipif(xgboost_ray is None, reason="xgboost_ray not installed")
async def test_mars_with_xgboost_sklearn_clf(ray_large_cluster, create_cluster):
    from xgboost_ray import RayDMatrix, RayParams, RayXGBClassifier
    from sklearn.datasets import load_breast_cancer

    assert create_cluster.session
    session = new_session(address=create_cluster.address, backend="oscar", default=True)
    with session:
        train_x, train_y = load_breast_cancer(return_X_y=True, as_frame=True)
        df: md.DataFrame = md.concat(
            [md.DataFrame(train_x), md.DataFrame(train_y)], axis=1
        )
        df.execute()
        columns = list(df.columns.to_pandas())
        print(f"Columns {columns}, pandas columns {train_x.columns}")
        assert columns[:-1] == list(train_x.columns)
        num_shards = 4
        ds = md.to_ray_dataset(df, num_shards)
        assert isinstance(ds, ray_dataset.Dataset)
        print(f"Columns {columns}, dataset columns {train_x.columns}")
        assert columns == ds.schema().names
        import gc

        gc.collect()  # Ensure MLDataset does hold mars dataframe to avoid gc.
        ray_params = RayParams(num_actors=2, cpus_per_actor=1)
        clf = RayXGBClassifier(
            ray_params=ray_params,
            random_state=42,
            use_label_encoder=False,
            num_class=2,
        )
        # train
        clf.fit(RayDMatrix(ds, "target"), y=None, ray_params=ray_params)
        clf.predict(RayDMatrix(ds, "target"))
        # Enable it when https://github.com/ray-project/xgboost_ray/issues/177 got fixed
        # pred = clf.predict(train_x)
        # print("predicted values: ", pred)


@require_ray
@pytest.mark.parametrize(
    "ray_large_cluster", [{"num_nodes": 3, "num_cpus": 16}], indirect=True
)
@pytest.mark.asyncio
@pytest.mark.skipif(sklearn is None, reason="sklearn not installed")
@pytest.mark.skipif(xgboost_ray is None, reason="xgboost_ray not installed")
async def test_mars_with_xgboost_sklearn_reg(ray_large_cluster, create_cluster):
    from xgboost_ray import RayDMatrix, RayParams, RayXGBRegressor
    from sklearn.datasets import make_regression

    assert create_cluster.session
    session = new_session(address=create_cluster.address, backend="oscar", default=True)
    with session:
        np_X, np_y = make_regression(n_samples=1_0000, n_features=10)
        X, y = md.DataFrame(np_X), md.DataFrame({"target": np_y})
        df: md.DataFrame = md.concat([md.DataFrame(X), md.DataFrame(y)], axis=1)
        df.execute()

        num_shards = 4
        ds = md.to_ray_dataset(df, num_shards)
        assert isinstance(ds, ray_dataset.Dataset)

        import gc

        gc.collect()  # Ensure MLDataset does hold mars dataframe to avoid gc.
        ray_params = RayParams(num_actors=2, cpus_per_actor=1)
        reg = RayXGBRegressor(ray_params=ray_params, random_state=42)
        # train
        reg.fit(RayDMatrix(ds, "target"), y=None, ray_params=ray_params)
        reg.predict(RayDMatrix(ds, "target"))
        reg.predict(pd.DataFrame(np_X))
