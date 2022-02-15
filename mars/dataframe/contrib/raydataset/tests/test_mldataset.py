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
ml_dataset = lazy_import("ray.util.data")
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
async def test_dataset_related_classes(ray_large_cluster):
    from ..mldataset import ChunkRefBatch

    # in order to pass checks
    value1 = np.random.rand(10, 10)
    value2 = np.random.rand(10, 10)
    df1 = pd.DataFrame(value1)
    df2 = pd.DataFrame(value2)
    if ray:
        obj_ref1, obj_ref2 = ray.put(df1), ray.put(df2)
        batch = ChunkRefBatch(shard_id=0, obj_refs=[obj_ref1, obj_ref2])
        assert batch.shard_id == 0
        # the first data in batch
        batch = iter(batch)
        pd.testing.assert_frame_equal(next(batch), df1)
        pd.testing.assert_frame_equal(next(batch), df2)


@require_ray
@pytest.mark.asyncio
@pytest.mark.parametrize("test_option", [[5, 5], [5, 4], [None, None]])
async def test_convert_to_ray_mldataset(ray_large_cluster, create_cluster, test_option):
    assert create_cluster.session
    session = new_session(address=create_cluster.address, backend="oscar", default=True)
    with session:
        value = np.random.rand(10, 10)
        chunk_size, num_shards = test_option
        df: md.DataFrame = md.DataFrame(value, chunk_size=chunk_size)
        df.execute()

        ds = mdd.to_ray_mldataset(df, num_shards=num_shards)
        assert isinstance(ds, ml_dataset.MLDataset)


@require_ray
@pytest.mark.asyncio
@pytest.mark.skipif(xgboost_ray is None, reason="xgboost_ray not installed")
async def test_mars_with_xgboost(ray_large_cluster, create_cluster):
    from xgboost_ray import RayDMatrix, RayParams, train, predict
    from sklearn.datasets import load_breast_cancer

    assert create_cluster.session
    session = new_session(address=create_cluster.address, backend="oscar", default=True)
    with session:
        train_x, train_y = load_breast_cancer(return_X_y=True, as_frame=True)
        df: md.DataFrame = md.concat(
            [md.DataFrame(train_x), md.DataFrame(train_y)], axis=1
        )
        df.execute()

        num_shards = 4
        ds = mdd.to_ray_mldataset(df, num_shards)
        assert isinstance(ds, ml_dataset.MLDataset)

        import gc

        gc.collect()  # Ensure MLDataset does hold mars dataframe to avoid gc.

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
        predict(bst, train_set, ray_params=RayParams(num_actors=2))
