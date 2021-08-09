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
import pytest

from mars import new_session, stop_server
from mars.contrib.dask import convert_dask_collection, mars_scheduler

try:
    import dask
except ImportError:
    dask = None


def setup_function():
    new_session()


def teardown_function():
    stop_server()


def require_dask_installed(func):
    return pytest.mark.skipif(dask is None, reason='dask not installed')(func)


@require_dask_installed
def test_delayed():
    from dask import delayed
    import numpy as np

    def calc_chunk(n: int, i: int):
        rs = np.random.RandomState(i)
        a = rs.uniform(-1, 1, size=(n, 2))
        d = np.linalg.norm(a, axis=1)
        return (d < 1).sum()

    def calc_pi(fs, N):
        return sum(fs) * 4 / N

    N = 200_000_000
    n = 10_000_000

    fs = [delayed(calc_chunk)(n, i) for i in range(N // n)]
    pi = delayed(calc_pi)(fs, N)

    dask_res = pi.compute()
    assert dask_res == pi.compute(scheduler=mars_scheduler)
    assert dask_res == convert_dask_collection(pi).execute().fetch()


@require_dask_installed
def test_partitioned_dataframe():
    import numpy as np
    import pandas as pd
    from dask import dataframe as dd
    from pandas._testing import assert_frame_equal

    data = np.random.randn(10000, 100)
    df = dd.from_pandas(
        pd.DataFrame(data, columns=[f"col{i}" for i in range(100)]), npartitions=4
    )
    df["col0"] = df["col0"] + df["col1"] / 2
    col2_mean = df["col2"].mean()
    df = df[df["col2"] > col2_mean]

    dask_res = df.compute()
    assert_frame_equal(dask_res, df.compute(scheduler=mars_scheduler), check_index_type=False)
    assert_frame_equal(dask_res, convert_dask_collection(df).execute().fetch(), check_index_type=False)


@require_dask_installed
def test_unpartitioned_dataframe():
    from dask import dataframe as dd
    from pandas._testing import assert_frame_equal
    import pandas as pd
    from sklearn.datasets import load_boston

    boston = load_boston()
    pd.DataFrame(boston.data, columns=boston['feature_names']).to_csv("./boston_housing_data.csv")

    df = dd.read_csv(r"./boston_housing_data.csv")
    df["CRIM"] = df["CRIM"] / 2

    dask_res = df.compute()
    assert_frame_equal(dask_res, df.compute(scheduler=mars_scheduler))
    assert_frame_equal(dask_res, convert_dask_collection(df).execute().fetch())


@require_dask_installed
def test_array():
    import dask.array as da
    from numpy.core.numeric import array_equal

    x = da.random.random((10000, 10000), chunks=(1000, 1000))
    y = x + x.T
    z = y[::2, 5000:].mean(axis=1)

    dask_res = z.compute()
    assert array_equal(dask_res, z.compute(scheduler=mars_scheduler))
    assert array_equal(dask_res, convert_dask_collection(z).execute().fetch())


@require_dask_installed
def test_bag():
    import dask

    b = dask.datasets.make_people()  # Make records of people
    result = (
        b.filter(lambda record: record["age"] > 30)
            .map(lambda record: record["occupation"])
            .frequencies(sort=True)
            .topk(10, key=1)
    )

    dask_res = result.compute()
    assert dask_res == result.compute(scheduler=mars_scheduler)
    assert dask_res == list(
        convert_dask_collection(result).execute().fetch()
    )  # TODO: dask-bag computation will return weird tuple, which we don't know why
