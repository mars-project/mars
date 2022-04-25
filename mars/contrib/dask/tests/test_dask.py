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

from ....utils import lazy_import
from .. import convert_dask_collection, mars_scheduler

dask_installed = lazy_import("dask", globals=globals()) is not None
mimesis_installed = lazy_import("mimesis", globals=globals()) is not None


@pytest.mark.skipif(not dask_installed, reason="dask not installed")
def test_delayed(setup_cluster):
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


@pytest.mark.skipif(not dask_installed, reason="dask not installed")
def test_partitioned_dataframe(setup_cluster):
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
    assert_frame_equal(
        dask_res, df.compute(scheduler=mars_scheduler), check_index_type=False
    )
    assert_frame_equal(
        dask_res, convert_dask_collection(df).execute().fetch(), check_index_type=False
    )


@pytest.mark.skipif(not dask_installed, reason="dask not installed")
def test_unpartitioned_dataframe(setup_cluster):
    from dask import dataframe as dd
    from pandas._testing import assert_frame_equal
    import pandas as pd
    from sklearn.datasets import load_boston

    boston = load_boston()
    pd.DataFrame(boston.data, columns=boston["feature_names"]).to_csv(
        "./boston_housing_data.csv"
    )

    df = dd.read_csv(r"./boston_housing_data.csv")
    df["CRIM"] = df["CRIM"] / 2

    dask_res = df.compute()
    assert_frame_equal(dask_res, df.compute(scheduler=mars_scheduler))
    assert_frame_equal(dask_res, convert_dask_collection(df).execute().fetch())


@pytest.mark.skipif(not dask_installed, reason="dask not installed")
def test_array(setup_cluster):
    import dask.array as da
    from numpy.core.numeric import array_equal

    x = da.random.random((10000, 10000), chunks=(1000, 1000))
    y = x + x.T
    z = y[::2, 5000:].mean(axis=1)

    dask_res = z.compute()
    assert array_equal(dask_res, z.compute(scheduler=mars_scheduler))
    assert array_equal(dask_res, convert_dask_collection(z).execute().fetch())


@pytest.mark.skipif(not dask_installed, reason="dask not installed")
@pytest.mark.skipif(not mimesis_installed, reason="mimesis not installed")
def test_bag(setup_cluster):
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
    assert dask_res == convert_dask_collection(result).execute().fetch()


@pytest.mark.skipif(not dask_installed, reason="dask not installed")
def test_dask_errors():
    with pytest.raises(TypeError):
        convert_dask_collection({"foo": 0, "bar": 1})


@pytest.mark.skipif(not dask_installed, reason="dask not installed")
def test_multiple_objects(setup_cluster):
    import dask

    def inc(x: int):
        return x + 1

    test_list = [dask.delayed(inc)(i) for i in range(10)]
    test_tuple = tuple(dask.delayed(inc)(i) for i in range(10))
    test_dict = {str(i): dask.delayed(inc)(i) for i in range(10)}

    for test_obj in (test_list, test_tuple, test_dict):
        assert dask.compute(test_obj) == dask.compute(
            test_obj, scheduler=mars_scheduler
        )


@pytest.mark.skipif(not dask_installed, reason="dask not installed")
def test_persist(setup_cluster):
    import dask

    def inc(x):
        return x + 1

    a = dask.delayed(inc)(1)
    task_mars_persist = dask.delayed(inc)(a.persist(scheduler=mars_scheduler))
    task_dask_persist = dask.delayed(inc)(a.persist())

    assert task_dask_persist.compute() == task_mars_persist.compute(
        scheduler=mars_scheduler
    )


@pytest.mark.skipif(not dask_installed, reason="dask not installed")
def test_partitioned_dataframe_persist(setup_cluster):
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

    df_mars_persist = df[df["col2"] > col2_mean.persist(scheduler=mars_scheduler)]
    df_dask_persist = df[df["col2"] > col2_mean.persist()]

    assert_frame_equal(
        df_dask_persist.compute(), df_mars_persist.compute(scheduler=mars_scheduler)
    )
