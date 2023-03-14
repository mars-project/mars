# Copyright 1999-2022 Alibaba Group Holding Ltd.
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

import itertools

import cloudpickle
import numpy as np
import pandas as pd

import mars
import mars.remote as mr
from mars.core.context import get_context
from mars.utils import Timer, readable_size


def send_1_to_1(n: int = None, cpu: bool = True):
    ctx = get_context()
    bands = [
        b
        for b in ctx.get_worker_bands()
        if (cpu and b[1].startswith("numa-")) or (not cpu and b[1].startswith("gpu-"))
    ]

    band_to_gen_data = {
        b: mr.spawn(_gen_data, kwargs=dict(n=n, band=b), expect_band=b)
        for i, b in enumerate(bands)
    }
    all_data = mars.execute(list(band_to_gen_data.values()))
    progress = 0.1
    ctx.set_progress(progress)
    infos = [d._fetch_infos(fields=["data_key", "store_size"]) for d in all_data]
    data_size = infos[0]["store_size"][0]
    band_to_data_keys = dict(zip(bands, [info["data_key"][0] for info in infos]))

    bands_to_durations = dict()
    size = len(bands) * (len(bands) - 1)
    for band1, band2 in itertools.permutations(bands, 2):
        fetch_data = mr.spawn(
            _fetch_data,
            args=(band_to_data_keys[band1],),
            kwargs=dict(band=band2),
            expect_band=band2,
        )
        fetch_time = fetch_data.execute().fetch()
        rate = readable_size(data_size / fetch_time)
        bands_to_durations[band1, band2] = (
            readable_size(data_size),
            f"{rate}B/s",
        )
        progress += 0.9 / size
        ctx.set_progress(min(progress, 1.0))
    return bands_to_durations


def _gen_data(n: int = None, band: str = None, check_addr: bool = True) -> pd.DataFrame:
    if check_addr:
        ctx = get_context()
        assert ctx.band == band
    n = n if n is not None else 5_000_000
    rs = np.random.RandomState(123)
    data = {
        "a": rs.rand(n),
        "b": rs.randint(n * 10, size=n),
        "c": [f"foo{i}" for i in range(n)],
    }
    return pd.DataFrame(data)


def _fetch_data(data_key: str, band: str = None):
    # do nothing actually
    ctx = get_context()
    assert ctx.band == band
    with Timer() as timer:
        ctx.get_chunks_result([data_key], fetch_only=True)
    return timer.duration


class TransferPackageSuite:
    """
    Benchmark that times performance of storage transfer
    """

    def setup(self):
        try:
            # make sure all submodules will serial functions instead of refs
            cloudpickle.register_pickle_by_value(__import__("benchmarks.storage"))
        except (AttributeError, ImportError):
            pass
        mars.new_session(n_worker=2, n_cpu=8)

    def teardown(self):
        mars.stop_server()
        try:
            cloudpickle.unregister_pickle_by_value(__import__("benchmarks.storage"))
        except (AttributeError, ImportError):
            pass

    def time_1_to_1(self):
        return mr.spawn(send_1_to_1).execute().fetch()


if __name__ == "__main__":
    suite = TransferPackageSuite()
    suite.setup()
    print(suite.time_1_to_1())
    suite.teardown()
