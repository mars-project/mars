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
from typing import List

import cloudpickle
import numpy as np
import pandas as pd

import mars
import mars.remote as mr
from mars.core.context import get_context
from mars.utils import Timer, readable_size


def send_1_to_1(n: int = None, n_out: int = 1):
    ctx = get_context()
    workers = ctx.get_worker_addresses()

    worker_to_gen_data = {
        w: mr.spawn(
            _gen_data,
            kwargs=dict(n=n, worker=w, n_out=n_out),
            expect_worker=w,
            n_output=n_out,
        )
        for i, w in enumerate(workers)
    }
    all_data = mars.execute(list(itertools.chain(*worker_to_gen_data.values())))
    progress = 0.1
    ctx.set_progress(progress)
    infos = np.array(
        [d._fetch_infos(fields=["data_key", "store_size"]) for d in all_data],
        dtype=object,
    )
    data_size = sum(info["store_size"][0] for info in infos[:n_out])
    worker_to_data_keys = dict()
    for worker, infos in zip(workers, np.split(infos, len(infos) // n_out)):
        worker_to_data_keys[worker] = [info["data_key"][0] for info in infos]

    workers_to_durations = dict()
    size = len(workers) * (len(workers) - 1)
    for worker1, worker2 in itertools.permutations(workers, 2):
        fetch_data = mr.spawn(
            _fetch_data,
            args=(worker_to_data_keys[worker1],),
            kwargs=dict(worker=worker2),
            expect_worker=worker2,
        )
        fetch_time = fetch_data.execute().fetch()
        rate = readable_size(data_size / fetch_time)
        workers_to_durations[worker1, worker2] = (
            readable_size(data_size),
            f"{rate}B/s",
        )
        progress += 0.9 / size
        ctx.set_progress(min(progress, 1.0))
    return workers_to_durations


def _gen_data(
    n: int = None, worker: str = None, check_addr: bool = True, n_out: int = 1
) -> List[pd.DataFrame]:
    if check_addr:
        ctx = get_context()
        assert ctx.worker_address == worker
    rs = np.random.RandomState(123)

    outs = []
    for _ in range(n_out):
        n = n if n is not None else 5_000_000
        data = {
            "a": rs.rand(n),
            "b": rs.randint(n * 10, size=n),
            "c": [f"foo{i}" for i in range(n)],
        }
        outs.append(pd.DataFrame(data))
    return outs


def _fetch_data(data_keys: List[str], worker: str = None):
    # do nothing actually
    ctx = get_context()
    assert ctx.worker_address == worker
    with Timer() as timer:
        ctx.get_chunks_result(data_keys, fetch_only=True)
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

    def time_1_to_1_small_objects(self):
        return mr.spawn(send_1_to_1, kwargs=dict(n=1_000, n_out=100)).execute().fetch()


if __name__ == "__main__":
    suite = TransferPackageSuite()
    suite.setup()
    print("- Bench 1 to 1 -")
    print(suite.time_1_to_1())
    print("- Bench 1 to 1 with small objects -")
    print(suite.time_1_to_1_small_objects())
    suite.teardown()
