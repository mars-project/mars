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

import concurrent.futures

from .... import remote
from ....deploy.oscar.session import get_default_session, new_session

try:
    from joblib.parallel import ParallelBackendBase, AutoBatchingMixin, \
        register_parallel_backend
except ImportError:
    ParallelBackendBase = object
    AutoBatchingMixin = object
    register_parallel_backend = None


class MarsDistributedBackend(AutoBatchingMixin, ParallelBackendBase):
    MIN_IDEAL_BATCH_DURATION = 0.2
    MAX_IDEAL_BATCH_DURATION = 1.0
    supports_timeout = True

    def __init__(self, service=None, session=None, backend=None, n_parallel=None):
        super().__init__()

        if session is None:
            if service is not None:
                self.session = new_session(service, backend=backend, default=False)
            else:
                self.session = get_default_session()
        else:
            self.session = session

        self.n_parallel = n_parallel or 1
        self.executor = None

    def get_nested_backend(self):
        return MarsDistributedBackend(session=self.session), -1

    def configure(self, n_jobs=1, parallel=None, **backend_args):
        self.parallel = parallel
        n_parallel = self.effective_n_jobs(n_jobs)
        self.executor = concurrent.futures.ThreadPoolExecutor(n_parallel)
        return n_parallel

    def effective_n_jobs(self, n_jobs):
        eff_n_jobs = super(MarsDistributedBackend, self).effective_n_jobs(n_jobs)
        if n_jobs == -1:
            eff_n_jobs = self.n_parallel
        return eff_n_jobs

    def apply_async(self, func, callback=None):
        # todo allow execute f() in remote end to reduce data copy latency
        def f():
            spawned = []
            for func_obj, args, kwargs in func.items:
                spawned.append(remote.spawn(func_obj, args=args, kwargs=kwargs))

            ret = remote.ExecutableTuple(spawned) \
                .execute(session=self.session) \
                .fetch(self.session)
            callback(ret)
            return ret

        future = self.executor.submit(f)
        future.get = future.result
        return future


def register_mars_backend():
    register_parallel_backend('mars', MarsDistributedBackend)
