# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

import multiprocessing

from .... import remote as mr
from ....session import Session
from ....utils import ceildiv

try:
    from tsfresh.utilities.distribution import DistributorBaseClass
except ImportError:
    DistributorBaseClass = object


class MarsDistributor(DistributorBaseClass):
    def __init__(self, session=None):
        self._session = session or Session.default_or_local()

    def calculate_best_chunk_size(self, data_length):
        if not hasattr(self._session, 'get_workers_meta'):
            return ceildiv(data_length, multiprocessing.cpu_count())
        else:
            metas = self._session.get_workers_meta()
            n_cores = sum(m['hardware']['cpu_total'] for m in metas.values())
            return ceildiv(data_length, n_cores)

    def distribute(self, func, partitioned_chunks, kwargs):
        tasks = []
        for partitioned_chunk in partitioned_chunks:
            tasks.append(mr.spawn(func, args=(partitioned_chunk,), kwargs=kwargs))
        executed = mr.ExecutableTuple(tasks).execute(session=self._session)
        fetched = executed.fetch(session=self._session)
        return [item for results in fetched for item in results]
