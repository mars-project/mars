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

import pandas as pd

from .... import remote as mr
from ....deploy.oscar.session import get_default_session
from ....utils import ceildiv

try:
    try:
        # fix for tsfresh 0.17.0, from this version on,
        # we need to inherit from IterableDistributorBaseClass
        from tsfresh.utilities.distribution import IterableDistributorBaseClass \
            as DistributorBaseClass
    except ImportError:  # pragma: no cover
        from tsfresh.utilities.distribution import DistributorBaseClass
except ImportError:  # pragma: no cover
    DistributorBaseClass = object


class MarsDistributor(DistributorBaseClass):
    def __init__(self, session=None):
        self._session = session or get_default_session()

    def calculate_best_chunk_size(self, data_length):
        n_cores = self._session.get_total_n_cpu()
        return ceildiv(data_length, n_cores)

    def distribute(self, func, partitioned_chunks, kwargs):
        def _wrapped_func(*args, **kw):
            # Series.value_counts() may not be able to handle
            if not getattr(pd.Series.value_counts, '_wrapped', False):
                old_value_counts = pd.Series.value_counts

                def _wrapped_value_counts(obj, *args, **kw):
                    try:
                        return old_value_counts(obj, *args, **kw)
                    except ValueError:  # pragma: no cover
                        return old_value_counts(obj.copy(), *args, **kw)

                pd.Series.value_counts = _wrapped_value_counts
                pd.Series.value_counts._wrapped = True

            return func(*args, **kw)

        tasks = []
        for partitioned_chunk in partitioned_chunks:
            tasks.append(mr.spawn(_wrapped_func, args=(partitioned_chunk,), kwargs=kwargs))
        executed = mr.ExecutableTuple(tasks).execute(session=self._session)
        fetched = executed.fetch(session=self._session)
        return [item for results in fetched for item in results]
