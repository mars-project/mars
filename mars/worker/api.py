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

from typing import List

from ..actors import new_client
from .transfer import ResultSenderActor


class WorkerAPI:
    def __init__(self, actor_ctx=None):
        self._actor_context = actor_ctx or new_client()

    def get_chunks_data(self, session_id, worker: str, chunk_keys: List[str], indexes: List = None,
                        compression_types: List[str] = None):
        """
        Fetch chunks data from a specified worker.
        :param session_id: session_id
        :param worker: worker address
        :param chunk_keys: chunk keys
        :param indexes: indexes on raw data
        :param compression_types: compression types which are acceptable for readers
        """
        sender_ref = self._actor_context.actor_ref(ResultSenderActor.default_uid(), address=worker)
        compression_type = max(compression_types) if compression_types else None
        return sender_ref.fetch_batch_data(session_id, chunk_keys, index_objs=indexes,
                                           compression_type=compression_type, _wait=False)
