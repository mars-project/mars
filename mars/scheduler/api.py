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
from .graph import GraphActor
from .chunkmeta import ChunkMetaClient
from .utils import SchedulerClusterInfoActor


class MetaAPI:
    def __init__(self, actor_ctx=None, scheduler_ip=None):
        self._actor_ctx = actor_ctx or new_client()
        self._cluster_info = self._actor_ctx.actor_ref(
            SchedulerClusterInfoActor.default_uid(), address=scheduler_ip)
        self._chunk_meta_client = ChunkMetaClient(self._actor_ctx, self._cluster_info)

    def get_tileable_metas(self, session_id, graph_key, tileable_keys, filter_fields: List[str]=None) -> List:
        graph_uid = GraphActor.gen_uid(session_id, graph_key)
        graph_ref = self._actor_ctx.actor_ref(graph_uid)
        return graph_ref.get_tileable_metas(tileable_keys, filter_fields=filter_fields, _wait=False)

    def get_chunk_metas(self, session_id, chunk_keys, filter_fields: List[str]=None) -> List:
        return self._chunk_meta_client.batch_get_chunk_meta(session_id, chunk_keys, filter_fields=filter_fields)

