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

from .utils import SchedulerActor


class CustomLogMetaActor(SchedulerActor):
    def __init__(self):
        super().__init__()

        self._custom_log_path_store = dict()

    def record_custom_log_path(self, session_id: str, tileable_op_key: str,
                               chunk_op_key: str, worker_address: str,
                               log_path: str):
        key = (session_id, tileable_op_key)
        if key not in self._custom_log_path_store:
            self._custom_log_path_store[key] = dict()
        self._custom_log_path_store[key][chunk_op_key] = (worker_address, log_path)

    def get_tileable_op_log_paths(self, session_id: str, tileable_op_key: str):
        return self._custom_log_path_store.get((session_id, tileable_op_key))
