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

from .utils import WorkerActor


class CustomLogFetchActor(WorkerActor):
    def __init__(self):
        super().__init__()
        self._dispatch_ref = None

    def post_create(self):
        from .dispatcher import DispatchActor

        self._dispatch_ref = self.ctx.actor_ref(DispatchActor.default_uid())
        self._dispatch_ref.register_free_slot(self.uid, 'custom_log')

    def fetch_logs(self, log_paths, offsets=None, sizes=None):
        result = []
        for i, log_path in enumerate(log_paths):
            log_result = dict()

            if offsets is not None:
                offset = offsets[i]
            else:
                offset = 0
            if sizes is not None:
                size = sizes[i]
            else:
                size = -1

            with self.ctx.fileobject(log_path, mode='r') as f:
                if offset:
                    f.seek(offset)

                log_result['log'] = f.read(size)
                log_result['offset'] = f.tell()

            result.append(log_result)

        return result
