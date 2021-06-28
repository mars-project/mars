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

import os
import shutil
import tempfile
from typing import Any, Dict, List

from .... import oscar as mo
from ....lib.aio import AioFileObject


class CustomLogActor(mo.Actor):
    def __init__(self,
                 custom_log_dir: str):
        self._custom_log_dir = custom_log_dir

    @staticmethod
    def _get_custom_log_dir(custom_log_dir: str, session_id: str):
        if custom_log_dir == 'auto':
            return tempfile.mkdtemp(prefix=f'marslog-{session_id}')
        elif custom_log_dir is None:
            return
        else:
            return os.path.join(custom_log_dir, session_id)

    def new_custom_log_dir(self, session_id: str):
        custom_log_dir = self._get_custom_log_dir(
            self._custom_log_dir, session_id)
        if custom_log_dir:
            os.makedirs(custom_log_dir, exist_ok=True)
            return custom_log_dir

    @classmethod
    def clear_custom_log_dirs(cls, paths: List[str]):
        _ = [shutil.rmtree(path, ignore_errors=True) for path in paths]

    @classmethod
    async def fetch_logs(cls,
                         log_paths: List[str],
                         offsets: List[int],
                         sizes: List[int]) -> List[Dict[str, Any]]:
        result = []
        for i, log_path in enumerate(log_paths):
            log_result = dict()

            offset = offsets[i]
            size = sizes[i]

            async with AioFileObject(open(log_path)) as f:
                if offset < 0:
                    # process negative offset
                    offset = max(os.path.getsize(log_path) + offset, 0)

                if offset:
                    await f.seek(offset)

                log_result['log'] = await f.read(size)
                log_result['offset'] = await f.tell()

            result.append(log_result)

        return result
