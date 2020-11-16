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

import logging
import os
import random
import time
from collections import defaultdict, deque
from typing import NamedTuple, Union

from ...config import options
from ...errors import SpillNotConfigured
from ...utils import tokenize, mod_hash
from ..utils import WorkerActor, parse_spill_dirs

logger = logging.getLogger(__name__)


def _build_file_path_by_name(session_id, filename):
    dirs = options.worker.spill_directory
    dir_name = dirs[mod_hash(filename, len(dirs))]
    return os.path.join(dir_name, str(session_id), filename)


class DataMeta(NamedTuple):
    filename: str
    start: int
    end: Union[int, None]


class DiskFileMergerActor(WorkerActor):
    def __init__(self):
        super().__init__()

        dirs = options.worker.spill_directory = parse_spill_dirs(options.worker.spill_directory)
        if not dirs:  # pragma: no cover
            raise SpillNotConfigured

        self._key_to_data_meta = dict()
        self._file_to_sizes = defaultdict(lambda: 0)
        self._file_to_keys = defaultdict(list)
        self._available_files = []
        self._concurrency = options.worker.filemerger.concurrency
        self._max_file_size = options.worker.filemerger.max_file_size

        self._reading_files = set()
        self._writing_files = set()

        self._pending_read_requests = deque()
        self._pending_write_requests = deque()

    def await_file_reader(self, session_id, data_key, with_lock=False, callback=None):
        if with_lock and len(self._reading_files) >= self._concurrency:
            self._pending_read_requests.append((session_id, data_key, callback))
            return
        self._reading_files.add((session_id, data_key))
        self.tell_promise(callback)

    def await_file_writer(self, session_id, with_lock=True, callback=None):
        if with_lock and len(self._writing_files) >= self._concurrency:
            self._pending_write_requests.append(callback)
            return

        if self._available_files:
            filename = self._available_files.pop(-1)
        else:
            filename = _build_file_path_by_name(
                session_id, tokenize(callback, time.time(), random.random()))

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self._writing_files.add(filename)
        self.tell_promise(callback, filename, self._file_to_sizes[filename])

    def release_file_reader(self, session_id, data_key):
        self._reading_files.difference_update([(session_id, data_key)])
        if self._pending_read_requests:
            session_id, data_key, callback = self._pending_read_requests.popleft()
            self.await_file_reader(session_id, data_key, callback)

    def release_file_writer(self, session_id, data_key, filename, start, end):
        self._writing_files.remove(filename)
        self._key_to_data_meta[(session_id, data_key)] = DataMeta(filename, start, end)
        self._file_to_sizes[filename] = end
        self._file_to_keys[filename].append((session_id, data_key))

        if self._file_to_sizes[filename] >= self._max_file_size:
            self._file_to_sizes.pop(filename)
        else:
            self._available_files.append(filename)

        if self._pending_write_requests:
            self.await_file_writer(session_id, callback=self._pending_write_requests.popleft())

    def get_file_metas(self, session_id, data_keys):
        metas = []
        for key in data_keys:
            meta = self._key_to_data_meta.get((session_id, key))
            metas.append(meta)
        return metas

    def delete_file_metas(self, session_id, data_keys):
        files_to_delete = []
        filtered_keys = []
        for key in data_keys:
            meta = self._key_to_data_meta.pop((session_id, key), None)
            if meta is None:  # pragma: no cover
                continue
            filtered_keys.append(key)
            filename = meta.filename
            keys_set = self._file_to_keys[filename]
            keys_set.remove((session_id, key))
            if not keys_set:
                if filename in self._writing_files:
                    continue
                del self._file_to_keys[filename]
                self._file_to_sizes.pop(filename, None)
                files_to_delete.append(filename)

        del_set = set(files_to_delete)
        if del_set:
            self._available_files = [f for f in self._available_files if f not in del_set]
        return files_to_delete, filtered_keys

    def dump_info(self):
        return self._key_to_data_meta, self._file_to_sizes, self._file_to_keys, \
            self._writing_files
