# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

import functools
import logging
import time
from collections import deque

from ... import promise
from ...config import options
from ...utils import log_unhandled
from ..utils import WorkerActor

logger = logging.getLogger(__name__)


class IORunnerActor(WorkerActor):
    """
    Actor handling spill read and write in single disk partition
    """
    _io_runner = True

    def __init__(self):
        super(IORunnerActor, self).__init__()
        self._work_items = deque()
        self._max_work_item_id = 0
        self._cur_work_items = dict()
        self._lock_free = options.worker.lock_free_fileio
        self._lock_work_item_id = dict()
        self._exec_start_time = None

    def post_create(self):
        super(IORunnerActor, self).post_create()

        from ..dispatcher import DispatchActor
        dispatch_ref = self.ctx.actor_ref(DispatchActor.default_uid())
        dispatch_ref.register_free_slot(self.uid, 'iorunner')

    @promise.reject_on_exception
    @log_unhandled
    def load_from(self, dest_device, session_id, data_key, src_device, callback):
        logger.debug('Copying (%s, %s) from %s into %s submitted in %s',
                     session_id, data_key, src_device, dest_device, self.uid)
        self._work_items.append((dest_device, session_id, data_key, src_device, False, callback))
        if self._lock_free or not self._cur_work_items:
            self._submit_next()

    def lock(self, session_id, data_key, callback):
        logger.debug('Requesting lock for (%s, %s) on %s', session_id, data_key, self.uid)
        self._work_items.append((None, session_id, data_key, None, True, callback))
        if self._lock_free or not self._cur_work_items:
            self._submit_next()

    def unlock(self, session_id, data_key):
        logger.debug('%s unlocked for (%s, %s)', self.uid, session_id, data_key)
        work_item_id = self._lock_work_item_id.pop((session_id, data_key), None)
        if work_item_id is None:
            return
        self._cur_work_items.pop(work_item_id)
        self._submit_next()

    @log_unhandled
    def _submit_next(self):
        if not self._work_items:
            return
        work_item_id = self._max_work_item_id
        self._max_work_item_id += 1
        dest_device, session_id, data_key, src_device, is_lock, cb = \
            self._cur_work_items[work_item_id] = self._work_items.popleft()

        if is_lock:
            self._lock_work_item_id[(session_id, data_key)] = work_item_id
            self.tell_promise(cb)
            logger.debug('%s locked for (%s, %s)', self.uid, session_id, data_key)
            return

        @log_unhandled
        def _finalize(exc, *_):
            del self._cur_work_items[work_item_id]
            self._exec_start_time = None
            if not exc:
                self.tell_promise(cb)
            else:
                self.tell_promise(cb, *exc, **dict(_accept=False))
            self._submit_next()

        logger.debug('Start copying (%s, %s) from %s into %s in %s',
                     session_id, data_key, src_device, dest_device, self.uid)
        self._exec_start_time = time.time()
        src_handler = self.storage_client.get_storage_handler(src_device)
        dest_handler = self.storage_client.get_storage_handler(dest_device)
        dest_handler.load_from(session_id, data_key, src_handler) \
            .then(functools.partial(_finalize, None), lambda *exc: _finalize(exc))
