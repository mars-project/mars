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

    @classmethod
    def gen_uid(cls, proc_id):
        return 'w:%d:io_runner_inproc' % proc_id

    def __init__(self, lock_free=False, dispatched=True):
        super().__init__()
        self._work_items = deque()
        self._max_work_item_id = 0
        self._cur_work_items = dict()

        self._lock_free = lock_free or options.worker.lock_free_fileio
        self._lock_work_items = dict()

        self._dispatched = dispatched

    def post_create(self):
        super().post_create()

        if self._dispatched:
            from ..dispatcher import DispatchActor
            dispatch_ref = self.ctx.actor_ref(DispatchActor.default_uid())
            dispatch_ref.register_free_slot(self.uid, 'iorunner')

    @promise.reject_on_exception
    @log_unhandled
    def load_from(self, dest_device, session_id, data_keys, src_device, callback):
        logger.debug('Copying %r from %s into %s submitted in %s',
                     data_keys, src_device, dest_device, self.uid)
        self._work_items.append((dest_device, session_id, data_keys, src_device, False, callback))
        if self._lock_free or not self._cur_work_items:
            self._submit_next()

    def lock(self, session_id, data_keys, callback):
        logger.debug('Requesting lock for %r on %s', data_keys, self.uid)
        self._work_items.append((None, session_id, data_keys, None, True, callback))
        if self._lock_free or not self._cur_work_items:
            self._submit_next()

    def unlock(self, work_item_id):
        data_keys = self._lock_work_items.pop(work_item_id)[2]
        logger.debug('%s unlocked for %r on work item %d', self.uid, data_keys, work_item_id)
        if work_item_id is not None:  # pragma: no branch
            self._cur_work_items.pop(work_item_id)
            self._submit_next()

    @log_unhandled
    def _submit_next(self):
        if not self._work_items:
            return
        work_item_id = self._max_work_item_id
        self._max_work_item_id += 1
        dest_device, session_id, data_keys, src_device, is_lock, cb = \
            self._cur_work_items[work_item_id] = self._work_items.popleft()

        if is_lock:
            self._lock_work_items[work_item_id] = self._cur_work_items[work_item_id]
            self.tell_promise(cb, work_item_id)
            logger.debug('%s locked for %r on work item %d', self.uid, data_keys, work_item_id)
            return

        @log_unhandled
        def _finalize(*exc):
            del self._cur_work_items[work_item_id]
            if not exc:
                self.tell_promise(cb)
            else:
                self.tell_promise(cb, *exc, _accept=False)
            self._submit_next()

        logger.debug('Start copying %r from %s into %s in %s',
                     data_keys, src_device, dest_device, self.uid)
        src_handler = self.storage_client.get_storage_handler(src_device)
        dest_handler = self.storage_client.get_storage_handler(dest_device)
        dest_handler.load_from(session_id, data_keys, src_handler) \
            .then(lambda *_: _finalize(), _finalize)
