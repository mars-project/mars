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
        self._cur_work_item = None
        self._exec_start_time = None

    def post_create(self):
        super(IORunnerActor, self).post_create()

        from ..dispatcher import DispatchActor
        dispatch_ref = self.ctx.actor_ref(DispatchActor.default_uid())
        dispatch_ref.register_free_slot(self.uid, 'iorunner')

        self.ref().daemon_io_process(_tell=True)

    @promise.reject_on_exception
    @log_unhandled
    def load_from(self, dest_device, session_id, data_key, src_device, callback):
        logger.debug('Copying (%s, %s) from %s into %s submitted in %s',
                     session_id, data_key, src_device, dest_device, self.uid)
        self._work_items.append((dest_device, session_id, data_key, src_device, False, callback))
        if self._cur_work_item is None:
            self._submit_next()

    def lock(self, session_id, data_key, callback):
        logger.debug('Requesting lock for (%s, %s) on %s', session_id, data_key, self.uid)
        self._work_items.append((None, session_id, data_key, None, True, callback))
        if self._cur_work_item is None:
            self._submit_next()

    def unlock(self, session_id, data_key):
        logger.debug('%s unlocked for (%s, %s)', self.uid, session_id, data_key)
        self._cur_work_item = None
        self._submit_next()

    def daemon_io_process(self):
        if self._exec_start_time is not None and time.time() > self._exec_start_time + 60:
            logger.warning('Work item %r in %s is taking a long time: %s seconds passed',
                           self._cur_work_item, self.uid, int(time.time() - self._exec_start_time))
        self.ref().daemon_io_process(_delay=10, _tell=True)

    @log_unhandled
    def _submit_next(self):
        if not self._work_items:
            return
        dest_device, session_id, data_key, src_device, is_lock, cb = \
            self._cur_work_item = self._work_items.popleft()

        if is_lock:
            self.tell_promise(cb)
            logger.debug('%s locked for (%s, %s)', self.uid, session_id, data_key)
            return

        @log_unhandled
        def _finalize(exc, *_):
            self._cur_work_item = None
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
