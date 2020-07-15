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
from collections import deque, OrderedDict

from .utils import WorkerActor
from .. import promise
from ..actors import ActorNotExist
from ..utils import mod_hash, log_unhandled

logger = logging.getLogger(__name__)


class DispatchActor(WorkerActor):
    """
    Dispatcher for multiple actors belonging to one category
    """
    def __init__(self):
        super().__init__()
        self._free_slots = dict()
        self._all_slots = dict()
        self._free_slot_requests = dict()
        self._status_ref = None

    def post_create(self):
        super().post_create()
        try:
            self.set_cluster_info_ref()
        except ActorNotExist:
            pass

        from .status import StatusActor
        self._status_ref = self.ctx.actor_ref(StatusActor.default_uid())
        if not self.ctx.has_actor(self._status_ref):
            self._status_ref = None

    def _safe_tell_promise_slot(self, callback, queue_name, slot):
        try:
            self.tell_promise(callback, slot)
        except (ActorNotExist, BrokenPipeError, ConnectionRefusedError,
                TimeoutError, promise.PromiseTimeout) as ex:
            from ..scheduler import ResourceActor
            logger.exception('Failed to tell slot %s of queue %s into promise %r',
                             slot, queue_name, callback)

            callback_addr = callback[0][-1]
            if callback_addr != self.address and self.get_schedulers() and \
                    not isinstance(ex, (TimeoutError, promise.PromiseTimeout)):
                self.get_actor_ref(ResourceActor.default_uid()).detach_dead_workers([callback_addr], _tell=True)

            if slot is not None:
                self.ref().register_free_slot(slot, queue_name, _tell=True)

    @log_unhandled
    def acquire_free_slot(self, queue_name, callback=None):
        """
        Acquire uid of a free actor when available
        :param queue_name: queue name
        :param callback: promise callback
        """
        if queue_name not in self._free_slots:
            self._safe_tell_promise_slot(callback, queue_name, None)
            return
        if not self._free_slots[queue_name]:
            # no slots free, we queue the callback
            self._free_slot_requests[queue_name].append(callback)
            logger.debug('Slots not enough for queue %s on %s. slot dump: %r',
                         queue_name, self.address, self.get_free_slots_num())
            return

        free_slot = self._free_slots[queue_name].popitem()[0]
        logger.debug('Slot %s allocated for queue %s on %s. slot dump: %r',
                     free_slot, queue_name, self.address, self.get_free_slots_num())
        self._safe_tell_promise_slot(callback, queue_name, free_slot)

        if self._status_ref is not None:
            self._status_ref.update_slots({queue_name: len(self._free_slots[queue_name])},
                                          _tell=True, _wait=False)

    @log_unhandled
    def get_hash_slot(self, queue_name, key):
        """
        Get uid of a slot by hash value
        :param queue_name: queue name
        :param key: key to be hashed
        """
        slots = list(self._all_slots[queue_name])
        uid = slots[mod_hash(key, len(slots))]
        return uid

    @log_unhandled
    def get_hash_slots(self, queue_name, keys):
        slots = list(self._all_slots[queue_name])
        uids = [slots[mod_hash(key, len(slots))] for key in keys]
        return uids

    @log_unhandled
    def get_slots(self, queue_name):
        """
        Get all uids of slots of a queue
        :param queue_name: queue name
        """
        if queue_name not in self._all_slots:
            return []
        return list(self._all_slots[queue_name].keys())

    def get_free_slots_num(self):
        """
        Get number of free slots of every queue
        """
        return dict((k, len(v)) for k, v in self._free_slots.items())

    @log_unhandled
    def register_free_slot(self, uid, queue_name):
        """
        Register free uid of a queue
        :param uid: uid of free actor
        :param queue_name: queue name
        """
        if queue_name not in self._free_slots:
            self._free_slot_requests[queue_name] = deque()
            self._free_slots[queue_name] = OrderedDict()
            self._all_slots[queue_name] = OrderedDict()
        self._free_slots[queue_name][uid] = ''
        self._all_slots[queue_name][uid] = ''

        logger.debug('Slot %s registered for queue %s on %s', uid, queue_name, self.address)

        if self._free_slot_requests[queue_name]:
            free_slot = self._free_slots[queue_name].popitem()[0]
            logger.debug('Slot %s allocated for queue %s on %s. slot dump: %r',
                         free_slot, queue_name, self.address, self.get_free_slots_num())
            self._safe_tell_promise_slot(self._free_slot_requests[queue_name].popleft(), queue_name, free_slot)

        if self._status_ref is not None:
            self._status_ref.update_slots({queue_name: len(self._free_slots[queue_name])},
                                          _tell=True, _wait=False)

    @log_unhandled
    def remove_slot(self, uid, queue_name):
        self._free_slots[queue_name].pop(uid, None)
        self._all_slots[queue_name].pop(uid, None)
