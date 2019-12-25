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

import itertools
from collections import defaultdict

from ..utils import WorkerActor


class DataAttrs(object):
    __slots__ = 'size', 'shape'

    def __init__(self, size=None, shape=None):
        self.size = size
        self.shape = shape

    def __reduce__(self):
        return DataAttrs, (self.size, self.shape)

    def __repr__(self):  # pragma: no cover
        cls = type(self)
        return '<%s.%s size=%r shape=%r at 0x%x>' \
               % (cls.__module__, cls.__name__, self.size, self.shape, id(self))


class StorageManagerActor(WorkerActor):
    def __init__(self):
        super().__init__()
        self._data_to_locations = dict()
        self._data_attrs = dict()
        self._proc_to_data = defaultdict(set)

        self._proc_holders = dict()

    def post_create(self):
        super().post_create()

        from ..daemon import WorkerDaemonActor
        daemon_ref = self.ctx.actor_ref(WorkerDaemonActor.default_uid())
        if self.ctx.has_actor(daemon_ref):  # pragma: no branch
            daemon_ref.register_process_callback(
                self.ref(), self.handle_process_down.__name__)

    def register_process_holder(self, proc_id, device, holder_ref):
        self._proc_holders[(proc_id, device)] = holder_ref

    def get_process_holder(self, proc_id, device):
        return self._proc_holders[(proc_id, device)]

    def register_data(self, session_id, data_keys, location, sizes, shapes=None):
        shapes = shapes or itertools.repeat(None)
        for key, size, shape in zip(data_keys, sizes, shapes):
            session_data_key = (session_id, key)
            try:
                location_set = self._data_to_locations[session_data_key]
            except KeyError:
                location_set = self._data_to_locations[session_data_key] = set()

            location_set.add(location)
            try:
                attrs = self._data_attrs[session_data_key]
                attrs.size = max(size, attrs.size)
                if shape:
                    attrs.shape = shape
            except KeyError:
                attrs = DataAttrs(size, shape)
            self._data_attrs[session_data_key] = attrs
            if location[0] > 0:
                self._proc_to_data[location[0]].add(session_data_key)

    def unregister_data(self, session_id, data_keys, location):
        for data_key in data_keys:
            session_data_key = (session_id, data_key)
            if location[0] > 0:
                self._proc_to_data[location[0]].difference_update([session_data_key])
            try:
                location_set = self._data_to_locations[session_data_key]
                location_set.difference_update([location])
                if not location_set:
                    del self._data_to_locations[session_data_key]
                    self._data_attrs.pop(session_data_key, None)
            except KeyError:
                pass

    def get_data_locations(self, session_id, data_keys):
        return [set(self._data_to_locations.get((session_id, key)) or ()) for key in data_keys]

    def get_data_sizes(self, session_id, data_keys):
        return [a.size if a is not None else None
                for a in self.get_data_attrs(session_id, data_keys)]

    def get_data_shapes(self, session_id, data_keys):
        return [a.shape if a is not None else None
                for a in self.get_data_attrs(session_id, data_keys)]

    def get_data_attrs(self, session_id, data_keys):
        res = [None] * len(data_keys)
        for idx, k in enumerate(data_keys):
            try:
                res[idx] = self._data_attrs[(session_id, k)]
            except KeyError:
                pass
        return res

    def filter_exist_keys(self, session_id, data_keys, devices):
        devices = set(devices)
        keys = []
        for k in data_keys:
            try:
                if devices & self._data_to_locations[(session_id, k)]:
                    keys.append(k)
            except KeyError:
                pass
        return keys

    def handle_process_down(self, proc_indices):
        affected_keys = set()
        for proc_id in proc_indices:
            affected_keys.update(self._proc_to_data[proc_id])
            del self._proc_to_data[proc_id]
        proc_indices_set = set(proc_indices)
        for k in affected_keys:
            affected_locs = [loc for loc in self._data_to_locations[k]
                             if loc[0] in proc_indices_set]
            location_set = self._data_to_locations[k]
            location_set.difference_update(affected_locs)
            if not location_set:
                del self._data_to_locations[k]
                self._data_attrs.pop(k, None)

    def dump_keys(self):
        return list(self._data_attrs.keys())
