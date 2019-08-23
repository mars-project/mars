# Copyright 1999-2019 Alibaba Group Holding Ltd.
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

from collections import defaultdict

from ..utils import WorkerActor


class DataAttrs(object):
    __slots__ = 'size', 'shape'

    def __init__(self, size=None, shape=None):
        self.size = size
        self.shape = shape


class StorageManagerActor(WorkerActor):
    def __init__(self):
        super(StorageManagerActor, self).__init__()
        self._data_to_locations = dict()
        self._data_attrs = dict()
        self._proc_to_data = defaultdict(set)

        self._proc_holders = dict()

    def post_create(self):
        super(StorageManagerActor, self).post_create()

        from ..daemon import WorkerDaemonActor
        daemon_ref = self.ctx.actor_ref(WorkerDaemonActor.default_uid())
        if self.ctx.has_actor(daemon_ref):
            daemon_ref.register_process_callback(
                self.ref(), self.handle_process_down.__name__)

    def register_process_holder(self, proc_id, device, holder_ref):
        self._proc_holders[(proc_id, device)] = holder_ref

    def get_process_holder(self, proc_id, device):
        return self._proc_holders[(proc_id, device)]

    def register_data(self, session_id, data_key, location, size, shape=None):
        session_data_key = (session_id, data_key)
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

    def unregister_data(self, session_id, data_key, location):
        session_data_key = (session_id, data_key)
        if location[0] > 0:
            self._proc_to_data[location[0]].difference_update([session_data_key])
        try:
            self._data_to_locations[session_data_key].difference_update([location])
            if not self._data_to_locations[session_data_key]:
                del self._data_to_locations[session_data_key]
                try:
                    del self._data_attrs[session_data_key]
                except KeyError:
                    pass
        except KeyError:
            pass

    def get_data_locations(self, session_id, data_key):
        try:
            return self._data_to_locations[(session_id, data_key)]
        except KeyError:
            return None

    def get_data_size(self, session_id, data_key):
        sizes = self.get_data_sizes(session_id, [data_key])
        return sizes[data_key] if sizes else None

    def get_data_sizes(self, session_id, data_keys):
        res = dict()
        for k in data_keys:
            try:
                res[k] = self._data_attrs[(session_id, k)].size
            except KeyError:
                pass
        return res

    def get_data_shape(self, session_id, data_key):
        shapes = self.get_data_shapes(session_id, [data_key])
        return shapes[data_key] if shapes else None

    def get_data_shapes(self, session_id, data_keys):
        res = dict()
        for k in data_keys:
            try:
                res[k] = self._data_attrs[(session_id, k)].shape
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
                try:
                    del self._data_attrs[k]
                except KeyError:
                    pass

    def dump_keys(self):
        return list(self._data_attrs.keys())
