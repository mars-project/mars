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

from datetime import datetime, timedelta
from urllib.parse import urlparse

from gevent.event import Event


def _normalize_path(path):
    while '//' in path:
        path = path.replace('//', '/')
    return path.rstrip('/') or '/'


class PathResult(object):
    def __init__(self, key, value=None, dir=False, children=None):
        self.key = key
        self.value = value
        self.dir = dir
        self.children = children or []

    def __repr__(self):
        arg_strs = ['key=%r' % self.key]
        if self.value:
            arg_strs.append('value=%r' % self.value)
        if self.dir:
            arg_strs.append('dir=%r' % self.dir)
        if self.children:
            arg_strs.append('children=%r' % self.children)
        return 'PathResult(' + ', '.join(arg_strs) + ')'


class LocalKVStore(object):
    def __init__(self):
        self._store = dict()
        self._expire_time = dict()
        self._children = {'/': set()}
        self._watch_event = {'/': set()}
        self._watch_event_r = {'/': set()}

    def _read_value(self, item):
        item = _normalize_path(item)
        if item in self._store:
            if item in self._expire_time and self._expire_time[item] < datetime.now():
                del self._store[item]
                raise KeyError(item)
            return PathResult(item, value=self._store[item])
        elif item in self._children:
            return PathResult(item, dir=True)
        else:
            raise KeyError(item)

    def _read_dir(self, item, recursive=False, sort=False):
        item = _normalize_path(item)
        r = PathResult(item, dir=True)
        if recursive:
            for ch in self._children[item]:
                v = self.read(item + '/' + ch, recursive=True)
                if v.dir and v.children:
                    r.children.extend(v.children)
                else:
                    r.children.append(v)
        else:
            r.children = [self._read_value(item + '/' + ch) for ch in self._children[item]]
        if sort:
            r.children = sorted(r.children, key=lambda ch: ch.key)
        return r

    def read(self, item, recursive=False, sort=False):
        item = _normalize_path(item)
        if item in self._store:
            if item in self._expire_time and self._expire_time[item] < datetime.now():
                del self._store[item]
                raise KeyError(item)
            return PathResult(item, value=self._store[item])
        else:
            return self._read_dir(item, recursive=recursive, sort=sort)

    def watch(self, key, timeout=None, recursive=None, sort=False):
        key = _normalize_path(key)
        segments = key.lstrip('/').split('/')
        path = ''
        parent = '/'
        watch_event = Event()
        event_paths = []

        for s in segments:
            if parent not in self._watch_event_r:
                self._watch_event_r[parent] = set()
            if parent not in self._watch_event:
                self._watch_event[parent] = set()
            if recursive:
                event_paths.append(parent)
                self._watch_event_r[parent].add(watch_event)

            path += '/' + s
            parent = path

        if path not in self._watch_event_r:
            self._watch_event_r[path] = set()
        if path not in self._watch_event:
            self._watch_event[path] = set()

        if recursive:
            event_paths.append(key)
            self._watch_event_r[key].add(watch_event)
        else:
            self._watch_event[key].add(watch_event)

        if not watch_event.wait(timeout):
            raise TimeoutError

        if recursive:
            for p in event_paths:
                self._watch_event_r[p].remove(watch_event)
        else:
            self._watch_event[key].remove(watch_event)

        return self.read(key, recursive=recursive, sort=sort)

    def eternal_watch(self, key, recursive=False):
        while True:
            response = self.watch(key, timeout=None, recursive=recursive)
            yield response

    def get_lock(self, lock_name):
        from gevent.lock import RLock
        lock_name = _normalize_path(lock_name)
        if lock_name not in self._store:
            lock = RLock()
            self.write(lock_name, lock)
        else:
            lock = self._store[lock_name]
        return lock

    def write(self, key, value=None, ttl=None, dir=False):
        key = _normalize_path(key)
        segments = key.lstrip('/').split('/')
        path = ''
        parent = '/'
        event_paths = []

        for s in segments:
            if parent not in self._children:
                self._children[parent] = set()
            self._children[parent].add(s)
            event_paths.append(parent)

            path += '/' + s
            if path != key and path in self._store:
                raise KeyError('Not a directory: %s' % key)
            parent = path
        if dir:
            self._children[key] = set()
        else:
            self._store[key] = value
            if ttl:
                self._expire_time[key] = datetime.now() + timedelta(seconds=ttl)

        event_paths.append(key)
        for p in event_paths:
            if p in self._watch_event_r:
                [e.set() for e in list(self._watch_event_r[p])]
        if key in self._watch_event:
            [e.set() for e in list(self._watch_event[key])]

    def delete(self, key, dir=False, recursive=False):
        key = _normalize_path(key)
        if not dir:
            del self._store[key]
        else:
            if not recursive and self._children[key]:
                raise KeyError('Dir %s not empty', key)
            for ch in list(self._children[key]):
                ch_key = key + '/' + ch
                if ch_key in self._store:
                    self.delete(ch_key)
                else:
                    self.delete(ch_key, dir=True, recursive=True)
            del self._children[key]
        dir_name, file_name = key.rsplit('/', 1)
        if not dir_name:
            dir_name = '/'
        self._children[dir_name].remove(file_name)

        if key in self._watch_event:
            [e.set() for e in list(self._watch_event[key])]
            del self._watch_event[key]

        if key in self._watch_event_r:
            base_path = ''
            event_paths = ['/']
            for sp in key.lstrip('/').split('/'):
                base_path += '/' + sp
                event_paths.append(base_path)

            for evt in list(self._watch_event_r[key]):
                evt.set()
                for p in event_paths:
                    if evt in self._watch_event_r.get(p, ()):
                        self._watch_event_r[p].remove(evt)
            del self._watch_event_r[key]


class EtcdKVStore(object):
    def __init__(self, etcd_client, base_path=''):
        self._etcd_client = etcd_client
        self._base_path = _normalize_path(base_path)

    def read(self, item, recursive=False, sort=False):
        from etcd_gevent import EtcdKeyError, EtcdKeyNotFound
        item = _normalize_path(self._base_path + item)

        try:
            val = self._etcd_client.read(item, recursive=recursive)
        except (EtcdKeyError, EtcdKeyNotFound):
            val = None
        if val is None:
            raise KeyError(item)

        r = PathResult(item, value=val.value, dir=val.dir)
        for ch in val.children:
            if ch.key == item:
                continue
            r.children.append(PathResult(ch.key, value=ch.value, dir=ch.dir))
        if sort:
            r.children = sorted(r.children, key=lambda ch: ch.key)
        return r

    def watch(self, item, timeout=None, recursive=None):
        from etcd_gevent import EtcdKeyError, EtcdWatchTimedOut
        item = _normalize_path(self._base_path + item)
        try:
            val = self._etcd_client.watch(item, timeout=timeout, recursive=recursive)
        except EtcdKeyError:
            raise KeyError(item)
        except EtcdWatchTimedOut:
            raise TimeoutError

        r = PathResult(item, value=val.value, dir=val.dir)
        for ch in val.children:
            r.children.append(PathResult(ch.key, value=ch.value, dir=ch.dir))
        return r

    def eternal_watch(self, key, recursive=False):
        while True:
            response = self.watch(key, timeout=0, recursive=recursive)
            yield response

    def get_lock(self, lock_name):
        from etcd_gevent import Lock
        lock_name = _normalize_path(lock_name)
        return Lock(self._etcd_client, lock_name)

    def write(self, key, value=None, ttl=None, dir=False):
        from etcd_gevent import EtcdKeyError, EtcdNotFile
        key = _normalize_path(self._base_path + key)
        try:
            self._etcd_client.write(key, value, ttl=ttl, dir=dir)
        except EtcdKeyError as ex:
            if dir and isinstance(ex, EtcdNotFile):
                return
            raise KeyError('%s not dir' % key)

    def delete(self, key, dir=False, recursive=False):
        from etcd_gevent import EtcdKeyError, EtcdDirNotEmpty
        key = _normalize_path(self._base_path + key)
        try:
            self._etcd_client.delete(key, dir=dir, recursive=recursive)
        except EtcdKeyError:
            raise KeyError(key)
        except EtcdDirNotEmpty:
            raise KeyError('Dir %s not empty', key)


def get(addr):
    if addr == ':inproc:':
        return LocalKVStore()
    parsed = urlparse(addr)
    if parsed.scheme == 'etcd':
        from etcd_gevent.client import Client as EtcdClient
        hosts = tuple((h.strip(), parsed.port) for h in parsed.hostname.split(','))
        client = EtcdClient(host=hosts, allow_reconnect=True)
        return EtcdKVStore(client, parsed.path)
    else:  # pragma: no cover
        raise ValueError('Scheme %s not supported.' % parsed.scheme)
