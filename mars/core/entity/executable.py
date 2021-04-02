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

from concurrent.futures import ThreadPoolExecutor
from weakref import WeakKeyDictionary, ref

from ...utils import enter_mode


class _TileableSession:
    def __init__(self, tensor, session):
        key = tensor.key, tensor.id

        def cb(_, sess=ref(session)):
            s = sess()
            if s:
                s.decref(key)
        self._tensor = ref(tensor, cb)


class _TileableDataCleaner:
    def __init__(self):
        self._tileable_to_sessions = WeakKeyDictionary()

    @enter_mode(build=True)
    def register(self, tensor, session):
        if tensor in self._tileable_to_sessions:
            self._tileable_to_sessions[tensor].append(
                _TileableSession(tensor, session))
        else:
            self._tileable_to_sessions[tensor] = \
                [_TileableSession(tensor, session)]


# we don't use __del__ to avoid potential Circular reference
_cleaner = _TileableDataCleaner()


class _ExecutableMixin:
    __slots__ = ()

    def execute(self, session=None, **kw):
        from ...session import Session

        if 'fetch' in kw and kw['fetch']:
            raise ValueError('Does not support fetch=True for `.execute()`,'
                             'please use `.fetch()` instead')
        else:
            kw['fetch'] = False

        wait = kw.pop('wait', True)

        if session is None:
            session = Session.default_or_local()

        def run():
            # no more fetch, thus just fire run
            session.run(self, **kw)
            # return Tileable or ExecutableTuple itself
            return self

        if wait:
            return run()
        else:
            # leverage ThreadPoolExecutor to submit task,
            # return a concurrent.future.Future
            thread_executor = ThreadPoolExecutor(1)
            return thread_executor.submit(run)

    def _get_session(self, session=None):
        from ...session import Session

        if session is None and len(self._executed_sessions) > 0:
            session = self._executed_sessions[-1]
        if session is None:
            session = Session.default

        return session

    def _check_session(self, session, action):
        if session is None:
            if isinstance(self, tuple):
                key = self[0].key
            else:
                key = self.key
            raise ValueError(
                f'Tileable object {key} must be executed first before {action}')

    def _fetch(self, session=None, **kw):
        session = self._get_session(session)
        self._check_session(session, 'fetch')
        return session.fetch(self, **kw)

    def fetch(self, session=None, **kw):
        return self._fetch(session=session, **kw)

    def fetch_log(self, session=None, offsets=None, sizes=None):
        session = self._get_session(session)
        self._check_session(session, 'fetch_log')
        return session.fetch_log([self], offsets=offsets, sizes=sizes)[0]

    def _attach_session(self, session):
        _cleaner.register(self, session)
        self._executed_sessions.append(session)


class _ExecuteAndFetchMixin:
    __slots__ = ()

    def _execute_and_fetch(self, session=None, **kw):
        if session is None and len(self._executed_sessions) > 0:
            session = self._executed_sessions[-1]

        wait = kw.pop('wait', True)

        def run():
            fetch_kwargs = kw.pop('fetch_kwargs', dict())
            if len(self._executed_sessions) == 0:
                # not executed before
                self.execute(session=session, **kw)
            return self.fetch(session=session, **fetch_kwargs)

        if wait:
            return run()
        else:
            thread_executor = ThreadPoolExecutor(1)
            return thread_executor.submit(run)


class _ToObjectMixin(_ExecuteAndFetchMixin):
    __slots__ = ()

    def to_object(self, session=None, **kw):
        return self._execute_and_fetch(session=session, **kw)


class ExecutableTuple(tuple, _ExecutableMixin, _ToObjectMixin):
    def __init__(self, *args):
        super().__init__()

        self._fields_to_idx = None
        self._fields = None
        self._raw_type = None

        if len(args) == 1 and isinstance(args[0], tuple):
            self._fields = getattr(args[0], '_fields', None)
            if self._fields is not None:
                self._raw_type = type(args[0])
                self._fields_to_idx = {f: idx for idx, f in enumerate(self._fields)}

        self._executed_sessions = []

    def __getattr__(self, item):
        if self._fields_to_idx is None or item not in self._fields_to_idx:
            raise AttributeError(item)
        return self[self._fields_to_idx[item]]

    def __dir__(self):
        result = list(super().__dir__()) + list(self._fields or [])
        return sorted(result)

    def __repr__(self):
        if not self._fields:
            return super().__repr__()
        items = []
        for k, v in zip(self._fields, self):
            items.append(f'{k}={v!r}')
        return '%s(%s)' % (self._raw_type.__name__, ', '.join(items))

    def execute(self, session=None, **kw):
        if len(self) == 0:
            return self
        return super().execute(session=session, **kw)

    def fetch(self, session=None, **kw):
        if len(self) == 0:
            return tuple()
        ret = super().fetch(session=session, **kw)
        if self._raw_type is not None:
            ret = self._raw_type(*ret)
        return ret

    def fetch_log(self, session=None, offsets=None, sizes=None):
        if len(self) == 0:
            return []
        session = self._get_session(session=session)
        return session.fetch_log(self, offsets=offsets, sizes=sizes)

    def _get_session(self, session=None):
        session = super()._get_session(session=session)
        if session is None:
            for item in self:
                session = item._get_session()
                if session is not None:
                    return session
        return session
