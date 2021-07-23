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

import asyncio
import concurrent.futures
import threading
from typing import List
from weakref import WeakKeyDictionary, ref

from ...lib.aio import get_isolation
from ...typing import SessionType, TileableType
from ..mode import enter_mode


_decref_pool = concurrent.futures.ThreadPoolExecutor()


class _TileableSession:
    def __init__(self,
                 tileable: TileableType,
                 session: SessionType):
        key = tileable.key

        def cb(_, sess=ref(session)):
            try:
                cur_thread_ident = threading.current_thread().ident
                decref_in_isolation = get_isolation().thread_ident == cur_thread_ident
            except KeyError:
                # isolation destroyed, no need to decref
                return

            def decref():
                from ...deploy.oscar.session import SyncSession
                s = sess()
                if s:
                    try:
                        s = SyncSession.from_isolated_session(s)
                        s.decref(key)
                    except (RuntimeError, ConnectionError, KeyError):
                        pass

            fut = _decref_pool.submit(decref)
            if not decref_in_isolation:
                # if decref in isolation, means that this tileable
                # is not required for main thread, thus we do not need
                # to wait for decref, otherwise, wait a bit
                fut.result(.5)

        self.tileable = ref(tileable, cb)


class _TileableDataCleaner:
    def __init__(self):
        self._tileable_to_sessions = WeakKeyDictionary()

    @enter_mode(build=True)
    def register(self,
                 tileable: TileableType,
                 session: SessionType):
        if tileable in self._tileable_to_sessions:
            self._tileable_to_sessions[tileable].append(
                _TileableSession(tileable, session))
        else:
            self._tileable_to_sessions[tileable] = \
                [_TileableSession(tileable, session)]


# we don't use __del__ to avoid potential Circular reference
_cleaner = _TileableDataCleaner()


def _get_session(
        executable: "_ExecutableMixin",
        session: SessionType = None):
    from ...deploy.oscar.session import get_default_session

    if session is None and len(executable._executed_sessions) > 0:
        session = executable._executed_sessions[-1]
    if session is None:
        session = get_default_session()

    return session


class _ExecutableMixin:
    __slots__ = ()
    _executed_sessions: List[SessionType]

    def execute(self, session: SessionType = None, **kw):
        from ...deploy.oscar.session import execute

        session = _get_session(self, session)
        return execute(self, session=session, **kw)

    def _check_session(self,
                       session: SessionType,
                       action: str):
        if session is None:
            if isinstance(self, tuple):
                key = self[0].key
            else:
                key = self.key
            raise ValueError(
                f'Tileable object {key} must be executed first before {action}')

    def _fetch(self, session: SessionType = None, **kw):
        from ...deploy.oscar.session import fetch

        session = _get_session(self, session)
        self._check_session(session, 'fetch')
        return fetch(self, session=session, **kw)

    def fetch(self, session: SessionType = None, **kw):
        return self._fetch(session=session, **kw)

    def fetch_log(self,
                  session: SessionType = None,
                  offsets: List[int] = None,
                  sizes: List[int] =None):
        from ...deploy.oscar.session import fetch_log

        session = _get_session(self, session)
        self._check_session(session, 'fetch_log')
        return fetch_log(self, session=session,
                         offsets=offsets, sizes=sizes)[0]

    def _attach_session(self, session: SessionType):
        if session not in self._executed_sessions:
            _cleaner.register(self, session)
            self._executed_sessions.append(session)


class _ExecuteAndFetchMixin:
    __slots__ = ()

    def _execute_and_fetch(self,
                           session: SessionType = None, **kw):
        from ...deploy.oscar.session import ExecutionInfo, SyncSession, fetch

        session = _get_session(self, session)
        fetch_kwargs = kw.pop('fetch_kwargs', dict())
        ret = self.execute(session=session, **kw)
        if isinstance(ret, ExecutionInfo):
            # wait=False
            aio_task = ret.aio_task

            async def _wait():
                await aio_task

            def run():
                asyncio.run_coroutine_threadsafe(
                    _wait(), loop=ret.loop).result()
                return fetch(self, session=session, **fetch_kwargs)

            return SyncSession._execution_pool.submit(run)
        else:
            # wait=True
            return self.fetch(session=session, **fetch_kwargs)


class _ToObjectMixin(_ExecuteAndFetchMixin):
    __slots__ = ()

    def to_object(self, session: SessionType = None, **kw):
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

    def execute(self, session: SessionType = None, **kw):
        from ...deploy.oscar.session import execute

        if len(self) == 0:
            return self

        session = _get_session(self, session)
        ret = execute(*self, session=session, **kw)
        if kw.get('wait', True):
            return self
        else:
            return ret

    def _fetch(self, session: SessionType = None, **kw):
        from ...deploy.oscar.session import fetch

        session = _get_session(self, session)
        self._check_session(session, 'fetch')
        return fetch(*self, session=session, **kw)

    def fetch(self, session: SessionType = None, **kw):
        if len(self) == 0:
            return tuple()
        ret = super().fetch(session=session, **kw)
        if self._raw_type is not None:
            ret = self._raw_type(*ret)
        if len(self) == 1:
            return ret,
        return ret

    def fetch_log(self,
                  session: SessionType = None,
                  offsets: List[int] = None,
                  sizes: List[int] = None):
        from ...deploy.oscar.session import fetch_log

        if len(self) == 0:
            return []
        session = self._get_session(session=session)
        return fetch_log(*self, session=session,
                         offsets=offsets, sizes=sizes)

    def _get_session(self, session: SessionType = None):
        if session is None:
            for item in self:
                session = _get_session(item, session)
                if session is not None:
                    return session
        return session
