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

import asyncio
import threading
from typing import Dict, Optional


class Isolation:
    loop: asyncio.AbstractEventLoop
    _stopped: Optional[asyncio.Event]
    _thread: Optional[threading.Thread]

    def __init__(self,
                 loop: asyncio.AbstractEventLoop,
                 threaded: bool = True):
        self.loop = loop
        self._threaded = threaded

        self._stopped = None
        self._thread = None
        self._thread_ident = None

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self._stopped = asyncio.Event()
        self.loop.run_until_complete(self._stopped.wait())

    def start(self):
        if self._threaded:
            self._thread = thread = threading.Thread(target=self._run)
            thread.daemon = True
            thread.start()
            self._thread_ident = thread.ident

    @property
    def thread_ident(self):
        return self._thread_ident

    async def _stop(self):
        self._stopped.set()

    def stop(self):
        if self._threaded:
            asyncio.run_coroutine_threadsafe(self._stop(), self.loop).result()
            self._thread.join()


_name_to_isolation: Dict[str, Isolation] = dict()


DEFAULT_ISOLATION = 'oscar'


def new_isolation(name: str = DEFAULT_ISOLATION,
                  loop: asyncio.AbstractEventLoop = None,
                  threaded: bool = True) -> Isolation:
    if name in _name_to_isolation:
        return _name_to_isolation[name]

    if loop is None:
        loop = asyncio.new_event_loop()

    isolation = Isolation(loop, threaded=threaded)
    isolation.start()
    _name_to_isolation[name] = isolation
    return isolation


def get_isolation(name: str = DEFAULT_ISOLATION):
    isolation = _name_to_isolation[name]
    if isolation.loop.is_closed():  # pragma: no cover
        _name_to_isolation.pop(name)
        raise KeyError(name)
    return isolation


def stop_isolation(name: str = DEFAULT_ISOLATION):
    if name in _name_to_isolation:
        return _name_to_isolation.pop(name).stop()
