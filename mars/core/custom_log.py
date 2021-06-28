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

import functools
import io
import os
import sys
import textwrap
import weakref
from typing import List, Callable, Type

from ..typing import OperandType, TileableType, SessionType
from .context import Context


class _LogWrapper:
    def __init__(self,
                 ctx: Context,
                 op: OperandType,
                 log_path: str):
        self.ctx = ctx
        self.op = op
        self.log_path = log_path

        self.file = open(log_path, 'w')
        self.stdout = sys.stdout

        self.raw_stdout = self.stdout
        while isinstance(self.raw_stdout, _LogWrapper):
            self.raw_stdout = self.raw_stdout.stdout

        # flag about registering log path
        self.is_log_path_registered = False

    def __enter__(self):
        self.file.__enter__()
        # set stdout
        sys.stdout = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.__exit__(exc_type, exc_val, exc_tb)
        # set back stdout
        sys.stdout = self.stdout

    def _register_log_path(self):
        if self.is_log_path_registered:
            return

        # register log path
        session_id = self.ctx.session_id
        tileable_op_key = self.op.tileable_op_key
        chunk_op_key = self.op.key
        worker_addr = self.ctx.current_address
        log_path = self.log_path

        self.ctx.register_custom_log_path(
            session_id, tileable_op_key, chunk_op_key,
            worker_addr, log_path)

        self.is_log_path_registered = True

    def write(self, data):
        self._register_log_path()

        # write into file
        self.file.write(data)
        # force flush to make sure `fetch_log` can get stdout in time
        self.file.flush()
        # write into previous stdout
        self.raw_stdout.write(data)

    def flush(self):
        self.raw_stdout.flush()


def redirect_custom_log(func: Callable[[Type, Context, OperandType], None]):
    """
    Redirect stdout to a file by wrapping ``Operand.execute(ctx, op)``
    """

    @functools.wraps(func)
    def wrap(cls,
             ctx: Context,
             op: OperandType):
        custom_log_dir = ctx.new_custom_log_dir()

        if custom_log_dir is None:
            return func(cls, ctx, op)

        log_path = os.path.join(custom_log_dir, op.key)

        with _LogWrapper(ctx, op, log_path):
            return func(cls, ctx, op)

    return wrap


_tileable_to_log_fetcher = weakref.WeakKeyDictionary()


class LogFetcher:
    def __init__(self,
                 tileable_op_key: str,
                 session: SessionType):
        self._tileable_op_key = tileable_op_key
        self._session = session
        self._chunk_op_key_to_result = dict()
        self._chunk_op_key_to_offsets = dict()

    def __len__(self):
        return len(self._chunk_op_key_to_result)

    @property
    def chunk_op_keys(self) -> List[str]:
        return list(self._chunk_op_key_to_result.keys())

    @property
    def results(self) -> list:
        return list(self._chunk_op_key_to_result.values())

    @property
    def offsets(self) -> List[List[int]]:
        return list(self._chunk_op_key_to_offsets.values())

    def fetch(self,
              offsets: List[int] = None,
              sizes: List[int] = None):
        if offsets is None:
            offsets = self._chunk_op_key_to_offsets

        if sizes is None:
            sizes = 1 * 1024 ** 2  # 1M each time

        result: dict = self._session.fetch_tileable_op_logs(
            self._tileable_op_key, offsets=offsets, sizes=sizes)

        if result is None:
            return

        for chunk_key, chunk_result in result.items():
            self._chunk_op_key_to_result[chunk_key] = chunk_result['log']
            self._chunk_op_key_to_offsets[chunk_key] = chunk_result['offset']

    def _display(self, representation: bool = True):
        if len(self) == 1:
            content = next(iter(self._chunk_op_key_to_result.values()))
            return repr(content) if representation else str(content)

        sio = io.StringIO()
        for chunk_op_key, content in self._chunk_op_key_to_result.items():
            sio.write(textwrap.dedent(
                f"""
                Chunk op key: {chunk_op_key}
                Out:
                {content}"""))
        result = sio.getvalue()
        return repr(result) if representation else str(result)

    def __repr__(self):
        return self._display(True)

    def __str__(self):
        return self._display(False)


def fetch(
        tileables: List[TileableType],
        session: SessionType,
        offsets: List[int] = None,
        sizes: List[int] = None):
    log_fetchers = []
    for tileable in tileables:
        tileable = tileable.data if hasattr(tileable, 'data') else tileable

        if tileable not in _tileable_to_log_fetcher:
            _tileable_to_log_fetcher[tileable] = LogFetcher(tileable.op.key, session)

        log_fetcher = _tileable_to_log_fetcher[tileable]
        log_fetcher.fetch(offsets=offsets, sizes=sizes)
        log_fetchers.append(log_fetcher)
    return log_fetchers
