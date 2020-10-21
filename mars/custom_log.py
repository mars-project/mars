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

import atexit
import functools
import io
import os
import shutil
import sys
import tempfile
import textwrap
import weakref

from .context import RunningMode, DistributedContext


_custom_log_dir = None


def _get_custom_log_dir():
    from .config import options

    global _custom_log_dir

    if _custom_log_dir is None:
        log_dir = options.custom_log_dir

        if log_dir is None:
            log_dir = tempfile.mkdtemp(prefix='mars-custom-log')

        _custom_log_dir = log_dir

        # remove log dir at exit
        atexit.register(lambda: shutil.rmtree(_custom_log_dir))

    return _custom_log_dir


class _LogWrapper:
    def __init__(self, ctx: DistributedContext, op,
                 log_path: str, custom_log_meta):
        self.ctx = ctx
        self.op = op
        self.log_path = log_path
        self.custom_log_meta = custom_log_meta

        self.file = open(log_path, 'w')
        self.stdout = sys.stdout
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
        worker_addr = self.ctx.get_local_address()
        log_path = self.log_path
        self.custom_log_meta.record_custom_log_path(
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
        self.stdout.write(data)

    def flush(self):
        self.stdout.flush()


def gen_log_path(session_id, op_key):
    filename = f"{str(session_id).replace('-', '_')}_{op_key}"
    custom_log_dir = _get_custom_log_dir()
    return os.path.join(custom_log_dir, filename)


def redirect_custom_log(func):
    """
    Redirect stdout to a file by wrapping ``Operand.execute(ctx, op)``
    """

    @functools.wraps(func)
    def wrap(cls, ctx: DistributedContext, op):
        # import inside, or Ray backend may fail
        from .config import options

        if getattr(ctx, 'running_mode', RunningMode.local) == RunningMode.local or \
                options.custom_log_dir is None:
            # do nothing for local scheduler
            return func(cls, ctx, op)

        custom_log_meta = ctx.get_custom_log_meta_ref()
        log_path = gen_log_path(ctx.session_id, op.key)

        with _LogWrapper(ctx, op, log_path, custom_log_meta):
            return func(cls, ctx, op)

    return wrap


_tileable_to_log_fetcher = weakref.WeakKeyDictionary()


class LogFetcher:
    def __init__(self, tileable_op_key, session):
        self._tileable_op_key = tileable_op_key
        self._session = session
        self._chunk_op_key_to_result = dict()
        self._chunk_op_key_to_offsets = dict()

    def __len__(self):
        return len(self._chunk_op_key_to_result)

    @property
    def chunk_op_keys(self):
        return list(self._chunk_op_key_to_result.keys())

    @property
    def results(self):
        return list(self._chunk_op_key_to_result.values())

    @property
    def offsets(self):
        return list(self._chunk_op_key_to_offsets.values())

    def fetch(self, offsets=None, sizes=None):
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

    def _display(self, representation=True):
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


def fetch(tileables, session, offsets=None, sizes=None):
    log_fetchers = []
    for tileable in tileables:
        tileable = tileable.data if hasattr(tileable, 'data') else tileable

        if tileable not in _tileable_to_log_fetcher:
            _tileable_to_log_fetcher[tileable] = LogFetcher(tileable.op.key, session)

        log_fetcher = _tileable_to_log_fetcher[tileable]
        log_fetcher.fetch(offsets=offsets, sizes=sizes)
        log_fetchers.append(log_fetcher)
    return log_fetchers
