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

import os
import sys
import tempfile
import subprocess

import numpy as np


from .. import opcodes
from ..operands import MergeDictOperand
from ..context import RunningMode
from ..serialize import BytesField, ListField, Int32Field, StringField, BoolField
from ..utils import to_binary


class RunScript(MergeDictOperand):
    _op_type_ = opcodes.RUN_SCRIPT

    _code = BytesField('code')
    _mode = StringField('mode')
    _retry_when_fail = BoolField('retry_when_fail')
    _command_args = ListField('command_args')
    _world_size = Int32Field('world_size')
    _rank = Int32Field('rank')

    def __init__(self, code=None, mode=None, world_size=None, rank=None, retry_when_fail=None,
                 command_args=None, **kw):
        super().__init__(_code=code, _mode=mode, _world_size=world_size, _rank=rank,
                         _retry_when_fail=retry_when_fail, _command_args=command_args,
                         **kw)

    @property
    def code(self):
        return self._code

    @property
    def mode(self):
        return self._mode

    @property
    def world_size(self):
        return self._world_size

    @property
    def rank(self):
        return self._rank

    @property
    def command_args(self):
        return self._command_args or []

    @property
    def retryable(self):
        return self._retry_when_fail

    def __call__(self):
        return self.new_tileable(None)

    @classmethod
    def tile(cls, op):
        out_chunks = []
        for i in range(op.world_size):
            chunk_op = op.copy().reset_key()
            chunk_op._rank = i
            out_chunks.append(chunk_op.new_chunk(None, index=(i,)))

        new_op = op.copy()
        return new_op.new_tileables(op.inputs, chunks=out_chunks,
                                    nsplits=(tuple(np.nan for _ in range(len(out_chunks))),))

    @classmethod
    def _execute_with_subprocess(cls, op, envs=None):
        # write source code into a temp file
        fd, filename = tempfile.mkstemp('.py')
        with os.fdopen(fd, 'wb') as f:
            f.write(op.code)

        new_envs = os.environ.copy()
        new_envs.update(envs or dict())
        try:
            # exec code in a new process
            process = subprocess.Popen([sys.executable, filename] + op.command_args,
                                       env=new_envs)
            process.wait()
            if process.returncode != 0:  # pragma: no cover
                raise RuntimeError('Run script failed')

        finally:
            os.remove(filename)

    @classmethod
    def _execute_with_exec(cls, op, local=None):
        local = local or dict()

        try:
            exec(op.code, local)
        finally:
            sys.stdout.flush()

    @classmethod
    def _build_envs(cls, ctx, op):
        # set mars envs
        envs = dict()
        envs['MARS_SESSION_ID'] = str(ctx.session_id)
        envs['RANK'] = str(op.rank)
        envs['WORLD_SIZE'] = str(op.world_size)
        if ctx.running_mode != RunningMode.local:
            envs['MARS_SCHEDULER_ADDRESS'] = str(ctx._scheduler_address)
        return envs

    @classmethod
    def _build_locals(cls, ctx, op):
        sess = ctx.get_current_session().as_default()

        return dict(session=sess)

    @classmethod
    def execute(cls, ctx, op):
        if op.merge:
            return super().execute(ctx, op)

        old_env = os.environ.copy()
        envs = cls._build_envs(ctx, op)

        try:
            if op.mode == 'spawn':
                cls._execute_with_subprocess(op, envs)
            elif op.mode == 'exec':
                os.environ.update(envs)
                cls._execute_with_exec(op, cls._build_locals(ctx, op))
            else:  # pragma: no cover
                raise TypeError('Unsupported mode {}'.format(op.mode))

            if op.rank == 0:
                ctx[op.outputs[0].key] = {'status': 'ok'}
            else:
                ctx[op.outputs[0].key] = {}
        finally:
            os.environ = old_env


def run_script(script, n_workers=1, mode='spawn', command_argv=None,
               session=None, retry_when_fail=False, run_kwargs=None):
    """
    Run script in Mars cluster.

    Parameters
    ----------
    script: str or file-like object
        Script to run.
    n_workers: int
        number of workers to run the script
    mode: str, 'spawn' as default
        'spawn' mode use subprocess to run script,
        'exec' mode use exec to run in current process.
    command_argv: list
        extra command args for script
    session: Mars session
        if not provided, will use default one
    retry_when_fail: bool, default False
       If True, retry when function failed.
    run_kwargs: dict
        extra kwargs for session.run

    Returns
    -------
    Object
        Mars Object.

    """

    if hasattr(script, 'read'):
        code = script.read()
    else:
        with open(os.path.abspath(script), 'rb') as f:
            code = f.read()
    if mode not in ['exec', 'spawn']:  # pragma: no cover
        raise TypeError('Unsupported mode {}'.format(mode))

    op = RunScript(code=to_binary(code), mode=mode, world_size=n_workers,
                   retry_when_fail=retry_when_fail, command_args=command_argv)
    return op().execute(session=session, **(run_kwargs or {})).fetch(session=session)
