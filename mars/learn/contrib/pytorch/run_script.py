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

import numpy as np

from .... import opcodes as OperandDef
from ....serialize import Int32Field, StringField
from ....context import get_context, RunningMode
from ....utils import to_binary
from ....remote.run_script import RunScript
from ..utils import pick_workers


class RunPyTorch(RunScript):
    _op_type_ = OperandDef.RUN_PYTORCH

    # used for chunk op
    _master_port = Int32Field('master_port')
    _master_addr = StringField('master_addr')
    _rank = Int32Field('rank')
    _init_method = StringField('init_method')

    def __init__(self, master_port=None, master_addr=None, init_method=None,
                 gpu=None, **kw):
        super().__init__(mode='spawn', _master_port=master_port, _master_addr=master_addr,
                         _init_method=init_method, _gpu=gpu, **kw)

    @property
    def master_port(self):
        return self._master_port

    @property
    def master_addr(self):
        return self._master_addr

    @property
    def init_method(self):
        return self._init_method

    def __call__(self):
        return self.new_tileable(None)

    @classmethod
    def tile(cls, op):
        ctx = get_context()

        if ctx.running_mode != RunningMode.distributed:
            workers = ['127.0.0.1'] * op.world_size
        else:
            workers = pick_workers(ctx.get_worker_addresses(), op.world_size)

        out_chunks = []
        for i in range(op.world_size):
            chunk_op = op.copy().reset_key()
            if ctx.running_mode == RunningMode.distributed:
                chunk_op._expect_worker = workers[i]
            if op.init_method is None:
                chunk_op._master_port = op.master_port
                chunk_op._master_addr = workers[0].split(':', 1)[0]
            chunk_op._rank = i
            chunk_op._init_method = op.init_method
            out_chunks.append(chunk_op.new_chunk(None, index=(i,)))

        new_op = op.copy()
        return new_op.new_tileables(op.inputs, chunks=out_chunks,
                                    nsplits=(tuple(np.nan for _ in range(len(out_chunks))),))

    @classmethod
    def _build_envs(cls, ctx, op):
        envs = super()._build_envs(ctx, op)
        if op.master_port is not None:
            envs['MASTER_PORT'] = str(op.master_port)
        if op.master_addr is not None:
            envs['MASTER_ADDR'] = str(op.master_addr)
        return envs

    @classmethod
    def execute(cls, ctx, op):
        assert ctx.get_local_address() == op.expect_worker

        super().execute(ctx, op)


def run_pytorch_script(script, n_workers, gpu=None, command_argv=None,
                       retry_when_fail=False, session=None, run_kwargs=None, port=None):
    """
    Run PyTorch script in Mars cluster.

    :param script: script to run
    :type script: str or file-like object
    :param n_workers: number of PyTorch workers
    :param gpu: run PyTorch script on GPU
    :param command_argv: extra command args for script
    :param retry_when_fail: bool, default False. If True, retry when function failed.
    :param session: Mars session, if not provided, will use default one
    :param run_kwargs: extra kwargs for session.run
    :param port: port of PyTorch worker or ps, will automatically increase for the same worker
    :return: return {'status': 'ok'} if succeeded, or error raised
    """
    if int(n_workers) <= 0:
        raise ValueError('n_workers should be at least 1')
    if hasattr(script, 'read'):
        code = script.read()
    else:
        with open(os.path.abspath(script), 'rb') as f:
            code = f.read()

    port = 29500 if port is None else port
    op = RunPyTorch(code=to_binary(code), world_size=int(n_workers), retry_when_fail=retry_when_fail,
                    gpu=gpu, master_port=port, command_args=command_argv)
    return op().execute(session=session, **(run_kwargs or {})).fetch(session=session)
