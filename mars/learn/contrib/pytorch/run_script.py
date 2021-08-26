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

import os
from typing import Any, BinaryIO, Dict, List, Optional, TextIO, Union

import numpy as np

from .... import opcodes as OperandDef
from ....core.context import get_context
from ....remote.run_script import RunScript, _extract_inputs
from ....serialization.serializables import Int32Field, StringField
from ....typing import SessionType, TileableType
from ....utils import to_binary
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
        super().__init__(_master_port=master_port, _master_addr=master_addr,
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

    @classmethod
    def tile(cls, op):
        ctx = get_context()

        workers = pick_workers(ctx.get_worker_addresses(), op.world_size)
        data, input_chunks = cls._get_chunk_data(op)

        out_chunks = []
        for i in range(op.world_size):
            chunk_op = op.copy().reset_key()
            chunk_op._data = data
            chunk_op.expect_worker = workers[i]
            if op.init_method is None:
                chunk_op._master_port = op.master_port
                chunk_op._master_addr = workers[0].split(':', 1)[0]
            chunk_op._rank = i
            chunk_op._init_method = op.init_method
            out_chunks.append(chunk_op.new_chunk(input_chunks, index=(i,)))

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
        assert ctx.current_address.split(':')[0] == op.expect_worker.split(':')[0]

        super().execute(ctx, op)


def run_pytorch_script(script: Union[bytes, str, BinaryIO, TextIO],
                       n_workers: int,
                       data: Dict[str, TileableType] = None,
                       gpu: Optional[bool] = None,
                       command_argv: List[str] = None,
                       retry_when_fail: bool = False,
                       session: SessionType = None,
                       run_kwargs: Dict[str, Any] = None,
                       port: int = None):
    """
    Run PyTorch script in Mars cluster.

    Parameters
    ----------
    script: str or file-like object
        Script to run
    n_workers : int
        Number of PyTorch workers
    data : dict
        Variable name to data.
    gpu : bool
        Run PyTorch script on GPU
    command_argv : list
        Extra command args for script
    retry_when_fail : bool
        If True, retry when function failed.
    session
        Mars session, if not provided, will use default one.
    run_kwargs : dict
        Extra kwargs for `session.run`.
    port : int
        Port of PyTorch worker or ps, will automatically increase for the same worker

    Returns
    -------
    status
        return {'status': 'ok'} if succeeded, or error raised
    """
    if int(n_workers) <= 0:
        raise ValueError('n_workers should be at least 1')
    if hasattr(script, 'read'):
        code = script.read()
    else:
        with open(os.path.abspath(script), 'rb') as f:
            code = f.read()

    inputs = _extract_inputs(data)
    port = 29500 if port is None else port
    op = RunPyTorch(data=data, code=to_binary(code),
                    world_size=int(n_workers), retry_when_fail=retry_when_fail,
                    gpu=gpu, master_port=port, command_args=command_argv)
    return op(inputs).execute(session=session, **(run_kwargs or {}))
