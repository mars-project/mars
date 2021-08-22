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
import json
from typing import Any, BinaryIO, Dict, List, Optional, TextIO, Union

import numpy as np

from .... import opcodes as OperandDef
from ....core import recursive_tile
from ....core.context import get_context
from ....remote.run_script import RunScript, _extract_inputs
from ....serialization.serializables import BytesField, Int32Field, DictField, StringField
from ....typing import SessionType, TileableType
from ....utils import to_binary
from ...utils import collect_ports
from ..utils import pick_workers


class RunTensorFlow(RunScript):
    _op_type_ = OperandDef.RUN_TENSORFLOW

    _code = BytesField('code')
    _n_workers = Int32Field('n_workers')
    _n_ps = Int32Field('n_ps')
    _tf_config = DictField('tf_config')
    _port = Int32Field('port')
    # used for chunk op
    _tf_task_type = StringField('tf_task_type')
    _tf_task_index = Int32Field('tf_task_index')

    def __init__(self, n_workers=None, n_ps=None, tf_config=None, port=None,
                 tf_task_type=None, tf_task_index=None, gpu=None, **kw):
        super().__init__(_n_workers=n_workers, _n_ps=n_ps,
                         _tf_config=tf_config, _port=port, _tf_task_type=tf_task_type,
                         _tf_task_index=tf_task_index, _gpu=gpu, **kw)

    @property
    def n_workers(self):
        return self._n_workers

    @property
    def n_ps(self):
        return self._n_ps or 0

    @property
    def n_roles(self):
        return self._n_workers + self._n_ps

    @property
    def tf_config(self):
        return self._tf_config

    @property
    def port(self):
        return self._port

    @property
    def tf_task_type(self):
        return self._tf_task_type

    @property
    def tf_task_index(self):
        return self._tf_task_index

    @classmethod
    def tile(cls, op):
        ctx = get_context()

        cluster_conf = {'worker': []}
        if op.n_ps > 0:
            cluster_conf['ps'] = []
        n_workers = op.n_workers

        out_chunks = []
        worker_addresses = ctx.get_worker_addresses()
        picked_workers = pick_workers(worker_addresses, op.n_roles)
        data, input_chunks = cls._get_chunk_data(op)

        ports = yield from recursive_tile(
            collect_ports(worker_addresses))
        yield ports.chunks
        ports = ctx.get_chunks_result([ports.chunks[0].key])[0]

        i = 0
        for worker, port in zip(picked_workers, ports):
            worker_addr = worker.rsplit(':', 1)[0]
            chunk_op = op.copy().reset_key()
            chunk_op._data = data
            addr = f'{worker_addr}:{port}'
            # tell graph actor that the chunk should be executed on the exact worker
            chunk_op.expect_worker = worker
            tp = 'worker' if i < n_workers else 'ps'
            chunk_op._tf_task_type = tp
            idx = i if i < n_workers else i - n_workers
            chunk_op._tf_task_index = idx
            cluster_conf[tp].append(addr)
            chunk_op._tf_config = {'cluster': cluster_conf,
                                   'task': {'type': tp, 'index': idx}}
            out_chunks.append(chunk_op.new_chunk(input_chunks, index=(i,)))
            i += 1

        new_op = op.copy()
        return new_op.new_tileables(op.inputs, chunks=out_chunks,
                                    nsplits=(tuple(np.nan for _ in range(len(out_chunks))),))

    @classmethod
    def _build_envs(cls, ctx, op):
        envs = super()._build_envs(ctx, op)
        envs.update({'TF_CONFIG': json.dumps(op.tf_config)})
        return envs

    @classmethod
    def execute(cls, ctx, op):
        if op.merge:
            return super().execute(ctx, op)

        assert ctx.current_address.split(':')[0] == op.expect_worker.split(':')[0]

        super().execute(ctx, op)

        if op.tf_task_type == 'worker' and op.tf_task_index == 0:
            ctx[op.outputs[0].key] = {'status': 'ok'}
        else:
            ctx[op.outputs[0].key] = {}


def run_tensorflow_script(script: Union[bytes, str, BinaryIO, TextIO],
                          n_workers: int,
                          n_ps: int = 0,
                          data: Dict[str, TileableType] = None,
                          gpu: Optional[bool] = None,
                          command_argv: List[str] = None,
                          retry_when_fail: bool = False,
                          session: SessionType = None,
                          run_kwargs: Dict[str, Any] = None,
                          port: int = None):
    """
    Run TensorFlow script in Mars cluster.

    Parameters
    ----------
    script: str or file-like object
        Script to run
    n_workers : int
        Number of TensorFlow workers.
    n_ps : int
        Number of TensorFlow PS workers.
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
        Port of TensorFlow worker or ps, will automatically increase for the same worker

    Returns
    -------
    status
        return {'status': 'ok'} if succeeded, or error raised
    """
    if int(n_workers) <= 0:
        raise ValueError('n_workers should be at least 1')
    if int(n_ps) < 0:
        raise ValueError('n_ps should be at least 0')
    if hasattr(script, 'read'):
        code = script.read()
    else:
        with open(os.path.abspath(script), 'rb') as f:
            code = f.read()

    inputs = _extract_inputs(data)
    op = RunTensorFlow(data=data, code=to_binary(code),
                       n_workers=int(n_workers), n_ps=int(n_ps),
                       retry_when_fail=retry_when_fail, gpu=gpu,
                       port=port, command_args=command_argv)
    return op(inputs).execute(session=session, **(run_kwargs or {}))
