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
import sys
from typing import Any, BinaryIO, Dict, List, TextIO, Union

import numpy as np

from .. import opcodes
from ..core import OutputType, TILEABLE_TYPE
from ..core.context import Context
from ..core.operand import MergeDictOperand
from ..serialization.serializables import BytesField, ListField, \
    Int32Field, DictField, BoolField
from ..typing import TileableType, SessionType
from ..utils import to_binary, build_fetch_tileable


class RunScript(MergeDictOperand):
    _op_type_ = opcodes.RUN_SCRIPT

    _code: bytes = BytesField('code')
    _data: Dict[str, TileableType] = DictField('data')
    _retry_when_fail: bool = BoolField('retry_when_fail')
    _command_args: List[str] = ListField('command_args')
    _world_size: int = Int32Field('world_size')
    _rank: int = Int32Field('rank')

    def __init__(self, code=None, data=None, world_size=None, rank=None,
                 retry_when_fail=None, command_args=None, **kw):
        super().__init__(_code=code, _data=data, _world_size=world_size, _rank=rank,
                         _retry_when_fail=retry_when_fail, _command_args=command_args,
                         **kw)
        if self.output_types is None:
            self.output_types = [OutputType.object]

    @property
    def code(self):
        return self._code

    @property
    def data(self):
        return self._data

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

    def __call__(self, inputs):
        return self.new_tileable(inputs)

    @classmethod
    def _get_chunk_data(cls, op: "RunScript"):
        new_data = None
        input_chunks = []
        inputs_iter = iter(op.inputs)
        if op.data:
            new_data = dict()
            for k, v in op.data.items():
                if isinstance(v, TILEABLE_TYPE):
                    v = next(inputs_iter)
                    new_data[k] = build_fetch_tileable(v)
                    input_chunks.extend(v.chunks)
                else:
                    new_data[k] = v
        return new_data, input_chunks

    @classmethod
    def tile(cls, op: "RunScript"):
        if len(op.inputs) > 0:
            # trigger inputs to execute
            yield

        new_data, input_chunks = cls._get_chunk_data(op)

        out_chunks = []
        for i in range(op.world_size):
            chunk_op = op.copy().reset_key()
            chunk_op._data = new_data
            chunk_op._rank = i
            out_chunks.append(chunk_op.new_chunk(input_chunks, index=(i,)))

        new_op = op.copy()
        return new_op.new_tileables(op.inputs, chunks=out_chunks,
                                    nsplits=(tuple(np.nan for _ in range(len(out_chunks))),))

    @classmethod
    def _build_envs(cls, ctx, op):
        # set mars envs
        envs = dict()
        envs['RANK'] = str(op.rank)
        envs['WORLD_SIZE'] = str(op.world_size)
        return envs

    @classmethod
    def _build_locals(cls, ctx: Union[Context, dict], op: "RunScript"):
        sess = ctx.get_current_session().as_default()
        local = {'session': sess,
                 '__name__': '__main__'}
        if op.data is not None:
            local.update(op.data)
        return local

    @classmethod
    def execute(cls, ctx, op):
        if op.merge:
            return super().execute(ctx, op)

        old_env = os.environ.copy()
        envs = cls._build_envs(ctx, op)
        old_argv = sys.argv.copy()

        try:
            os.environ.update(envs)
            sys.argv = ['script']
            sys.argv.extend(op.command_args)

            exec(op.code, cls._build_locals(ctx, op))

            if op.rank == 0:
                ctx[op.outputs[0].key] = {'status': 'ok'}
            else:
                ctx[op.outputs[0].key] = {}
        finally:
            os.environ = old_env
            sys.argv = old_argv
            sys.stdout.flush()


def _extract_inputs(data: Dict[str, TileableType] = None) -> List[TileableType]:
    if data is not None and not isinstance(data, dict):
        raise TypeError('`data` must be a dict whose key is '
                        'variable name and value is data')

    inputs = []
    if data is not None:
        for v in data.values():
            if isinstance(v, TILEABLE_TYPE):
                inputs.append(v)

    return inputs


def run_script(script: Union[bytes, str, BinaryIO, TextIO],
               data: Dict[str, TileableType] = None,
               n_workers: int = 1,
               command_argv: List[str] = None,
               session: SessionType = None,
               retry_when_fail: bool = False,
               run_kwargs: Dict[str, Any] = None):
    """
    Run script in Mars cluster.

    Parameters
    ----------
    script: str or file-like object
        Script to run.
    data: dict
        Variable name to data.
    n_workers: int
        number of workers to run the script
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

    inputs = _extract_inputs(data)
    op = RunScript(data=data, code=to_binary(code), world_size=n_workers,
                   retry_when_fail=retry_when_fail, command_args=command_argv)
    return op(inputs).execute(session=session, **(run_kwargs or {}))
