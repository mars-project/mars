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

from threading import Thread

from .... import opcodes as OperandDef
from ....core import NotSupportTile
from ....serialization.serializables import Int32Field
from ....utils import to_binary
from ...operands import LearnOperand, LearnOperandMixin, OutputType


class StartTracker(LearnOperand, LearnOperandMixin):
    _op_type_ = OperandDef.START_TRACKER
    _op_module_ = 'learn.contrib.xgboost'

    _n_workers = Int32Field('n_workers')

    def __init__(self, n_workers=None, output_types=None, pure_depends=None, **kw):
        super().__init__(_n_workers=n_workers, _output_types=output_types,
                         _pure_depends=pure_depends, **kw)
        if self.output_types is None:
            self.output_types = [OutputType.object]

    @property
    def n_workers(self):
        return self._n_workers

    @classmethod
    def tile(cls, op):
        raise NotSupportTile('StartTracker is a chunk op')

    @classmethod
    def execute(cls, ctx, op):
        """Start Rabit tracker"""
        from .tracker import RabitTracker

        env = {'DMLC_NUM_WORKER': op.n_workers}
        rabit_context = RabitTracker(hostIP=ctx.current_address.split(':', 1)[0],
                                     nslave=op.n_workers)
        env.update(rabit_context.slave_envs())

        rabit_context.start(op.n_workers)
        thread = Thread(target=rabit_context.join)
        thread.daemon = True
        thread.start()

        rabit_args = [to_binary(f'{k}={v}') for k, v in env.items()]
        ctx[op.outputs[0].key] = rabit_args
