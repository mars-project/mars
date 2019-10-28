# Copyright 1999-2018 Alibaba Group Holding Ltd.
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
from ....tensor.operands import TensorOperand, TensorOperandMixin
from ....serialize import Int32Field


class StartTracker(TensorOperand, TensorOperandMixin):
    _op_module_ = 'learn'
    _op_type_ = OperandDef.START_TRACKER

    _n_workers = Int32Field('n_workers')

    def __init__(self, n_workers=None, **kw):
        super(StartTracker, self).__init__(_n_workers=n_workers, **kw)

    @property
    def n_workers(self):
        return self._n_workers

    @classmethod
    def execute(cls, ctx, op):
        """Start Rabit tracker"""
        from .tracker import RabitTracker

        env = {'DMLC_NUM_WORKER': op.n_workers}
        rabit_context = RabitTracker(hostIP=ctx.get_local_address(),
                                     nslave=op.n_workers)
        env.update(rabit_context.slave_envs())

        rabit_context.start(op.n_workers)
        thread = Thread(target=rabit_context.join)
        thread.daemon = True
        thread.start()

        ctx[op.outputs[0].key] = env
