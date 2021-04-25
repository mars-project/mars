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

from typing import Any, Dict

from .....core import OperandType
from .....tests.core import _check_args, ObjectCheckMixin
from ..subtask import SubtaskProcessor


class CheckedSubtaskProcessor(ObjectCheckMixin, SubtaskProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        check_options = dict()
        kwargs = self.subtask.extra_config or dict()
        for key in _check_args:
            check_options[key] = kwargs.get(key, True)
        self._check_options = check_options

    def _execute_operand(self,
                         ctx: Dict[str, Any],
                         op: OperandType):
        super()._execute_operand(ctx, op)
        if self._check_options.get('check_all', True):
            for out in op.outputs:
                if out not in self._chunk_graph.result_chunks:
                    continue
                self.assert_object_consistent(out, ctx[out.key])
