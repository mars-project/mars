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

import pytest

from mars.core.operand import Operand, TileableOperandMixin, \
    execute, estimate_size


class MyOperand(Operand, TileableOperandMixin):
    @classmethod
    def execute(cls, ctx, op):
        return 1

    @classmethod
    def estimate_size(cls, ctx, op):
        return 1


class MyOperand2(MyOperand):
    @classmethod
    def execute(cls, ctx, op):
        raise NotImplementedError

    @classmethod
    def estimate_size(cls, ctx, op):
        raise NotImplementedError


def test_execute():
    MyOperand.register_executor(lambda *_: 2)
    assert execute(dict(), MyOperand(_key='1')) == 2
    assert execute(dict(), MyOperand2(_key='1')) == 2

    MyOperand.unregister_executor()
    assert execute(dict(), MyOperand(_key='1')) == 1
    MyOperand2.unregister_executor()
    with pytest.raises(KeyError):
        execute(dict(), MyOperand2(_key='1'))


def test_estimate_size():
    MyOperand.register_size_estimator(lambda *_: 2)
    assert estimate_size(dict(), MyOperand(_key='1')) == 2
    assert estimate_size(dict(), MyOperand2(_key='1')) == 2

    MyOperand.unregister_size_estimator()
    assert estimate_size(dict(), MyOperand(_key='1')) == 1
    MyOperand2.unregister_size_estimator()
    with pytest.raises(KeyError):
        estimate_size(dict(), MyOperand2(_key='1'))
