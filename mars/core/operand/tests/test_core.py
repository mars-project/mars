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

import numpy as np
import pytest

from ... import OutputType
from .. import Operand, TileableOperandMixin, execute, estimate_size


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


class _OperandMixin(TileableOperandMixin):
    @classmethod
    def tile(cls, op):
        out = op.outputs[0]
        params = out.params.copy()
        params["index"] = (0,) * out.ndim
        chunk = op.copy().reset_key().new_chunk(None, kws=[params])
        new_params = out.params.copy()
        new_params["chunks"] = [chunk]
        new_params["nsplits"] = ()
        return op.copy().new_tileables(op.inputs, kws=[new_params])


class MyOperand3(Operand, _OperandMixin):
    @classmethod
    def execute(cls, ctx, op):
        raise ValueError("intend to fail")

    @classmethod
    def post_execute(cls, ctx, op):  # pragma: no cover
        ctx[op.outputs[0].key] += 1


class MyOperand4(Operand, _OperandMixin):
    @classmethod
    def post_execute(cls, ctx, op):
        ctx[op.outputs[0].key] += 1


class MyOperand5(MyOperand4):
    pass


def test_execute():
    op = MyOperand(extra_params={"my_extra_params": 1})
    assert op.extra_params["my_extra_params"] == 1
    MyOperand.register_executor(lambda *_: 2)
    assert execute(dict(), MyOperand(_key="1")) == 2
    assert execute(dict(), MyOperand2(_key="1")) == 2

    MyOperand.unregister_executor()
    assert execute(dict(), MyOperand(_key="1")) == 1
    MyOperand2.unregister_executor()
    with pytest.raises(KeyError):
        execute(dict(), MyOperand2(_key="1"))


def test_estimate_size():
    MyOperand.register_size_estimator(lambda *_: 2)
    assert estimate_size(dict(), MyOperand(_key="1")) == 2
    assert estimate_size(dict(), MyOperand2(_key="1")) == 2

    MyOperand.unregister_size_estimator()
    assert estimate_size(dict(), MyOperand(_key="1")) == 1
    MyOperand2.unregister_size_estimator()
    with pytest.raises(KeyError):
        estimate_size(dict(), MyOperand2(_key="1"))


def test_unknown_dtypes():
    op = MyOperand(_output_types=[OutputType.dataframe])
    df = op.new_tileable(None, dtypes=None)
    op2 = MyOperand(_output_types=[OutputType.scalar])
    with pytest.raises(ValueError) as exc_info:
        op2.new_tileable([df])
    assert "executed first" in exc_info.value.args[0]


def test_post_execute(setup):
    op = MyOperand3(_output_types=[OutputType.tensor])
    t = op.new_tileable(None, dtype=np.dtype(float), shape=())
    with pytest.raises(ValueError, match="intend to fail"):
        t.execute()

    op = MyOperand5(_output_types=[OutputType.tensor])
    t2 = op.new_tileable(None, dtype=np.dtype(float), shape=())

    def execute_error(*_):
        raise ValueError("intend to fail again")

    with pytest.raises(ValueError, match="intend to fail again"):
        operand_executors = {MyOperand4: execute_error}
        t2.execute(extra_config={"operand_executors": operand_executors}).fetch()

    def execute_normally(ctx, op):
        ctx[op.outputs[0].key] = 1

    operand_executors = {MyOperand5: execute_normally}
    assert (
        t2.execute(extra_config={"operand_executors": operand_executors}).fetch() == 2
    )
