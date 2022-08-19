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

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from .... import dataframe as md
from .... import tensor as mt
from ....dataframe.base.apply import ApplyOperand
from ....config import option_context
from ....core import TileableGraph, TileableType, OperandType
from ....services.task.supervisor.tests import CheckedTaskPreprocessor
from ....services.subtask.worker.tests import CheckedSubtaskProcessor
from ..local import _load_config
from ..tests.session import new_test_session, CONFIG_FILE


class FakeCheckedTaskPreprocessor(CheckedTaskPreprocessor):
    def _check_nsplits(self, tiled: TileableType):
        raise RuntimeError("Premeditated")


class FakeCheckedSubtaskProcessor(CheckedSubtaskProcessor):
    def _execute_operand(self, ctx: Dict[str, Any], op: OperandType):
        if self._check_options.get("check_all", True):
            raise RuntimeError("Premeditated")
        else:
            return super()._execute_operand(ctx, op)


class FuncKeyCheckedTaskPreprocessor(CheckedTaskPreprocessor):
    def tile(self, tileable_graph: TileableGraph):
        ops = [t.op for t in tileable_graph if isinstance(t.op, ApplyOperand)]
        assert all(hasattr(op, "func_key") for op in ops)
        assert all(op.func_key is None for op in ops)
        assert all(op.func is not None for op in ops)
        assert all(op.need_clean_up_func is False for op in ops)
        result = super().tile(tileable_graph)
        for op in ops:
            assert hasattr(op, "func_key")
            if op.func_key is not None:
                assert op.need_clean_up_func is True
                assert op.func is None
            else:
                assert op.need_clean_up_func is False
                assert op.func is not None
        return result


class FuncKeyCheckedSubtaskProcessor(CheckedSubtaskProcessor):
    def _execute_operand(self, ctx: Dict[str, Any], op: OperandType):
        if isinstance(op, ApplyOperand):
            assert hasattr(op, "func_key")
            if op.func_key is not None:
                assert op.need_clean_up_func is True
                assert op.func is None
            else:
                assert op.need_clean_up_func is False
                assert op.func is not None
            result = super()._execute_operand(ctx, op)
            assert op.func is not None
            return result
        else:
            return super()._execute_operand(ctx, op)


@pytest.fixture(scope="module")
def setup():
    with option_context({"show_progress": False}):
        yield


def test_checked_session(setup):
    sess = new_test_session(default=True)

    a = mt.ones((10, 10))
    b = a + 1
    b.execute()

    np.testing.assert_array_equal(sess.fetch(b), np.ones((10, 10)) + 1)

    sess.stop_server()


def test_check_task_preprocessor(setup):
    config = _load_config(CONFIG_FILE)
    config["task"][
        "task_preprocessor_cls"
    ] = "mars.deploy.oscar.tests.test_checked_session.FakeCheckedTaskPreprocessor"

    sess = new_test_session(default=True, config=config)

    a = mt.ones((10, 10))
    b = a + 1

    with pytest.raises(RuntimeError, match="Premeditated"):
        b.execute()

    # test test config
    b.execute(extra_config={"check_nsplits": False})

    sess.stop_server()


def test_check_subtask_processor(setup):
    config = _load_config(CONFIG_FILE)
    config["subtask"][
        "subtask_processor_cls"
    ] = "mars.deploy.oscar.tests.test_checked_session.FakeCheckedSubtaskProcessor"

    sess = new_test_session(default=True, config=config)

    a = mt.ones((10, 10))
    b = a + 1

    with pytest.raises(RuntimeError, match="Premeditated"):
        b.execute()

    # test test config
    b.execute(extra_config={"check_all": False})

    sess.stop_server()


def test_clean_up_and_restore_func(setup):
    config = _load_config(CONFIG_FILE)
    config["task"][
        "task_preprocessor_cls"
    ] = "mars.deploy.oscar.tests.test_checked_session.FuncKeyCheckedTaskPreprocessor"
    config["subtask"][
        "subtask_processor_cls"
    ] = "mars.deploy.oscar.tests.test_checked_session.FuncKeyCheckedSubtaskProcessor"

    sess = new_test_session(default=True, config=config)

    cols = [chr(ord("A") + i) for i in range(10)]
    df_raw = pd.DataFrame(dict((c, [i**2 for i in range(20)]) for c in cols))
    df = md.DataFrame(df_raw, chunk_size=5)

    x_small = pd.Series([i for i in range(10)])
    y_small = pd.Series([i for i in range(10)])
    x_large = pd.Series([i for i in range(10**4)])
    y_large = pd.Series([i for i in range(10**4)])

    def closure_small(z):
        return pd.concat([x_small, y_small], ignore_index=True)

    def closure_large(z):
        return pd.concat([x_large, y_large], ignore_index=True)

    # no need to clean up func, func_key won't be set
    r_small = df.apply(closure_small, axis=1)
    r_small.execute()
    # need to clean up func, func_key will be set
    r_large = df.apply(closure_large, axis=1)
    r_large.execute()

    sess.stop_server()
