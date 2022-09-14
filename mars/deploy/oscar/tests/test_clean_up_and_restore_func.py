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

import pandas as pd
import pytest

from .... import dataframe as md
from ....dataframe.base.apply import ApplyOperand
from ....config import option_context
from ....core import TileableGraph, OperandType
from ....services.task.supervisor.tests import CheckedTaskPreprocessor
from ....services.subtask.worker.tests import CheckedSubtaskProcessor
from ....services.task.supervisor.preprocessor import TaskPreprocessor
from ....services.subtask.worker.processor import SubtaskProcessor

from ....utils import lazy_import
from ..local import _load_config as _load_mars_config
from ..tests.session import new_test_session, CONFIG_FILE

ray = lazy_import("ray")


class MarsBackendFuncCheckedTaskPreprocessor(CheckedTaskPreprocessor):
    def tile(self, tileable_graph: TileableGraph):
        ops = [t.op for t in tileable_graph if isinstance(t.op, ApplyOperand)]
        for op in ops:
            assert hasattr(op, "func_key")
            assert op.func_key is None
            assert op.func is not None
            assert callable(op.func)
            assert op.need_clean_up_func is False
        result = super().tile(tileable_graph)
        for op in ops:
            assert hasattr(op, "func_key")
            assert op.func_key is None
            if op.need_clean_up_func:
                assert isinstance(op.func, bytes)
            else:
                assert callable(op.func)
        return result


class MarsBackendFuncCheckedSubtaskProcessor(CheckedSubtaskProcessor):
    def _execute_operand(self, ctx: Dict[str, Any], op: OperandType):
        if isinstance(op, ApplyOperand):
            assert hasattr(op, "func_key")
            assert op.func_key is None
            if op.need_clean_up_func:
                assert isinstance(op.func, bytes)
            else:
                assert callable(op.func)
            result = super()._execute_operand(ctx, op)
            assert op.func is not None
            assert callable(op.func)
            return result
        else:
            return super()._execute_operand(ctx, op)


class RayBackendFuncTaskPreprocessor(TaskPreprocessor):
    def tile(self, tileable_graph: TileableGraph):
        ops = [t.op for t in tileable_graph if isinstance(t.op, ApplyOperand)]
        for op in ops:
            assert hasattr(op, "func_key")
            assert op.func_key is None
            assert op.func is not None
            assert callable(op.func)
            assert op.need_clean_up_func is False
        result = super().tile(tileable_graph)
        for op in ops:
            assert hasattr(op, "func_key")
            if op.need_clean_up_func:
                assert op.func is None
                assert isinstance(op.func_key, ray.ObjectRef)
            else:
                assert callable(op.func)
                assert op.func_key is None
        return result


class RayBackendFuncSubtaskProcessor(SubtaskProcessor):
    def _execute_operand(self, ctx: Dict[str, Any], op: OperandType):
        if isinstance(op, ApplyOperand):
            assert hasattr(op, "func_key")
            if op.need_clean_up_func:
                assert op.func is None
                assert isinstance(op.func_key, ray.ObjectRef)
            else:
                assert callable(op.func)
                assert op.func_key is None
            result = super()._execute_operand(ctx, op)
            assert op.func is not None
            assert callable(op.func)
            return result
        else:
            return super()._execute_operand(ctx, op)


@pytest.fixture(scope="module")
def setup():
    with option_context({"show_progress": False}):
        yield


def test_mars_backend_clean_up_and_restore_func(setup):
    config = _load_mars_config(CONFIG_FILE)
    config["task"][
        "task_preprocessor_cls"
    ] = "mars.deploy.oscar.tests.test_clean_up_and_restore_func.MarsBackendFuncCheckedTaskPreprocessor"
    config["subtask"][
        "subtask_processor_cls"
    ] = "mars.deploy.oscar.tests.test_clean_up_and_restore_func.MarsBackendFuncCheckedSubtaskProcessor"

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

    r_small = df.apply(closure_small, axis=1)
    r_small.execute()
    r_large = df.apply(closure_large, axis=1)
    r_large.execute()

    sess.stop_server()


@pytest.mark.parametrize("multiplier", [1, 3, 4])
def test_clean_up_and_restore_callable(setup, multiplier):
    config = _load_mars_config(CONFIG_FILE)
    config["task"][
        "task_preprocessor_cls"
    ] = "mars.deploy.oscar.tests.test_clean_up_and_restore_func.MarsBackendFuncCheckedTaskPreprocessor"
    config["subtask"][
        "subtask_processor_cls"
    ] = "mars.deploy.oscar.tests.test_clean_up_and_restore_func.MarsBackendFuncCheckedSubtaskProcessor"

    sess = new_test_session(default=True, config=config)

    cols = [chr(ord("A") + i) for i in range(10)]
    df_raw = pd.DataFrame(dict((c, [i**2 for i in range(20)]) for c in cols))
    df = md.DataFrame(df_raw, chunk_size=5)

    class callable_df:
        __slots__ = "x", "__dict__"

        def __init__(self, multiplier: int = 1):
            self.x = pd.Series([i for i in range(10**multiplier)])
            self.y = pd.Series([i for i in range(10**multiplier)])

        def __call__(self, pdf):
            return pd.concat([self.x, self.y], ignore_index=True)

    cdf = callable_df(multiplier=multiplier)

    r_callable = df.apply(cdf, axis=1)
    r_callable.execute()

    sess.stop_server()
