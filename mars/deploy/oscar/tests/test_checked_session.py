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
import pytest

from .... import tensor as mt
from ....config import option_context
from ....core import TileableType, OperandType
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
