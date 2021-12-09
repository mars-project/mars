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
import asyncio

import pytest
from ..profiling import ProfilingData, ProfilingDataOperator, DummyOperator
from ...tests.core import check_dict_structure_same, mock


def test_profiling_data():
    ProfilingData.init("abc")
    try:
        for n in ["general", "serialization"]:
            assert isinstance(ProfilingData["abc", n], ProfilingDataOperator)
        assert ProfilingData["def"] is DummyOperator
        assert ProfilingData["abc", "def"] is DummyOperator
        assert ProfilingData["abc", "def", 1] is DummyOperator
        ProfilingData["def"].set("a", 1)
        ProfilingData["def"].inc("b", 1)
        assert ProfilingData["def"].empty()
        assert sum(ProfilingData["def"].nest("a").values()) == 0
        ProfilingData["abc", "serialization"].set("a", 1)
        ProfilingData["abc", "serialization"].inc("b", 1)
        with pytest.raises(TypeError):
            assert ProfilingData["abc", "serialization"].nest("a")
        assert sum(ProfilingData["abc", "serialization"].nest("c").values()) == 0
        assert not ProfilingData["abc", "serialization"].empty()
    finally:
        v = ProfilingData.pop("abc")
        check_dict_structure_same(
            v,
            {
                "general": {},
                "serialization": {"a": 1, "b": 1, "c": {}},
                "most_calls": {},
                "slowest_calls": {},
                "band_subtasks": {},
                "slowest_subtasks": {},
            },
        )


@pytest.mark.asyncio
@mock.patch("mars.oscar.profiling.logger.warning")
async def test_profiling_debug(fake_warning):
    ProfilingData.init("abc", 0.1)
    assert len(ProfilingData._debug_task) == 1
    assert not ProfilingData._debug_task["abc"].done()
    await asyncio.sleep(0.5)
    assert fake_warning.call_count > 1
    ProfilingData.pop("abc")
    call_count = fake_warning.call_count
    assert len(ProfilingData._debug_task) == 0
    await asyncio.sleep(0.5)
    assert fake_warning.call_count == call_count
