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
import dataclasses
import os

import pytest
from ..profiling import (
    ProfilingData,
    ProfilingDataOperator,
    DummyOperator,
    _ProfilingOptions,
    _CallStats,
    _SubtaskStats,
)
from ..backends.message import SendMessage
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
                "slow_calls": {},
                "band_subtasks": {},
                "slow_subtasks": {},
            },
        )


@pytest.mark.asyncio
@mock.patch("mars.oscar.profiling.logger.warning")
async def test_profiling_debug(fake_warning):
    ProfilingData.init("abc", {"debug_interval_seconds": 0.1})
    assert len(ProfilingData._debug_task) == 1
    assert not ProfilingData._debug_task["abc"].done()
    await asyncio.sleep(0.5)
    assert fake_warning.call_count > 1
    ProfilingData.pop("abc")
    call_count = fake_warning.call_count
    assert len(ProfilingData._debug_task) == 0
    await asyncio.sleep(0.5)
    assert fake_warning.call_count == call_count

    ProfilingData.init("abc", {"debug_interval_seconds": 0.1})
    assert len(ProfilingData._debug_task) == 1
    await asyncio.sleep(0.5)
    assert fake_warning.call_count > call_count
    ProfilingData._data.clear()
    call_count = fake_warning.call_count
    await asyncio.sleep(0.5)
    assert len(ProfilingData._debug_task) == 0
    assert fake_warning.call_count == call_count


@pytest.mark.asyncio
async def test_profiling_options():
    with pytest.raises(ValueError):
        ProfilingData.init("abc", 1.2)
    with pytest.raises(ValueError):
        ProfilingData.init("abc", ["invalid"])
    with pytest.raises(ValueError):
        ProfilingData.init("abc", {"invalid": True})
    with pytest.raises(ValueError):
        ProfilingData.init("abc", {"debug_interval_seconds": "abc"})

    # Test the priority, options first, then env var.
    env_key = "MARS_PROFILING_DEBUG_INTERVAL_SECONDS"
    try:
        os.environ[env_key] = "2"
        options = _ProfilingOptions(True)
        assert options.debug_interval_seconds == 2.0
        options = _ProfilingOptions({"debug_interval_seconds": 1.0})
        assert options.debug_interval_seconds == 1.0
    finally:
        os.environ.pop(env_key)

    # Test option value cache.
    d = {"debug_interval_seconds": 1.0}
    options = _ProfilingOptions(d)
    assert options.debug_interval_seconds == 1.0
    d["debug_interval_seconds"] = 2.0
    assert options.debug_interval_seconds == 1.0
    try:
        os.environ[env_key] = "2"
        assert options.debug_interval_seconds == 1.0
    finally:
        os.environ.pop(env_key)


def test_collect():
    options = _ProfilingOptions(
        {"slow_calls_duration_threshold": 0, "slow_subtasks_duration_threshold": 0}
    )

    # Test collect message with incomparable arguments.
    from ..core import ActorRef

    fake_actor_ref = ActorRef("def", b"uid")
    fake_message1 = SendMessage(b"abc", fake_actor_ref, ["name", {}])
    fake_message2 = SendMessage(b"abc", fake_actor_ref, ["name", 1])

    cs = _CallStats(options)
    cs.collect(fake_message1, 1.0)
    cs.collect(fake_message2, 1.0)

    @dataclasses.dataclass
    class _FakeSubtask:
        extra_config: dict

    # Test collect subtask with incomparable arguments.
    band = ("1.2.3.4", "numa-0")
    subtask1 = _FakeSubtask({})
    subtask2 = _FakeSubtask(None)
    ss = _SubtaskStats(options)
    ss.collect(subtask1, band, 1.0)
    ss.collect(subtask2, band, 1.0)

    # Test call stats order.
    cs = _CallStats(options)
    for i in range(20):
        fake_message = SendMessage(
            f"{i}".encode(), fake_actor_ref, ["name", True, (i,), {}]
        )
        cs.collect(fake_message, i)
    d = cs.to_dict()
    assert list(d["most_calls"].values())[0] == 20
    assert list(d["slow_calls"].values()) == list(reversed(range(10, 20)))

    # Test subtask stats order.
    ss = _SubtaskStats(options)
    counter = 0
    for i in range(20):
        for j in range(i):
            fake_message = _FakeSubtask(counter)
            ss.collect(fake_message, (str(j), "numa-0"), counter)
            counter += 1
    d = ss.to_dict()
    assert list(d["band_subtasks"].values()) == [19, 18, 17, 16, 15, 5, 4, 3, 2, 1]
    assert list(d["slow_subtasks"].values()) == list(
        reversed(range(counter - 10, counter))
    )
