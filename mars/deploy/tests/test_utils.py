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

import os

import pytest

from ...services import NodeRole
from ..utils import (
    load_service_config_file,
    get_third_party_modules_from_config,
    next_in_thread,
)

_cwd = os.path.abspath(os.getcwd())


@pytest.mark.parametrize("cwd", [_cwd, os.path.dirname(_cwd)])
def test_load_service_config(cwd):
    old_cwd = os.getcwd()
    try:
        os.chdir(cwd)
        cfg = load_service_config_file(
            os.path.join(os.path.dirname(__file__), "inherit_test_cfg2.yml")
        )

        assert "services" in cfg
        assert cfg["test_list"] == ["item1", "item2", "item3"]
        assert cfg["test_list2"] == ["item3"]
        assert set(cfg["test_dict"].keys()) == {"key1", "key2", "key3"}
        assert set(cfg["test_dict"]["key2"].values()) == {"val2_modified"}
        assert all(not k.startswith("@") for k in cfg.keys())
    finally:
        os.chdir(old_cwd)


def test_get_third_party_modules_from_config():
    r = get_third_party_modules_from_config({}, NodeRole.SUPERVISOR)
    assert r == []

    r = get_third_party_modules_from_config({}, NodeRole.WORKER)
    assert r == []

    config = {"third_party_modules": {"supervisor": ["a.module"]}}
    r = get_third_party_modules_from_config(config, NodeRole.SUPERVISOR)
    assert r == ["a.module"]
    r = get_third_party_modules_from_config(config, NodeRole.WORKER)
    assert r == []

    config = {"third_party_modules": {"worker": ["b.module"]}}
    r = get_third_party_modules_from_config(config, NodeRole.WORKER)
    assert r == ["b.module"]
    r = get_third_party_modules_from_config(config, NodeRole.SUPERVISOR)
    assert r == []

    config = {"third_party_modules": ["ab.module"]}
    r = get_third_party_modules_from_config(config, NodeRole.SUPERVISOR)
    assert r == ["ab.module"]
    r = get_third_party_modules_from_config(config, NodeRole.WORKER)
    assert r == ["ab.module"]

    os.environ["MARS_LOAD_MODULES"] = "c.module,d.module"
    try:
        r = get_third_party_modules_from_config(config, NodeRole.SUPERVISOR)
        assert r == ["ab.module", "c.module", "d.module"]
        r = get_third_party_modules_from_config(config, NodeRole.WORKER)
        assert r == ["ab.module", "c.module", "d.module"]
        r = get_third_party_modules_from_config({}, NodeRole.SUPERVISOR)
        assert r == ["c.module", "d.module"]
        r = get_third_party_modules_from_config({}, NodeRole.WORKER)
        assert r == ["c.module", "d.module"]
    finally:
        os.environ.pop("MARS_LOAD_MODULES", None)

    config = {"third_party_modules": "ab.module"}
    with pytest.raises(TypeError, match="str"):
        get_third_party_modules_from_config(config, NodeRole.SUPERVISOR)
    config = {"third_party_modules": {"supervisor": "a.module"}}
    with pytest.raises(TypeError, match="str"):
        get_third_party_modules_from_config(config, NodeRole.SUPERVISOR)


@pytest.mark.asyncio
async def test_next_in_thread():
    def gen_fun():
        yield 1
        yield 2

    gen = gen_fun()

    assert await next_in_thread(gen) == 1
    assert await next_in_thread(gen) == 2
    with pytest.raises(StopAsyncIteration):
        await next_in_thread(gen)
