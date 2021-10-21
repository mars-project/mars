#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import pickle
import threading

import pytest

from ..config import options, option_context, is_integer, is_string, Config


def test_config_context():
    with pytest.raises(AttributeError):
        _ = options.a.b.c

    options.register_option("c.d.e", "a", is_string)
    assert "c" in dir(options)
    assert "d" in dir(options.c)

    try:
        with option_context() as ctx:
            ctx.register_option("a.b.c", 1, validator=is_integer)
            assert ctx.a.b.c == 1

            ctx.a.b.c = 2
            assert ctx.a.b.c == 2

            with pytest.raises(ValueError):
                ctx.a.b.c = "a"

            assert ctx.c.d.e == "a"

            ctx.c.d.e = "b"

        assert options.c.d.e == "a"

        options.c.d.e = "c"

        assert options.c.d.e == "c"

        with pytest.raises(AttributeError):
            _ = options.a.b.c  # noqa: F841
    finally:
        options.unregister_option("c.d.e")


def test_multi_thread_config():
    options.register_option("a.b.c", 1)

    class T(threading.Thread):
        def __init__(self, is_first, condition):
            super().__init__()
            self.is_first = is_first
            self.condition = condition

        def run(self):
            self.condition.acquire()
            if self.is_first:
                options.a.b.c = 2
                self.condition.notify()
            else:
                self.condition.wait()
                assert options.a.b.c == 1
            self.condition.release()

    try:
        cond = threading.Condition()
        a = T(True, cond)
        b = T(False, cond)
        b.start()
        a.start()
        a.join()
        b.join()
    finally:
        options.unregister_option("a.b.c")


def test_config_copy():
    cfg = Config()
    cfg.register_option("a.b.c", 1)
    cfg.redirect_option("a.c", "a.b.c")

    target_cfg = Config()
    target_cfg.register_option("a.b.c", -1)
    target_cfg.redirect_option("a.c", "a.b.c")

    src_cfg_dict = cfg.to_dict()
    assert src_cfg_dict == {"a.b.c": 1}

    target_cfg.update(src_cfg_dict)
    assert target_cfg.a.b.c == 1


def test_pickle_config():
    cfg = Config()
    cfg.register_option("a.b.c", 1)
    cfg.redirect_option("a.c", "a.b.c")

    s = pickle.dumps(cfg)
    new_cfg = pickle.loads(s)
    assert new_cfg.a.b.c == 1
    assert new_cfg.a.c == 1
