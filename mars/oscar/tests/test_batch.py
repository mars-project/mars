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

import asyncio
import sys

import pytest

from ..batch import extensible, build_args_binder


def _wrap_async(use_async):
    def wrapper(func):
        async def _wrapped(*args, **kwargs):
            return func(*args, **kwargs)

        _wrapped.__name__ = func.__name__
        return _wrapped if use_async else func

    return wrapper


def test_args_binder():
    anon_binder = build_args_binder(lambda x, y=10: None, remove_self=False)
    assert (20, 10) == anon_binder(20)

    def fun1(a, b=10):
        pass

    binder1 = build_args_binder(fun1, remove_self=False)
    assert (20, 10) == binder1(20)

    async def fun2(*, kw_only=10, **kw):
        pass

    binder2 = build_args_binder(fun2, remove_self=False)
    assert (20, {"ext_arg": 5}) == binder2(kw_only=20, ext_arg=5)

    async def fun3(x, *args, kw_only=10, **kw):
        pass

    binder3 = build_args_binder(fun3, remove_self=False)
    assert 10 == binder3(20, 36, ext_arg=5).kw_only
    assert (20, (36,), 10, {"ext_arg": 5}) == binder3(20, 36, ext_arg=5)


def test_extensible_bind():
    class TestClass:
        def __init__(self):
            self.a_list = []
            self.b_list = []

        @extensible
        def method(self, a, b=10):
            pass

        @method.batch
        def method(self, args_list, kwargs_list):
            for args, kwargs in zip(args_list, kwargs_list):
                a, b = self.method.bind(*args, **kwargs)
                self.a_list.append(a)
                self.b_list.append(b)

    test_inst = TestClass()
    test_inst.method.batch(
        test_inst.method.delay(20),
        test_inst.method.delay(30, 5),
    )
    assert test_inst.a_list == [20, 30]
    assert test_inst.b_list == [10, 5]


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.skipif(
    sys.version_info[:2] < (3, 7), reason="only run with Python 3.7 or greater"
)
async def test_extensible_no_batch(use_async):
    class TestClass:
        def __init__(self):
            self.arg_list = []
            self.kwarg_list = []

        @extensible
        @_wrap_async(use_async)
        def method(self, *args, **kwargs):
            self.arg_list.append(tuple(a - 1 for a in args))
            self.kwarg_list.append({k: v - 1 for k, v in kwargs.items()})
            return len(self.kwarg_list)

    test_inst = TestClass()
    ret = test_inst.method.batch(
        test_inst.method.delay(12, kwarg=34), test_inst.method.delay(10, kwarg=33)
    )
    ret = await ret if use_async else ret
    assert ret == [1, 2]
    assert test_inst.arg_list == [(11,), (9,)]
    assert test_inst.kwarg_list == [{"kwarg": 33}, {"kwarg": 32}]

    if use_async:
        test_inst = TestClass()
        ret = await test_inst.method.batch(
            test_inst.method.delay(12, kwarg=34), test_inst.method.delay(10, kawarg=33)
        )
        assert ret == [1, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_extensible_batch_only(use_async):
    class TestClass:
        def __init__(self):
            self.arg_list = []
            self.kwarg_list = []

        @extensible
        @_wrap_async(use_async)
        def not_implemented_method(self, *args, **kw):
            raise NotImplementedError

        @extensible
        @_wrap_async(use_async)
        def method(self, *args, **kwargs):
            raise NotImplementedError

        @method.batch
        @_wrap_async(use_async)
        def method(self, args_list, kwargs_list):
            self.arg_list.extend(args_list)
            self.kwarg_list.extend(kwargs_list)
            return [len(self.kwarg_list)] * len(args_list)

    if use_async:
        assert asyncio.iscoroutinefunction(TestClass.method)

    test_inst = TestClass()
    ret = test_inst.method.batch()
    ret = await ret if use_async else ret
    assert ret == []

    test_inst = TestClass()
    ret = test_inst.method.batch(test_inst.method.delay(12))
    ret = await ret if use_async else ret
    assert ret == [1]

    test_inst = TestClass()
    ret = test_inst.method.batch(test_inst.method.delay(12), test_inst.method.delay(10))
    ret = await ret if use_async else ret
    assert ret == [2, 2]
    assert test_inst.arg_list == [(12,), (10,)]
    assert test_inst.kwarg_list == [{}, {}]

    test_inst = TestClass()
    for _ in range(2):
        with pytest.raises(NotImplementedError):
            ret = test_inst.not_implemented_method()
            await ret if use_async else ret
    ret = test_inst.method(12, kwarg=34)
    ret = await ret if use_async else ret
    assert ret == 1
    assert test_inst.arg_list == [(12,)]
    assert test_inst.kwarg_list == [{"kwarg": 34}]


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.version_info[:2] < (3, 7), reason="only run with Python 3.7 or greater"
)
@pytest.mark.parametrize("use_async", [False, True])
async def test_extensible_single_with_batch(use_async):
    class TestClass:
        def __init__(self):
            self.arg_list = []
            self.kwarg_list = []

        @extensible
        @_wrap_async(use_async)
        def method(self, *args, **kwargs):
            self.arg_list.append(tuple(a * 2 for a in args))
            self.kwarg_list.append({k: v * 2 for k, v in kwargs.items()})
            return len(self.kwarg_list)

        @method.batch
        @_wrap_async(use_async)
        def method(self, args_list, kwargs_list):
            self.arg_list.extend([tuple(a * 2 + 1 for a in args) for args in args_list])
            self.kwarg_list.extend(
                [{k: v * 2 + 1 for k, v in kwargs.items()} for kwargs in kwargs_list]
            )
            return [len(self.kwarg_list)] * len(args_list)

    if use_async:
        assert asyncio.iscoroutinefunction(TestClass.method)

    test_inst = TestClass()
    ret = test_inst.method(15, kwarg=56)
    ret = await ret if use_async else ret
    assert ret == 1
    ret = test_inst.method.batch(
        test_inst.method.delay(16, kwarg=57), test_inst.method.delay(17, kwarg=58)
    )
    ret = await ret if use_async else ret
    assert ret == [3, 3]
    assert test_inst.arg_list == [(30,), (33,), (35,)]
    assert test_inst.kwarg_list == [{"kwarg": 112}, {"kwarg": 115}, {"kwarg": 117}]
