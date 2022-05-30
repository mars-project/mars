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
import inspect
import textwrap
from collections import namedtuple
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple


def build_args_binder(func, remove_self: bool = True) -> Optional[Callable]:
    try:
        spec = inspect.getfullargspec(func)
    except TypeError:  # pragma: no cover
        return None

    sig_list = list(spec.args)
    args_list = list(spec.args)
    if remove_self:
        args_list = args_list[1:]

    if spec.varargs:
        sig_list.append(f"*{spec.varargs}")
        args_list.append(spec.varargs)
    elif spec.kwonlyargs:
        sig_list.append("*")

    sig_list.extend(spec.kwonlyargs)
    args_list.extend(spec.kwonlyargs)

    if spec.varkw:
        sig_list.append(f"**{spec.varkw}")
        args_list.append(spec.varkw)

    if getattr(func, "__name__", None).isidentifier():
        ret_func_name = f"{func.__name__}_binder"
        ret_type_name = f"_Args_{func.__name__}"
    else:
        ret_func_name = f"anon_{id(func)}_binder"
        ret_type_name = f"_ArgsAnon_{id(func)}"

    func_str = textwrap.dedent(
        f"""
    def {ret_func_name}({', '.join(sig_list)}):
        return {ret_type_name}({', '.join(args_list)})
    """
    )

    glob_vars = globals().copy()
    glob_vars[ret_type_name] = namedtuple(ret_type_name, args_list)
    loc_vars = dict()
    exec(func_str, glob_vars, loc_vars)
    ext_func = loc_vars[ret_func_name]
    ext_func.__defaults__ = spec.defaults
    ext_func.__kwdefaults__ = spec.kwonlydefaults

    return ext_func


@dataclass
class _DelayedArgument:
    args: Tuple
    kwargs: Dict


class _ExtensibleCallable:
    func: Callable
    batch_func: Optional[Callable]
    is_async: bool
    has_single_func: bool

    def __call__(self, *args, **kwargs):
        if self.is_async:
            return self._async_call(*args, **kwargs)
        else:
            return self._sync_call(*args, **kwargs)

    async def _async_call(self, *args, **kwargs):
        try:
            if self.has_single_func:
                return await self.func(*args, **kwargs)
        except NotImplementedError:
            self.has_single_func = False

        if self.batch_func is not None:
            ret = await self.batch_func([args], [kwargs])
            return None if ret is None else ret[0]
        raise NotImplementedError

    def _sync_call(self, *args, **kwargs):
        try:
            if self.has_single_func:
                return self.func(*args, **kwargs)
        except NotImplementedError:
            self.has_single_func = False

        if self.batch_func is not None:
            return self.batch_func([args], [kwargs])[0]
        raise NotImplementedError


class _ExtensibleWrapper(_ExtensibleCallable):
    def __init__(
        self,
        func: Callable,
        batch_func: Optional[Callable] = None,
        bind_func: Optional[Callable] = None,
        is_async: bool = False,
    ):
        self.func = func
        self.batch_func = batch_func
        self.bind_func = bind_func
        self.is_async = is_async
        self.has_single_func = True

    @staticmethod
    def delay(*args, **kwargs):
        return _DelayedArgument(args=args, kwargs=kwargs)

    @staticmethod
    def _gen_args_kwargs_list(delays):
        args_list = [delay.args for delay in delays]
        kwargs_list = [delay.kwargs for delay in delays]
        return args_list, kwargs_list

    async def _async_batch(self, args_list, kwargs_list):
        # when there is only one call in batch, calling one-pass method
        # will be more efficient
        if len(args_list) == 0:
            return []
        elif len(args_list) == 1:
            return [await self._async_call(*args_list[0], **kwargs_list[0])]
        elif self.batch_func:
            return await self.batch_func(args_list, kwargs_list)
        else:
            # this function has no batch implementation
            # call it separately
            tasks = [
                asyncio.create_task(self.func(*args, **kwargs))
                for args, kwargs in zip(args_list, kwargs_list)
            ]
            try:
                return await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                _ = [task.cancel() for task in tasks]
                return await asyncio.gather(*tasks)

    def _sync_batch(self, args_list, kwargs_list):
        if len(args_list) == 0:
            return []
        elif self.batch_func:
            return self.batch_func(args_list, kwargs_list)
        else:
            # this function has no batch implementation
            # call it separately
            return [
                self.func(*args, **kwargs)
                for args, kwargs in zip(args_list, kwargs_list)
            ]

    def batch(self, *delays):
        args_list, kwargs_list = self._gen_args_kwargs_list(delays)
        return self.call_with_lists(args_list, kwargs_list)

    def call_with_lists(self, args_list, kwargs_list):
        if self.is_async:
            return self._async_batch(args_list, kwargs_list)
        else:
            return self._sync_batch(args_list, kwargs_list)

    def bind(self, *args, **kwargs):
        if self.bind_func is None:
            raise TypeError(f"bind function not exist for method {self.func.__name__}")
        return self.bind_func(*args, **kwargs)


class _ExtensibleAccessor(_ExtensibleCallable):
    func: Callable
    batch_func: Optional[Callable]

    def __init__(self, func: Callable):
        self.func = func
        self.batch_func = None
        self.bind_func = build_args_binder(func, remove_self=True)
        self.is_async = asyncio.iscoroutinefunction(self.func)
        self.has_single_func = True

    def batch(self, func: Callable):
        self.batch_func = func
        return self

    def __get__(self, instance, owner):
        if instance is None:
            # calling from class
            return self.func

        func = self.func.__get__(instance, owner)
        batch_func = (
            self.batch_func.__get__(instance, owner)
            if self.batch_func is not None
            else None
        )
        bind_func = (
            self.bind_func.__get__(instance, owner)
            if self.bind_func is not None
            else None
        )

        return _ExtensibleWrapper(
            func, batch_func=batch_func, bind_func=bind_func, is_async=self.is_async
        )


def extensible(func: Callable):
    """
    `extensible` means this func could be functionality extended,
    especially for batch operations.

    Consider remote function calls, each function may have operations
    like opening file, closing file, batching them can help to reduce the cost,
    especially for remote function calls.

    Parameters
    ----------
    func : callable
        Function

    Returns
    -------
    func
    """
    return _ExtensibleAccessor(func)
