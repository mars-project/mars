# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

from urllib.parse import urlparse
from typing import Any, Dict, Type, Tuple

from .backend import get_backend
from .context import get_context
from .core import _Actor


async def create_actor(actor_cls, *args, uid=None, address=None, **kwargs):
    ctx = get_context()
    return await ctx.create_actor(actor_cls, *args, uid=uid, address=address, **kwargs)


async def has_actor(actor_ref):
    ctx = get_context()
    return await ctx.has_actor(actor_ref)


async def destroy_actor(actor_ref):
    ctx = get_context()
    return await ctx.destroy_actor(actor_ref)


async def actor_ref(*args, **kwargs):
    ctx = get_context()
    return await ctx.actor_ref(*args, **kwargs)


async def create_actor_pool(address: str,
                            n_process: int = None,
                            **kwargs):
    if address is None:
        raise ValueError('address has to be provided')
    if '://' not in address:
        scheme = None
    else:
        scheme = urlparse(address).scheme or None

    return await get_backend(scheme).create_actor_pool(
        address, n_process=n_process, **kwargs)


class Actor(_Actor):
    def __new__(cls, *args, **kwargs):
        try:
            return _actor_implementation[cls](*args, **kwargs)
        except KeyError:
            return super().__new__(cls, *args, **kwargs)

    async def __post_create__(self):
        """
        Method called after actor creation
        """
        return await super().__post_create__()

    async def __pre_destroy__(self):
        """
        Method called before actor destroy
        """
        return await super().__pre_destroy__()

    async def __on_receive__(self, message: Tuple[Any]):
        """
        Handle message from other actors and dispatch them to user methods

        Parameters
        ----------
        message : tuple
            Message shall be (method_name,) + args + (kwargs,)
        """
        return await super().__on_receive__(message)


_actor_implementation: Dict[Type[Actor], Type[Actor]] = dict()


def register_actor_implementation(actor_cls: Type[Actor], impl_cls: Type[Actor]):
    _actor_implementation[actor_cls] = impl_cls


def unregister_actor_implementation(actor_cls: Type[Actor]):
    try:
        del _actor_implementation[actor_cls]
    except KeyError:
        pass
