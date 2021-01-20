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

from .context import get_context


async def create_actor(actor_cls, *args, uid=None, address=None, **kwargs):
    ctx = get_context()
    return await ctx.create_actor(actor_cls, *args, uid=uid, address=address, **kwargs)


async def has_actor(actor_ref):
    ctx = get_context()
    return await ctx.has_actor(actor_ref)


async def destroy_actor(actor_ref):
    ctx = get_context()
    return await ctx.destroy_actor(actor_ref)


def actor_ref(*args, **kwargs):
    ctx = get_context()
    return ctx.actor_ref(*args, **kwargs)
