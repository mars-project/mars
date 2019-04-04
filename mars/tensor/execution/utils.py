# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

try:
    import tiledb
except (ImportError, OSError):  # pragma: no cover
    tildb = None

from ...compat import functools32


# As TileDB Ctx's creation is a bit time-consuming,
# we just cache the Ctx
# also remember the arguments should be hashable
@functools32.lru_cache(10)
def _create_tiledb_ctx(conf_tuple):
    if conf_tuple is not None:
        return tiledb.Ctx(dict(conf_tuple))
    return tiledb.Ctx()


def get_tiledb_ctx(conf):
    key = tuple(conf.items()) if conf is not None else None
    return _create_tiledb_ctx(key)


def estimate_fuse_size(ctx, chunk):
    from ...graph import DAG
    from .core import Executor

    dag = DAG()
    keys = set(c.key for c in chunk.composed)
    for c in chunk.composed:
        dag.add_node(c)
        for inp in c.inputs:
            if inp.key not in keys:
                continue
            if inp not in dag:
                dag.add_node(inp)
            dag.add_edge(inp, c)

    size_ctx = ctx.copy()
    executor = Executor(storage=size_ctx)
    ctx[chunk.key] = executor.execute_graph(dag, [chunk.key], mock=True)[0]
