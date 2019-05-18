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
    import pandas as pd
except ImportError:  # pragma: no cover
    pass


def _concat(ctx, chunk):
    inputs = [ctx[input.key] for input in chunk.inputs]

    if isinstance(inputs[0], tuple):
        ctx[chunk.key] = tuple(_base_concat(chunk, [input[i] for input in inputs])
                               for i in range(len(inputs[0])))
    else:
        ctx[chunk.key] = _base_concat(chunk, inputs)


def _base_concat(chunk, inputs):
    if chunk.op.axis is not None:
        # TODO: remove this when we support concat on dataframe
        raise NotImplementedError
    else:
        # auto generated concat when executing a dataframe
        n_rows = max(inp.index[0] for inp in chunk.inputs) + 1
        n_cols = int(len(inputs) // n_rows)
        assert n_rows * n_cols == len(inputs)

        concats = []
        for i in range(n_rows):
            concat = pd.concat([inputs[i * n_cols + j] for j in range(n_cols)], axis='columns')
            concats.append(concat)

        ret = pd.concat(concats)
        if getattr(chunk.index_value, 'should_be_monotonic', False):
            ret.sort_index(inplace=True)
        if getattr(chunk.columns, 'should_be_monotonic', False):
            ret.sort_index(axis=1, inplace=True)
        return ret


def register_merge_handler():
    from ..expressions.merge import DataFrameConcat
    from ...executor import register

    register(DataFrameConcat, _concat)
