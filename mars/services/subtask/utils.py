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

from typing import Any, Dict, List, Iterator, Tuple
from ...core import ChunkGraph
from ...core.operand import (
    Fetch,
    FetchShuffle,
    MapReduceOperand,
    VirtualOperand,
)
from .core import Subtask


def iter_input_data_keys(
    subtask: Subtask,
    chunk_graph: ChunkGraph,
    chunk_key_to_data_keys: Dict[str, List[str]],
) -> Iterator[Tuple[str, bool]]:
    """An iterator yield (input data key, is shuffle)."""
    data_keys = set()
    for chunk in chunk_graph.iter_indep():
        if isinstance(chunk.op, Fetch) and chunk.key not in subtask.pure_depend_keys:
            data_keys.add(chunk.key)
            yield chunk.key, False
        elif isinstance(chunk.op, FetchShuffle):
            for key in chunk_key_to_data_keys[chunk.key]:
                if key not in data_keys:
                    data_keys.add(key)
                    yield key, True


def get_mapper_data_keys(key: str, context: Dict[str, Any]) -> List[str]:
    """Get the mapper data keys of key from context."""
    return [
        store_key
        for store_key in context
        if isinstance(store_key, tuple) and store_key[0] == key
    ]


def iter_output_data(
    chunk_graph: ChunkGraph, context: Dict[str, Any]
) -> Iterator[Tuple[str, Any, bool]]:
    """An iterator yield (output chunk key, output data, is shuffle)."""
    data_keys = set()
    for result_chunk in chunk_graph.result_chunks:
        # skip virtual operands for result chunks
        if isinstance(result_chunk.op, VirtualOperand):
            continue
        key = result_chunk.key
        if key in context:
            # non shuffle op
            data = context[key]
            # update meta
            if not isinstance(data, tuple):
                result_chunk.params = result_chunk.get_params_from_data(data)
            # check key after update meta
            if key in data_keys:
                continue
            yield key, data, False
            data_keys.add(key)
        else:
            assert isinstance(result_chunk.op, MapReduceOperand)
            keys = get_mapper_data_keys(key, context)
            for key in keys:
                if key in data_keys:
                    continue
                # shuffle op
                data = context[key]
                yield key, data, True
                data_keys.add(key)
