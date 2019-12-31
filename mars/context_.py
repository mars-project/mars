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

import random
import itertools
from collections import defaultdict
from typing import List

from .utils import merge_chunks
from .worker.api import WorkerAPI
from .scheduler.api import MetaAPI
from .tensor.datasource import empty
from .tensor.indexing.getitem import TensorIndexTilesHandler, slice_split
from .tensor.core import TENSOR_TYPE


class Context:
    def __init__(self):
        self._worker_api = WorkerAPI()
        self._meta_api = MetaAPI()

    # Meta API
    def get_tileable_metas(self, session_id, graph_key, tileable_keys, filter_fields: List[str]=None) -> List:
        return self._meta_api.get_tileable_metas(session_id, graph_key, tileable_keys, filter_fields)

    def get_chunk_metas(self, session_id, chunk_keys, filter_fields: List[str] = None) -> List:
        return self._meta_api.get_chunk_metas(session_id, chunk_keys, filter_fields)

    # Worker API
    def get_chunks_data(self, session_id, worker: str, chunk_keys: List[str], indexes: List=None,
                        compression_types: List[str]=None) -> List:
        return self._worker_api.get_chunks_data(session_id, worker, chunk_keys, indexes=indexes,
                                                compression_types=compression_types)

    # Fetch tileable data by tileable keys and indexes.
    def get_tileable_data(self, session_id, graph_key: str, tileable_key: str, indexes: List=None,
                          compression_types: List[str]=None):
        nsplits, chunk_keys, chunk_indexes = self.get_tileable_metas(session_id, graph_key, [tileable_key])
        chunk_idx_to_keys = dict(zip(chunk_indexes, chunk_keys))
        chunk_keys_to_idx = dict(zip(chunk_keys, chunk_indexes))
        endpoints = self.get_chunk_metas(session_id, chunk_keys, filter_fields=['workers'])
        chunk_keys_to_worker = dict((chunk_key, random.choice(es)) for es, chunk_key in zip(endpoints, chunk_keys))
        chunk_workers = defaultdict(list)
        [chunk_workers[e].append(chunk_key) for e, chunk_key in chunk_keys_to_worker.items()]

        chunk_results = dict()
        if not indexes:
            for endpoint, chunks in chunk_workers.items():
                datas = self.get_chunks_data(session_id, endpoint, chunks, compression_types=compression_types)
                chunk_results.update(dict(zip([chunk_keys_to_idx[k] for k in chunks], datas)))
        elif all(isinstance(ind, slice) for ind in indexes):
            chunk_results = dict()
            indexes = dict()
            for axis, s in enumerate(indexes):
                idx_to_slices = slice_split(s, nsplits[axis])
                indexes[axis] = idx_to_slices

            result_chunks = dict()
            for chunk_index in itertools.product(*[v.keys() for v in indexes.values()]):
                slice_obj = tuple(indexes[axis][chunk_idx] for axis, chunk_idx in enumerate(chunk_index))
                chunk_key = chunk_idx_to_keys[chunk_index]
                result_chunks[chunk_key] = (chunk_index, slice_obj)
            for endpoint, chunks in chunk_workers.items():
                to_fetch_keys = []
                to_fetch_indexes = []
                to_fetch_idx = []
                for r_chunk, (chunk_index, slice_obj) in result_chunks.items():
                    if r_chunk in chunks:
                        to_fetch_keys.append(r_chunk)
                        to_fetch_indexes.append(to_fetch_indexes)
                        to_fetch_idx.append(chunk_index)
                if to_fetch_keys:
                    datas = self.get_chunks_data(session_id, endpoint, to_fetch_keys, indexes=to_fetch_indexes,
                                                 compression_types=compression_types)
                    chunk_results.update(dict(zip(to_fetch_idx, datas)))
        else:
            if any(isinstance(ind, TENSOR_TYPE) for ind in indexes):
                raise TypeError("Doesn't support indexing by tensors")
            # Reuse the getitem logic to get each chunk's indexes
            tileable_shape = tuple(sum(s) for s in nsplits)
            empty_tileable = empty(tileable_shape, chunk_size=nsplits)._inplace_tile()
            indexed = empty_tileable[indexes]
            index_handler = TensorIndexTilesHandler(indexed.op)
            index_handler._extract_indexes_info()
            index_handler._preprocess_fancy_indexes()
            index_handler._process_fancy_indexes()
            index_handler._process_in_tensor()

            result_chunks = dict()
            for c in index_handler._out_chunks:
                result_chunks[c.inputs[0].key] = [c.index, c.op.indexes]

            for endpoint, chunks in chunk_workers.items():
                to_fetch_keys = []
                to_fetch_indexes = []
                to_fetch_idx = []
                for r_chunk, (chunk_index, slice_obj) in result_chunks.items():
                    if r_chunk in chunks:
                        to_fetch_keys.append(r_chunk)
                        to_fetch_indexes.append(to_fetch_indexes)
                        to_fetch_idx.append(chunk_index)
                if to_fetch_keys:
                    datas = self.get_chunks_data(session_id, endpoint, to_fetch_keys, indexes=to_fetch_indexes,
                                                 compression_types=compression_types)
                    chunk_results.update(dict(zip(to_fetch_idx, datas)))

        if len(chunk_results) == 1:
            ret = list(chunk_results.values())[0]
        else:
            ret = merge_chunks(chunk_results)
        return ret

