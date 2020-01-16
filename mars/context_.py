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
from .serialize import dataserializer
from .worker.api import WorkerAPI
from .scheduler.api import MetaAPI
from .tensor.datasource import empty
from .tensor.indexing.getitem import TensorIndexTilesHandler, slice_split
from .tensor.core import TENSOR_TYPE


class Context:
    def __init__(self, actor_ctx=None, scheduler_endpoint=None):
        self._worker_api = WorkerAPI()
        self._meta_api = MetaAPI(actor_ctx=actor_ctx, scheduler_endpoint=scheduler_endpoint)

    # Meta API
    def get_tileable_metas(self, session_id, tileable_keys, filter_fields: List[str]=None) -> List:
        return self._meta_api.get_tileable_metas(session_id, tileable_keys, filter_fields)

    def get_chunk_metas(self, session_id, chunk_keys, filter_fields: List[str] = None) -> List:
        return self._meta_api.get_chunk_metas(session_id, chunk_keys, filter_fields)

    def get_tileable_key_by_name(self, session_id, name: str):
        return self._meta_api.get_tileable_key_by_name(session_id, name)

    # Worker API
    def get_chunks_data(self, session_id, worker: str, chunk_keys: List[str], indexes: List=None,
                        compression_types: List[str]=None):
        return self._worker_api.get_chunks_data(session_id, worker, chunk_keys, indexes=indexes,
                                                compression_types=compression_types)

    # Fetch tileable data by tileable keys and indexes.
    def get_tileable_data(self, session_id, tileable_key: str, indexes: List=None,
                          compression_types: List[str]=None):
        nsplits, chunk_keys, chunk_indexes = self.get_tileable_metas(session_id, [tileable_key])[0]
        chunk_idx_to_keys = dict(zip(chunk_indexes, chunk_keys))
        chunk_keys_to_idx = dict(zip(chunk_keys, chunk_indexes))
        endpoints = self.get_chunk_metas(session_id, chunk_keys, filter_fields=['workers'])
        chunk_keys_to_worker = dict((chunk_key, random.choice(es)) for es, chunk_key in zip(endpoints, chunk_keys))

        chunk_workers = defaultdict(list)
        [chunk_workers[e].append(chunk_key) for chunk_key, e in chunk_keys_to_worker.items()]

        chunk_results = dict()
        if not indexes:
            datas = []
            for endpoint, chunks in chunk_workers.items():
                datas.append(self.get_chunks_data(session_id, endpoint, chunks, compression_types=compression_types))
            datas = [d.result() for d in datas]
            for (endpoint, chunks), d in zip(chunk_workers.items(), datas):
                d = [dataserializer.loads(db) for db in d]
                chunk_results.update(dict(zip([chunk_keys_to_idx[k] for k in chunks], d)))
        elif all(isinstance(ind, slice) for ind in indexes):
            axis_slices = dict()
            for axis, s in enumerate(indexes):
                idx_to_slices = slice_split(s, nsplits[axis])
                axis_slices[axis] = idx_to_slices

            result_chunks = dict()
            for chunk_index in itertools.product(*[v.keys() for v in axis_slices.values()]):
                slice_obj = tuple(axis_slices[axis][chunk_idx] for axis, chunk_idx in enumerate(chunk_index))
                chunk_key = chunk_idx_to_keys[chunk_index]
                result_chunks[chunk_key] = (chunk_index, slice_obj)

            chunk_datas = dict()
            for endpoint, chunks in chunk_workers.items():
                to_fetch_keys = []
                to_fetch_indexes = []
                to_fetch_idx = []
                for r_chunk, (chunk_index, slice_obj) in result_chunks.items():
                    if r_chunk in chunks:
                        to_fetch_keys.append(r_chunk)
                        to_fetch_indexes.append(slice_obj)
                        to_fetch_idx.append(chunk_index)
                if to_fetch_keys:
                    datas = self.get_chunks_data(session_id, endpoint, to_fetch_keys, indexes=to_fetch_indexes,
                                                 compression_types=compression_types)
                    chunk_datas[tuple(to_fetch_idx)] = datas
            chunk_datas = dict((k, v.result()) for k, v in chunk_datas.items())
            for idx, d in chunk_datas.items():
                d = [dataserializer.loads(db) for db in d]
                chunk_results.update(dict(zip(idx, d)))
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

        chunk_results = [(k, v) for k, v in chunk_results.items()]
        if len(chunk_results) == 1:
            ret = chunk_results[0][1]
        else:
            ret = merge_chunks(chunk_results)
        return ret

