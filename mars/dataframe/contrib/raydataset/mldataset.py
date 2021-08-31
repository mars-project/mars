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

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from ....utils import ceildiv, lazy_import

ray = lazy_import('ray')
parallel_it = lazy_import('ray.util.iter')
ml_dataset = lazy_import('ray.util.data')


class ChunkRefBatch:
    def __init__(self,
                 shard_id: int,
                 obj_refs: 'ray.ObjectRef'):
        """Iterable batch holding a list of ray.ObjectRefs.

        Args:
            shard_id (int): id of the shard
            prefix (str): prefix name of the batch
            obj_refs (List[ray.ObjectRefs]): list of ray.ObjectRefs
        """
        self._shard_id = shard_id
        self._obj_refs = obj_refs

    @property
    def shard_id(self) -> int:
        return self._shard_id

    def __iter__(self) -> Iterable[pd.DataFrame]:
        """Returns the item_generator required from ParallelIteratorWorker."""
        for obj_ref in self._obj_refs:
            yield ray.get(obj_ref)


def _group_chunk_refs(chunk_addr_refs: List[Tuple[Tuple, 'ray.ObjectRef']],
                      num_shards: int):
    """Group fetched ray.ObjectRefs into a dict for later use.

    Args:
        chunk_addr_refs (List[Tuple[Tuple, ray.ObjectRef]]): a list of tuples of
            band & ray.ObjectRef of each chunk.
        num_shards (int): the number of shards that will be created for the MLDataset.

    Returns:
        Dict[str, List[ray.ObjectRef]]: a dict that defines which group of ray.ObjectRefs will
            be in an ChunkRefBatch.
    """
    group_to_obj_refs = defaultdict(list)
    if not num_shards:
        for addr, obj_ref in chunk_addr_refs:
            group_to_obj_refs[addr].append(obj_ref)
    else:
        splits = np.array_split([ref for _, ref in chunk_addr_refs],
                                num_shards)
        for idx, split in enumerate(splits):
            group_to_obj_refs['group-' + str(idx)] = list(split)
    return group_to_obj_refs


def _create_ml_dataset(name: str,
                       group_to_obj_refs: Dict[str, List['ray.ObjectRef']]):
    record_batches = []
    for rank, obj_refs in enumerate(group_to_obj_refs.values()):
        record_batches.append(ChunkRefBatch(shard_id=rank,
                                            obj_refs=obj_refs))
    worker_cls = ray.remote(num_cpus=0)(parallel_it.ParallelIteratorWorker)
    actors = [worker_cls.remote(g, False) for g in record_batches]
    it = parallel_it.from_actors(actors, name)
    ds = ml_dataset.from_parallel_iter(
        it, need_convert=False, batch_size=0, repeated=False)
    return ds


def _rechunk_if_needed(df, num_shards: int = None):
    chunk_size = df.extra_params.raw_chunk_size or max(df.shape)
    num_rows = df.shape[0]
    num_columns = df.shape[1]
    # if chunk size not set, num_chunks_in_row = 1
    # if chunk size is set more than max(df.shape), num_chunks_in_row = 1
    # otherwise, num_chunks_in_row depends on ceildiv(num_rows, chunk_size)
    num_chunks_in_row = ceildiv(num_rows, chunk_size)
    naive_num_partitions = ceildiv(num_rows, num_columns)

    need_re_execute = False
    # ensure each part holds all columns
    if chunk_size < num_columns:
        df = df.rebalance(axis=1, num_partitions=1)
        need_re_execute = True
    if num_shards and num_chunks_in_row < num_shards:
        df = df.rebalance(axis=0, num_partitions=num_shards)
        need_re_execute = True
    if not num_shards and num_chunks_in_row == 1:
        df = df.rebalance(axis=0, num_partitions=naive_num_partitions)
        need_re_execute = True
    if need_re_execute:
        df.execute()
    return df


def to_ray_mldataset(df,
                     num_shards: int = None):
    """Create a MLDataset from Mars DataFrame

    Args:
        df (mars.dataframe.Dataframe): the Mars DataFrame
        num_shards (int, optional): the number of shards that will be created
            for the MLDataset. Defaults to None.
            If num_shards is None, chunks will be grouped by nodes where they lie.
            Otherwise, chunks will be grouped by their order in DataFrame.

    Returns:
        a MLDataset
    """
    df = _rechunk_if_needed(df, num_shards)
    # chunk_addr_refs is fetched directly rather than in batches
    # during `fetch` procedure, it'll be checked that df has been executed
    # items in chunk_addr_refs are ordered by positions in df
    # while adjacent chunks may belong to different addrs, i.e.
    #       chunk1 for addr1,
    #       chunk2 & chunk3 for addr2,
    #       chunk4 for addr1
    fetched_infos: Dict[str, List] = df.fetch_infos(fields=['band', 'object_id'])
    chunk_addr_refs: List[Tuple[Tuple, 'ray.ObjectRef']] = [(band, object_id) for band, object_id in
                                                            zip(fetched_infos['band'],
                                                                fetched_infos['object_id'])]
    group_to_obj_refs: Dict[str, List[ray.ObjectRef]] = _group_chunk_refs(chunk_addr_refs, num_shards)
    return _create_ml_dataset("from_mars", group_to_obj_refs)
