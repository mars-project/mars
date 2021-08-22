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
from ..utils import ceildiv, lazy_import
from collections import defaultdict
from typing import Dict, Iterable, List

ray = lazy_import('ray')
parallel_it = lazy_import('ray.util.iter')
ml_dataset = lazy_import('ray.util.data')


class RayObjectPiece:
    def __init__(self,
                 index,
                 addr,
                 obj_ref):
        """Represents a single entity holding the object ref.

        Args:
            index (int): index of the data in DataFrame
            addr (BandType): band that stores the data
            obj_ref (ray.ObjectRef): ObjectRef to the data
        """
        self._index = index
        self._addr = addr
        self._obj_ref = obj_ref

    @property
    def index(self):
        return self._index

    @property
    def addr(self):
        return self._addr

    def read(self) -> pd.DataFrame:
        return ray.get(self._obj_ref)


class RecordBatch:
    def __init__(self,
                 shard_id: int,
                 record_pieces: List[RayObjectPiece]):
        """Iterable batch holding a list of RayObjectPieces.

        Args:
            shard_id (int): id of the shard
            prefix (str): prefix name of the batch
            record_pieces (List[RayObjectPiece]): list of RayObjectRefs
        """
        self._shard_id = shard_id
        self._record_pieces = record_pieces

    @property
    def shard_id(self) -> int:
        return self._shard_id

    def __iter__(self) -> Iterable[pd.DataFrame]:
        """Returns the item_generator required from ParallelIteratorWorker."""
        for piece in self._record_pieces:
            yield piece.read()


def _group_pieces(index_to_pieces: Dict[int, RayObjectPiece],
                  num_shards: int,
                  num_addrs: int):
    group_to_pieces = defaultdict(list)
    if not num_shards or num_shards == num_addrs:
        for piece in index_to_pieces.values():
            group_to_pieces[str(piece.addr)].append(piece)
    else:
        splits = np.array_split(list(index_to_pieces.values()), num_shards)
        for idx, split in enumerate(splits):
            pieces = list(split)
            group_to_pieces['group-' + str(idx)] = pieces
    return group_to_pieces


def _create_ml_dataset(name: str,
                       group_to_pieces: Dict[str, List[RayObjectPiece]]):
    record_batches = []
    for rank, pieces in enumerate(group_to_pieces.values()):
        record_batches.append(RecordBatch(shard_id=rank,
                                          record_pieces=pieces))
    worker_cls = ray.remote(parallel_it.ParallelIteratorWorker)
    actors = [worker_cls.remote(g, False) for g in record_batches]
    it = parallel_it.from_actors(actors, name)
    ds = ml_dataset.from_parallel_iter(
        it, need_convert=False, batch_size=0, repeated=False)
    return ds


def _rechunk_if_needed(df, num_partitions: int=None):
    num_rows = df.shape[0]
    num_columns = df.shape[1]
    need_re_execute = False
    chunk_size = df.extra_params.raw_chunk_size or max(df.shape)

    if chunk_size < num_columns:
        # ensure each part holds all columns
        df = df.rebalance(axis=1, num_partitions=1)
        need_re_execute = True
    if num_partitions and chunk_size > ceildiv(num_rows, num_partitions):
        # ensure enough parts for num_partitions
        df = df.rebalance(axis=0, num_partitions=num_partitions)
        need_re_execute = True
    if need_re_execute:
        df.execute()
    return df


def to_ray_mldataset(df,
                     num_partitions: int = None,
                     num_shards: int = None):
    """Create a MLDataset from Mars DataFrame

    Args:
        df (mars.dataframe.Dataframe): the Mars DataFrame
        num_partitions (int, optional): the number of partitions into which
            the df will be divided. Defaults to None.
        num_shards (int, optional): the number of shards that will be created
            for the MLDataset. Defaults to None.

    Returns:
        a MLDataset
    """
    if num_partitions and num_shards:
        assert num_partitions % num_shards == 0,\
                f"num_partitions: {num_partitions} should be a multiple of "\
                f"num_shards: {num_shards}"
    df = _rechunk_if_needed(df, num_partitions=num_partitions)
    # chunk_addr_refs is fetched directly rather than in batches
    # during `fetch` procedure, it'll be checked that df has been executed
    chunk_addr_refs = df.fetch(only_refs=True)
    addrs = set()
    index_to_pieces = {}
    # items in chunk_addr_refs are ordered by positions in df
    # while adjacent chunks may belong to different addrs, i.e.
    #       chunk1 for addr1,
    #       chunk2 & chunk3 for addr2,
    #       chunk4 for addr1
    for idx, (addr, obj_ref) in enumerate(chunk_addr_refs):
        index_to_pieces[idx] = RayObjectPiece(idx, addr, obj_ref)
        addrs.add(addr)
    group_to_pieces = _group_pieces(index_to_pieces, num_shards,
                                    num_addrs=len(list(addrs)))
    return _create_ml_dataset("from_mars", group_to_pieces)
