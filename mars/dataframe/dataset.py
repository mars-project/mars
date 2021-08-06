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
from ..utils import lazy_import
from typing import Iterable, List

ray = lazy_import('ray')
parallel_it = lazy_import('ray.util.iter')
ml_dataset = lazy_import('ray.util.data')


class RayObjectPiece:
    def __init__(self,
                 addr,
                 obj_ref):
        """Represents a single entity holding the object ref."""
        self.addr = addr
        self.obj_ref = obj_ref

    def read(self, shuffle: bool) -> pd.DataFrame:
        df: pd.DataFrame = ray.get(self.obj_ref)

        if shuffle:
            df = df.sample(frac=1.0)
        return df


class RecordBatch:
    def __init__(self,
                 shard_id: int,
                 prefix: str,
                 record_pieces: List[RayObjectPiece],
                 shuffle: bool,
                 shuffle_seed: int):
        """Holding a list of RayObjectPieces."""
        self._shard_id = shard_id
        self._prefix = prefix
        self.record_pieces = record_pieces
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def shard_id(self) -> int:
        return self._shard_id

    def __iter__(self) -> Iterable[pd.DataFrame]:
        """Returns the item_generator required from ParallelIteratorWorker."""
        if self.shuffle:
            np.random.seed(self.shuffle_seed)
            np.random.shuffle(self.record_pieces)

        for piece in self.record_pieces:
            yield piece.read(self.shuffle)


def _create_ml_dataset(name: str,
                       record_pieces: List[RayObjectPiece],
                       num_shards: int,
                       shuffle: bool,
                       shuffle_seed: int):
    if shuffle_seed:
        np.random.seed(shuffle_seed)
    else:
        np.random.seed(0)

    # TODO: (maybe) combine some chunks according to num_shards
    record_batches = []
    for rank, pieces in enumerate(record_pieces):
        record_batches.append(RecordBatch(shard_id=rank,
                                          prefix=name,
                                          record_pieces=[pieces],
                                          shuffle=shuffle,
                                          shuffle_seed=shuffle_seed))
    worker_cls = ray.remote(parallel_it.ParallelIteratorWorker)
    actors = [worker_cls.remote(g, False) for g in record_batches]
    it = parallel_it.from_actors(actors, name)
    ds = ml_dataset.from_parallel_iter(
        it, need_convert=False, batch_size=0, repeated=False)
    return ds


class RayMLDataset:
    @staticmethod
    def from_mars(df,
                  num_shards: int = None,
                  shuffle: bool = False,
                  shuffle_seed: int = None):
        if num_shards:
            df = df.rebalance(axis=1, num_partitions=1)
            df = df.rebalance(axis=0, num_partitions=num_shards)
            df.execute()
        # it's ensured that df has been executed
        chunk_addr_refs = df.fetch(only_refs=True)
        record_pieces = [RayObjectPiece(addr, obj_ref) for addr, obj_ref in chunk_addr_refs]
        return _create_ml_dataset("from_mars", record_pieces, num_shards, shuffle, shuffle_seed)
