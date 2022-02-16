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

from ....utils import lazy_import

ray = lazy_import("ray")
parallel_it = lazy_import("ray.util.iter")
ml_dataset = lazy_import("ray.util.data")


class ChunkRefBatch:
    def __init__(self, shard_id: int, obj_refs: "ray.ObjectRef"):
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


def _group_chunk_refs(
    chunk_addr_refs: List[Tuple[Tuple, "ray.ObjectRef"]], num_shards: int
):
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
        splits = np.array_split([ref for _, ref in chunk_addr_refs], num_shards)
        for idx, split in enumerate(splits):
            group_to_obj_refs["group-" + str(idx)] = list(split)
    return group_to_obj_refs


def _rechunk_if_needed(df, num_shards: int = None):
    try:
        if num_shards:
            assert isinstance(num_shards, int) and num_shards > 0
            df = df.rebalance(axis=0, num_partitions=num_shards)
        df = df.rechunk({1: df.shape[1]})
        df = df.reset_index(drop=True)
        return df.execute()
    except Exception as e:  # pragma: no cover
        raise Exception(f"rechunk failed df.shape {df.shape}") from e


def to_ray_mldataset(df, num_shards: int = None):
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
    fetched_infos: Dict[str, List] = df.fetch_infos(fields=["band", "object_id"])
    chunk_addr_refs: List[Tuple[Tuple, "ray.ObjectRef"]] = [
        (band, object_id)
        for band, object_id in zip(fetched_infos["band"], fetched_infos["object_id"])
    ]
    group_to_obj_refs: Dict[str, List[ray.ObjectRef]] = _group_chunk_refs(
        chunk_addr_refs, num_shards
    )

    record_batches = []
    for rank, obj_refs in enumerate(group_to_obj_refs.values()):
        record_batches.append(ChunkRefBatch(shard_id=rank, obj_refs=obj_refs))
    worker_cls = ray.remote(num_cpus=0)(parallel_it.ParallelIteratorWorker)
    actors = [worker_cls.remote(g, False) for g in record_batches]
    it = parallel_it.from_actors(actors, "from_mars")
    dataset = ml_dataset.from_parallel_iter(it, need_convert=False, batch_size=0)
    # Hold mars dataframe to avoid mars dataframe and ray object gc.
    dataset.dataframe = df

    def __getstate__():
        state = dataset.__dict__.copy()
        state.pop("dataframe", None)
        return state

    # `dataframe` is not serializable by ray.
    dataset.__getstate__ = __getstate__
    return dataset
