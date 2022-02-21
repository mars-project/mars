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

from ....utils import lazy_import
from .mldataset import _rechunk_if_needed
from typing import Dict, List

ray = lazy_import("ray")
# Ray Datasets is available in early preview at ray.data with Ray 1.6+
# (and ray.experimental.data in Ray 1.5)
ray_dataset = lazy_import("ray.data")


def to_ray_dataset(df, num_shards: int = None):
    """Create a Ray Dataset from Mars DataFrame

    Args:
        df (mars.dataframe.Dataframe): the Mars DataFrame
        num_shards (int, optional): the number of shards that will be created
            for the Ray Dataset. Defaults to None.
            If num_shards is None, chunks will be grouped by nodes where they lie.
            Otherwise, chunks will be grouped by their order in DataFrame.

    Returns:
        a Ray Dataset
    """
    df = _rechunk_if_needed(df, num_shards)
    # chunk_addr_refs is fetched directly rather than in batches
    # during `fetch` procedure, it'll be checked that df has been executed
    # items in chunk_addr_refs are ordered by positions in df
    # while adjacent chunks may belong to different addrs, i.e.
    #       chunk1 for addr1,
    #       chunk2 & chunk3 for addr2,
    #       chunk4 for addr1
    chunk_refs: List["ray.ObjectRef"] = get_chunk_refs(df)
    dataset = ray_dataset.from_pandas_refs(chunk_refs)
    # Hold mars dataframe to avoid mars dataframe and ray object gc.
    dataset.dataframe = df

    def __getstate__():
        state = dataset.__dict__.copy()
        state.pop("dataframe", None)
        return state

    # `dataframe` is not serializable by ray.
    dataset.__getstate__ = __getstate__
    return dataset


def get_chunk_refs(df):
    fetched_infos: Dict[str, List] = df.fetch_infos(fields=["object_id"])
    return fetched_infos["object_id"]
