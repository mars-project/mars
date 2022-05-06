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

import asyncio
import itertools
import logging

from collections import defaultdict
from typing import Dict, List, Optional

from .... import oscar as mo
from ...meta.api import MetaAPI
from ...storage.api import StorageAPI
from ..errors import TileableNotTracked

logger = logging.getLogger(__name__)


class LifecycleTrackerActor(mo.Actor):
    _meta_api: MetaAPI

    def __init__(self, session_id: str):
        self._session_id = session_id
        self._tileable_key_to_chunk_keys = dict()
        self._tileable_ref_counts = defaultdict(lambda: 0)
        self._chunk_ref_counts = defaultdict(lambda: 0)

        self._meta_api: Optional[MetaAPI] = None

    async def __post_create__(self):
        self._meta_api = await MetaAPI.create(self._session_id, self.address)

    async def __pre_destroy__(self):
        chunk_keys = [
            chunk_key
            for chunk_key, ref_count in self._chunk_ref_counts.items()
            if ref_count > 0
        ]
        # remove all chunks
        await self._remove_chunks(chunk_keys)

    @staticmethod
    def gen_uid(session_id):
        return f"{session_id}_lifecycle_tracker"

    def _track(self, tileable_key: str, chunk_keys: List[str]):
        if tileable_key not in self._tileable_key_to_chunk_keys:
            self._tileable_key_to_chunk_keys[tileable_key] = []
        chunk_keys_set = set(self._tileable_key_to_chunk_keys[tileable_key])
        incref_chunk_keys = []
        tileable_ref_count = self._tileable_ref_counts.get(tileable_key, 0)
        for chunk_key in chunk_keys:
            if chunk_key in chunk_keys_set:
                continue
            if tileable_ref_count > 0:
                incref_chunk_keys.extend([chunk_key] * tileable_ref_count)
            self._tileable_key_to_chunk_keys[tileable_key].append(chunk_key)
        if incref_chunk_keys:
            self._incref_chunks(incref_chunk_keys)

    @mo.extensible
    async def track(self, tileable_key: str, chunk_keys: List[str]):
        return await asyncio.to_thread(self._track, tileable_key, chunk_keys)

    @classmethod
    def _check_ref_counts(cls, keys: List[str], ref_counts: List[int]):
        if ref_counts is not None and len(keys) != len(ref_counts):
            raise ValueError(
                f"`ref_counts` should have same size as `keys`, expect {len(keys)}, got {len(ref_counts)}"
            )

    def _incref_chunks(self, chunk_keys: List[str], counts: List[int] = None):
        counts = counts if counts is not None else itertools.repeat(1)
        for chunk_key, count in zip(chunk_keys, counts):
            self._chunk_ref_counts[chunk_key] += count

    async def incref_chunks(self, chunk_keys: List[str], counts: List[int] = None):
        logger.debug(
            "Increase reference count for chunks %s",
            {ck: self._chunk_ref_counts[ck] for ck in chunk_keys},
        )
        self._check_ref_counts(chunk_keys, counts)
        return await asyncio.to_thread(self._incref_chunks, chunk_keys, counts=counts)

    def _get_remove_chunk_keys(self, chunk_keys: List[str], counts: List[int] = None):
        to_remove_chunk_keys = []
        counts = counts if counts is not None else itertools.repeat(1)
        for chunk_key, count in zip(chunk_keys, counts):
            ref_count = self._chunk_ref_counts[chunk_key]
            ref_count -= count
            assert ref_count >= 0, f"chunk key {chunk_key} will have negative ref count"
            self._chunk_ref_counts[chunk_key] = ref_count
            if ref_count == 0:
                # remove
                to_remove_chunk_keys.append(chunk_key)
        return to_remove_chunk_keys

    async def decref_chunks(self, chunk_keys: List[str], counts: List[int] = None):
        self._check_ref_counts(chunk_keys, counts)
        logger.debug(
            "Decrease reference count for chunks %s",
            {ck: self._chunk_ref_counts[ck] for ck in chunk_keys},
        )
        to_remove_chunk_keys = await asyncio.to_thread(
            self._get_remove_chunk_keys, chunk_keys, counts=counts
        )
        # make _remove_chunks release actor lock so that multiple `decref_chunks` can run concurrently.
        yield self._remove_chunks(to_remove_chunk_keys)

    async def _remove_chunks(self, to_remove_chunk_keys: List[str]):
        if not to_remove_chunk_keys:
            return
        # get meta
        logger.debug("Remove chunks %s with a refcount of zero", to_remove_chunk_keys)
        get_metas = []
        for to_remove_chunk_key in to_remove_chunk_keys:
            get_metas.append(
                self._meta_api.get_chunk_meta.delay(
                    to_remove_chunk_key, fields=["bands"], error="ignore"
                )
            )
        metas = await self._meta_api.get_chunk_meta.batch(*get_metas)

        # filter chunks that not exist
        new_to_remove_chunk_keys = []
        new_metas = []
        for to_remove_chunk_key, meta in zip(to_remove_chunk_keys, metas):
            if meta is not None:
                new_to_remove_chunk_keys.append(to_remove_chunk_key)
                new_metas.append(meta)
        to_remove_chunk_keys = new_to_remove_chunk_keys
        metas = new_metas

        all_bands = [meta["bands"] for meta in metas]
        key_to_addresses = dict()
        for to_remove_chunk_key, bands in zip(to_remove_chunk_keys, all_bands):
            key_to_addresses[to_remove_chunk_key] = bands

        # remove data via storage API
        storage_api_to_deletes = defaultdict(list)
        for key, bands in key_to_addresses.items():
            for band in bands:
                # storage API is cached for same arguments
                storage_api = await StorageAPI.create(
                    self._session_id, band[0], band[1]
                )
                storage_api_to_deletes[storage_api].append(
                    storage_api.delete.delay(key, error="ignore")
                )
        await asyncio.gather(
            *[
                storage_api.delete.batch(*deletes)
                for storage_api, deletes in storage_api_to_deletes.items()
            ]
        )

        # delete meta
        delete_metas = []
        for to_remove_chunk_key in to_remove_chunk_keys:
            delete_metas.append(
                self._meta_api.del_chunk_meta.delay(to_remove_chunk_key)
            )
        await self._meta_api.del_chunk_meta.batch(*delete_metas)

    def get_chunk_ref_counts(self, chunk_keys: List[str]) -> List[int]:
        return [self._chunk_ref_counts[chunk_key] for chunk_key in chunk_keys]

    def get_all_chunk_ref_counts(self) -> Dict[str, int]:
        result = dict()
        for chunk_key, ref_count in self._chunk_ref_counts.items():
            if ref_count > 0:
                result[chunk_key] = ref_count
        return result

    def _incref_tileables(self, tileable_keys: List[str], counts: List[int] = None):
        counts = counts if counts is not None else itertools.repeat(1)
        for tileable_key, count in zip(tileable_keys, counts):
            if tileable_key not in self._tileable_key_to_chunk_keys:
                raise TileableNotTracked(f"tileable {tileable_key} not tracked before")
            self._tileable_ref_counts[tileable_key] += count
            incref_chunk_keys = self._tileable_key_to_chunk_keys[tileable_key]
            # incref chunks for this tileable
            logger.debug(
                "Incref chunks %s while increfing tileable %s",
                incref_chunk_keys,
                tileable_key,
            )
            chunk_counts = None if count == 1 else [count] * len(incref_chunk_keys)
            self._incref_chunks(incref_chunk_keys, counts=chunk_counts)

    async def incref_tileables(
        self, tileable_keys: List[str], counts: List[int] = None
    ):
        self._check_ref_counts(tileable_keys, counts)
        return await asyncio.to_thread(
            self._incref_tileables, tileable_keys, counts=counts
        )

    def _get_decref_chunk_keys(
        self, tileable_keys: List[str], counts: List[int] = None
    ) -> Dict[str, int]:
        decref_chunk_keys = dict()
        counts = counts if counts is not None else itertools.repeat(1)
        for tileable_key, count in zip(tileable_keys, counts):
            if tileable_key not in self._tileable_key_to_chunk_keys:
                raise TileableNotTracked(f"tileable {tileable_key} not tracked before")
            self._tileable_ref_counts[tileable_key] -= count

            for chunk_key in self._tileable_key_to_chunk_keys[tileable_key]:
                if chunk_key not in decref_chunk_keys:
                    decref_chunk_keys[chunk_key] = count
                else:
                    decref_chunk_keys[chunk_key] += count
        logger.debug(
            "Decref chunks %s while decrefing tileables %s",
            decref_chunk_keys,
            tileable_keys,
        )
        return decref_chunk_keys

    async def decref_tileables(
        self, tileable_keys: List[str], counts: List[int] = None
    ):
        self._check_ref_counts(tileable_keys, counts)
        decref_chunk_key_to_counts = await asyncio.to_thread(
            self._get_decref_chunk_keys, tileable_keys, counts=counts
        )
        to_remove_chunk_keys = await asyncio.to_thread(
            self._get_remove_chunk_keys,
            list(decref_chunk_key_to_counts),
            counts=list(decref_chunk_key_to_counts.values()),
        )
        # make _remove_chunks release actor lock
        yield self._remove_chunks(to_remove_chunk_keys)

    def get_tileable_ref_counts(self, tileable_keys: List[str]) -> List[int]:
        return [
            self._tileable_ref_counts[tileable_key] for tileable_key in tileable_keys
        ]
