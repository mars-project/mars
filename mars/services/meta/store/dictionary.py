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

from abc import ABCMeta
from typing import Dict, List, Tuple, Any

from ....utils import implements
from .base import AbstractMetaStore, register_meta_store, object_types


class DictMetaStoreType(ABCMeta):
    def __new__(mcs, name: str, bases: Tuple, namespace: Dict):
        for tp in object_types:
            # set
            for method_name in (f'set_{tp}_meta', f'set_{tp}_chunk_meta'):
                @implements(getattr(AbstractMetaStore, method_name))
                async def _set(self, object_id: str, **meta):
                    return self._set_meta(object_id, **meta)

                namespace[method_name] = _set
            # get
            for method_name in (f'get_{tp}_meta', f'get_{tp}_chunk_meta'):
                @implements(getattr(AbstractMetaStore, method_name))
                async def _get(self, object_id: str, fields: List[str] = None):
                    return self._get_meta(object_id, fields=fields)

                namespace[method_name] = _get
            # del
            for method_name in (f'del_{tp}_meta', f'del_{tp}_chunk_meta'):
                @implements(getattr(AbstractMetaStore, method_name))
                async def _del(self, object_id: str):
                    return self._del_meta(object_id)

                namespace[method_name] = _del

        return ABCMeta.__new__(mcs, name, bases, namespace)


@register_meta_store
class DictMetaStore(AbstractMetaStore, metaclass=DictMetaStoreType):
    name = 'mock'

    def __init__(self, session_id: str, **kw):
        super().__init__(session_id)
        self._store: Dict[str, Dict[str, Any]] = dict()
        if kw:  # pragma: no cover
            raise TypeError(f'Keyword arguments {kw!r} cannot be recognized.')

    def _set_meta(self, object_id: str, **meta):
        self._store[object_id] = meta

    def _get_meta(self,
                  object_id: str,
                  fields: List[str] = None) -> Dict[str, Any]:
        meta = self._store[object_id]
        if fields:
            return {k: meta[k] for k in fields}
        return meta

    def _del_meta(self, object_id: str):
        del self._store[object_id]
