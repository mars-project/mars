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

from ...operands import Fetch, FetchShuffle, FetchMixin
from ...serialize import DataTypeField
from ..operands import TensorOperandMixin


class TensorFetchMixin(TensorOperandMixin, FetchMixin):
    __slots__ = ()


class TensorFetch(Fetch, TensorFetchMixin):
    _dtype = DataTypeField('dtype')

    def __init__(self, dtype=None, to_fetch_key=None, sparse=False, **kw):
        super().__init__(
            _dtype=dtype, _to_fetch_key=to_fetch_key, _sparse=sparse, **kw)

    @property
    def dtype(self):
        return getattr(self, '_dtype', None)

    def _new_chunks(self, inputs, kws=None, **kw):
        if '_key' in kw and self._to_fetch_key is None:
            self._to_fetch_key = kw['_key']
        return super()._new_chunks(inputs, kws=kws, **kw)

    def _new_tileables(self, inputs, kws=None, **kw):
        if '_key' in kw and self._to_fetch_key is None:
            self._to_fetch_key = kw['_key']
        return super()._new_tileables(inputs, kws=kws, **kw)


class TensorFetchShuffle(FetchShuffle, TensorFetchMixin):
    _dtype = DataTypeField('dtype')

    def __init__(self, dtype=None, to_fetch_keys=None, to_fetch_idxes=None, **kw):
        super().__init__(
            _dtype=dtype, _to_fetch_keys=to_fetch_keys, _to_fetch_idxes=to_fetch_idxes, **kw)

    @property
    def dtype(self):
        return getattr(self, '_dtype', None)
