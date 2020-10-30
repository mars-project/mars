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

from ...core import OutputType, register_fetch_class
from ...serialize.core import TupleField, ValueType
from ...operands import Fetch, FetchShuffle, FetchMixin
from ...utils import on_serialize_shape, on_deserialize_shape
from ..operands import DataFrameOperandMixin


class DataFrameFetchMixin(DataFrameOperandMixin, FetchMixin):
    __slots__ = ()


class DataFrameFetch(Fetch, DataFrameFetchMixin):
    # required fields
    _shape = TupleField('shape', ValueType.int64,
                        on_serialize=on_serialize_shape, on_deserialize=on_deserialize_shape)

    def __init__(self, to_fetch_key=None, sparse=False, output_types=None, **kw):
        super().__init__(
            _to_fetch_key=to_fetch_key, _sparse=sparse, _output_types=output_types, **kw)

    def _new_chunks(self, inputs, kws=None, **kw):
        if '_key' in kw and self._to_fetch_key is None:
            self._to_fetch_key = kw['_key']
        if '_shape' in kw and self._shape is None:
            self._shape = kw['_shape']
        return super()._new_chunks(inputs, kws=kws, **kw)

    def _new_tileables(self, inputs, kws=None, **kw):
        if '_key' in kw and self._to_fetch_key is None:
            self._to_fetch_key = kw['_key']
        return super()._new_tileables(inputs, kws=kws, **kw)


class DataFrameFetchShuffle(FetchShuffle, DataFrameFetchMixin):
    # required fields
    _shape = TupleField('shape', ValueType.int64,
                        on_serialize=on_serialize_shape, on_deserialize=on_deserialize_shape)

    def __init__(self, to_fetch_keys=None, to_fetch_idxes=None, output_types=None, **kw):
        super().__init__(
            _to_fetch_keys=to_fetch_keys, _to_fetch_idxes=to_fetch_idxes,
            _output_types=output_types, **kw)


register_fetch_class(OutputType.dataframe, DataFrameFetch, DataFrameFetchShuffle)
register_fetch_class(OutputType.dataframe_groupby, DataFrameFetch, DataFrameFetchShuffle)
register_fetch_class(OutputType.series, DataFrameFetch, DataFrameFetchShuffle)
register_fetch_class(OutputType.series_groupby, DataFrameFetch, DataFrameFetchShuffle)
register_fetch_class(OutputType.index, DataFrameFetch, DataFrameFetchShuffle)
register_fetch_class(OutputType.categorical, DataFrameFetch, DataFrameFetchShuffle)
