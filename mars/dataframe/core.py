#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

from ..core import Entity, TilesableData
from ..serialize import Serializable, DataTypeField, AnyField, ListField


class IndexValue(Serializable):
    __slots__ = ()

    class Index(Serializable):
        _name = AnyField('name')
        _data = ListField('data')
        _dtype = DataTypeField('dtype')

    class RangeIndex(Serializable):
        _name = AnyField('name')
        _slice = AnyField('slice')


class IndexData(TilesableData):
    __slots__ = ()

    # optional field
    _dtype = DataTypeField('dtype')


class Index(Entity):
    _allow_data_type_ = (IndexData,)


class SeriesData(TilesableData):
    pass


class Series(Entity):
    _allow_data_type_ = (SeriesData,)


class DataFrameData(TilesableData):
    pass


class DataFrame(Entity):
    _allow_data_type_ = (DataFrameData,)
