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

import sys
from collections.abc import Iterable

import cloudpickle
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy


class GroupByWrapper:
    def __init__(self, obj, groupby_obj=None, keys=None, axis=0, level=None, grouper=None,
                 exclusions=None, selection=None, as_index=True, sort=True,
                 group_keys=True, squeeze=False, observed=False, mutated=False,
                 grouper_cache=None):

        def fill_value(v, key):
            return v if v is not None or groupby_obj is None else getattr(groupby_obj, key)

        self.obj = obj
        self.keys = fill_value(keys, 'keys')
        self.axis = fill_value(axis, 'axis')
        self.level = fill_value(level, 'level')
        self.exclusions = fill_value(exclusions, 'exclusions')
        self.selection = selection
        self.as_index = fill_value(as_index, 'as_index')
        self.sort = fill_value(sort, 'sort')
        self.group_keys = fill_value(group_keys, 'group_keys')
        self.squeeze = fill_value(squeeze, 'squeeze')
        self.observed = fill_value(observed, 'observed')
        self.mutated = fill_value(mutated, 'mutated')

        if groupby_obj is None:
            if obj.ndim == 2:
                self.groupby_obj = DataFrameGroupBy(
                    obj, keys=keys, axis=axis, level=level, grouper=grouper, exclusions=exclusions,
                    as_index=as_index, group_keys=group_keys, squeeze=squeeze, observed=observed,
                    mutated=mutated)
            else:
                self.groupby_obj = SeriesGroupBy(
                    obj, keys=keys, axis=axis, level=level, grouper=grouper, exclusions=exclusions,
                    as_index=as_index, group_keys=group_keys, squeeze=squeeze, observed=observed,
                    mutated=mutated)
        else:
            self.groupby_obj = groupby_obj

        if grouper_cache:
            self.groupby_obj.grouper._cache = grouper_cache
        if selection:
            self.groupby_obj = self.groupby_obj[selection]

        self.is_frame = isinstance(self.groupby_obj, DataFrameGroupBy)

    def __getitem__(self, item):
        return GroupByWrapper(
            self.obj, keys=self.keys, axis=self.axis, level=self.level,
            grouper=self.groupby_obj.grouper, exclusions=self.exclusions, selection=item,
            as_index=self.as_index, sort=self.sort, group_keys=self.group_keys,
            squeeze=self.squeeze, observed=self.observed, mutated=self.mutated)

    def __getattr__(self, item):
        if item.startswith('_'):  # pragma: no cover
            return object.__getattribute__(self, item)
        if item in getattr(self.obj, 'columns', ()):
            return self.__getitem__(item)
        return getattr(self.groupby_obj, item)

    def __iter__(self):
        return self.groupby_obj.__iter__()

    def __sizeof__(self):
        return sys.getsizeof(self.obj) \
            + sys.getsizeof(getattr(self.groupby_obj.grouper, '_cache', None))

    @property
    def empty(self):
        return self.obj.empty

    @property
    def shape(self):
        shape = list(self.groupby_obj.obj.shape)
        if self.is_frame and self.selection:
            shape[1] = len(self.selection)
        return tuple(shape)

    def to_tuple(self, truncate=False, pickle_function=False):
        if self.selection and truncate:
            if isinstance(self.selection, Iterable) and not isinstance(self.selection, str):
                item_list = list(self.selection)
            else:
                item_list = [self.selection]
            item_set = set(item_list)

            if isinstance(self.keys, list):
                sel_keys = self.keys
            elif self.keys in self.obj.columns:
                sel_keys = [self.keys]
            else:
                sel_keys = []

            all_items = item_list + [k for k in sel_keys or () if k not in item_set]
            if set(all_items) == set(self.obj.columns):
                obj = self.obj
            else:
                obj = self.obj[all_items]
        else:
            obj = self.obj

        if pickle_function and callable(self.keys):
            keys = cloudpickle.dumps(self.keys)
        else:
            keys = self.keys

        return obj, keys, self.axis, self.level, self.exclusions, self.selection, \
            self.as_index, self.sort, self.group_keys, self.squeeze, self.observed, \
            self.mutated, getattr(self.groupby_obj.grouper, '_cache', dict())

    @classmethod
    def from_tuple(cls, tp):
        obj, keys, axis, level, exclusions, selection, as_index, sort, group_keys, squeeze, \
            observed, mutated, grouper_cache = tp

        if isinstance(keys, (bytes, bytearray)):
            keys = cloudpickle.loads(keys)

        return cls(obj, keys=keys, axis=axis, level=level, exclusions=exclusions, selection=selection,
                   as_index=as_index, sort=sort, group_keys=group_keys, squeeze=squeeze, observed=observed,
                   mutated=mutated, grouper_cache=grouper_cache)


def wrapped_groupby(obj, by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True,
                    squeeze=False, observed=False):
    groupby_obj = obj.groupby(by=by, axis=axis, level=level, as_index=as_index, sort=sort,
                              group_keys=group_keys, squeeze=squeeze, observed=observed)
    return GroupByWrapper(obj, groupby_obj=groupby_obj)
