# -*- coding: utf-8 -*-
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

import functools
import inspect
import logging
import os
import sys
import types
from typing import Dict

import numpy as np
import pandas as pd
import pytest

try:
    from flaky import flaky as _raw_flaky
except ImportError:
    _raw_flaky = None
try:
    import mock
except ImportError:
    from unittest import mock
_mock = mock

from ..core.operand import OperandStage
from ..utils import lazy_import


cupy = lazy_import('cupy', globals=globals())
cudf = lazy_import('cudf', globals=globals())
ray = lazy_import('ray', globals=globals())

logger = logging.getLogger(__name__)


def flaky(o=None, *args, **kwargs):
    platform = kwargs.pop('platform', '')
    if _raw_flaky is None or not sys.platform.startswith(platform):
        if o is not None:
            return o

        def ident(x):
            return x

        return ident
    elif o is not None:
        return _raw_flaky(o, *args, **kwargs)
    else:
        return _raw_flaky(*args, **kwargs)


def patch_method(method, *args, **kwargs):
    if hasattr(method, '__qualname__'):
        return mock.patch(method.__module__ + '.' + method.__qualname__, *args, **kwargs)
    elif hasattr(method, 'im_class'):
        return mock.patch('.'.join(
            [method.im_class.__module__, method.im_class.__name__, method.__name__]),
            *args, **kwargs
        )
    else:
        return mock.patch(method.__module__ + '.' + method.__name__, *args, **kwargs)


def patch_cls(target_cls):
    def _wrapper(cls):
        class Super(cls.__bases__[0]):
            pass

        cls.__patch_super__ = Super

        target = target_cls.__module__ + '.' + target_cls.__qualname__
        for name, obj in cls.__dict__.items():
            if name.startswith('__') and name != '__init__':
                continue
            p = mock.patch(target + '.' + name, obj, create=True)
            original, local = p.get_original()
            setattr(cls.__patch_super__, name, original)
            p.start()

        return cls

    return _wrapper


def patch_super():
    back = inspect.currentframe().f_back
    if not back or '__class__' not in back.f_locals:
        raise RuntimeError('Calling super() in the incorrect context.')

    patch_super_cls = back.f_locals['__class__'].__patch_super__
    patch_self = back.f_locals.get('self')

    class _SuperAccessor:
        def __getattribute__(self, item):
            func = getattr(patch_super_cls, item)
            if func == mock.DEFAULT:
                raise AttributeError(f"super object has no attribute '{item}'")
            if patch_self:
                return types.MethodType(func, patch_self)
            return func

    return _SuperAccessor()


def print_entrance(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            print(f"Start to execute function {func} with args {args} and kwargs {kwargs}")
            result = func(*args, **kwargs)
            print(f"Finished executing function {func} with args {args} and kwargs {kwargs}")
            return result
        except NotImplementedError:
            return NotImplemented

    return wrapper


def print_async_entrance(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            print(f"Start to execute function {func} with args {args} and kwargs {kwargs}")
            result = await func(*args, **kwargs)
            print(f"Finished executing function {func} with args {args} and kwargs {kwargs}")
            return result
        except NotImplementedError:
            return NotImplemented

    return wrapper


def require_cupy(func):
    if pytest:
        func = pytest.mark.cuda(func)
    func = pytest.mark.skipif(cupy is None, reason='cupy not installed')(func)
    return func


def require_cudf(func):
    if pytest:
        func = pytest.mark.cuda(func)
    func = pytest.mark.skipif(cudf is None, reason='cudf not installed')(func)
    return func


def require_ray(func):
    if pytest:
        func = pytest.mark.ray(func)
    func = pytest.mark.skipif(ray is None, reason='ray not installed')(func)
    return func


def require_hadoop(func):
    if pytest:
        func = pytest.mark.hadoop(func)
    func = pytest.mark.skipif(not os.environ.get('WITH_HADOOP'),
                              reason='Only run when hadoop is installed')(func)
    return func


def assert_groupby_equal(left, right, sort_keys=False, sort_index=True, with_selection=False):
    if hasattr(left, 'groupby_obj'):
        left = left.groupby_obj
    if hasattr(right, 'groupby_obj'):
        right = right.groupby_obj

    if type(left) is not type(right):
        raise AssertionError(f'Type of groupby not consistent: {type(left)} != {type(right)}')

    left_selection = getattr(left, '_selection', None)
    right_selection = getattr(right, '_selection', None)
    if sort_keys:
        left = sorted(left, key=lambda p: p[0])
        right = sorted(right, key=lambda p: p[0])
    else:
        left, right = list(left), list(right)
    if sort_index:
        left = [(k, v.sort_index()) for k, v in left]
        right = [(k, v.sort_index()) for k, v in right]

    if len(left) != len(right):
        raise AssertionError(f'Count of groupby keys not consistent: {len(left)} != {len(right)}')

    left_keys = [p[0] for p in left]
    right_keys = [p[0] for p in right]
    if left_keys != right_keys:
        raise AssertionError(f'Group keys not consistent: {left_keys!r} != {right_keys!r}')
    for (left_key, left_frame), (right_key, right_frame) in zip(left, right):
        if with_selection:
            if left_selection and isinstance(left_frame, pd.DataFrame):
                left_frame = left_frame[left_selection]
            if right_selection and isinstance(right_frame, pd.DataFrame):
                right_frame = right_frame[right_selection]

        if isinstance(left_frame, pd.DataFrame):
            pd.testing.assert_frame_equal(left_frame, right_frame)
        else:
            pd.testing.assert_series_equal(left_frame, right_frame)


_check_options = dict()
_check_args = ['check_all', 'check_series_name', 'check_index_name', 'check_dtypes',
               'check_dtype', 'check_shape', 'check_nsplits',
               'check_index_value', 'check_columns_value']


class ObjectCheckMixin:
    _check_options: Dict

    @staticmethod
    def adapt_index_value(value):
        if hasattr(value, 'to_pandas'):
            return value.to_pandas()
        return value

    def assert_shape_consistent(self, expected_shape, real_shape):
        if not self._check_options['check_shape']:
            return

        if len(expected_shape) != len(real_shape):
            raise AssertionError('ndim in metadata %r is not consistent with real ndim %r'
                                 % (len(expected_shape), len(real_shape)))
        for e, r in zip(expected_shape, real_shape):
            if not np.isnan(e) and e != r:
                raise AssertionError('shape in metadata %r is not consistent with real shape %r'
                                     % (expected_shape, real_shape))

    @staticmethod
    def assert_dtype_consistent(expected_dtype, real_dtype):
        if isinstance(real_dtype, pd.DatetimeTZDtype):
            real_dtype = real_dtype.base
        if expected_dtype != real_dtype:
            if expected_dtype == np.dtype('O') and real_dtype.type is np.str_:
                # real dtype is string, this matches expectation
                return
            if expected_dtype is None:
                raise AssertionError('Expected dtype cannot be None')
            if isinstance(real_dtype, pd.CategoricalDtype) and isinstance(expected_dtype, pd.CategoricalDtype):
                return
            if not np.can_cast(real_dtype, expected_dtype) and not np.can_cast(expected_dtype, real_dtype):
                raise AssertionError('cannot cast between dtype of real dtype %r and dtype %r defined in metadata'
                                     % (real_dtype, expected_dtype))

    def assert_tensor_consistent(self, expected, real):
        from ..lib.sparse import SparseNDArray
        np_types = (np.generic, np.ndarray, pd.Timestamp, SparseNDArray)
        if cupy is not None:
            np_types += (cupy.ndarray,)

        if isinstance(real, tuple):
            # allow returning a batch of chunks for some operands
            real = real[0]
        if isinstance(real, (str, int, bool, float, complex)):
            real = np.array([real])[0]
        if not isinstance(real, np_types):
            raise AssertionError(f'Type of real value ({type(real)}) not one of {np_types!r}')
        if not hasattr(expected, 'dtype'):
            return
        if self._check_options['check_dtypes']:
            self.assert_dtype_consistent(expected.dtype, real.dtype)
        if self._check_options['check_shape']:
            self.assert_shape_consistent(expected.shape, real.shape)

    @classmethod
    def assert_index_value_consistent(cls, expected_index_value, real_index):
        if expected_index_value.has_value():
            expected_index = expected_index_value.to_pandas()
            try:
                pd.testing.assert_index_equal(expected_index, cls.adapt_index_value(real_index))
            except AssertionError as e:
                raise AssertionError(
                    f'Index of real value ({real_index}) not equal to ({expected_index})') from e

    def assert_dataframe_consistent(self, expected, real):
        dataframe_types = (pd.DataFrame,)
        if cudf is not None:
            dataframe_types += (cudf.DataFrame,)

        if isinstance(real, tuple):
            # allow returning a batch of chunks for some operands
            real = real[0]
        if not isinstance(real, dataframe_types):
            raise AssertionError(f'Type of real value ({type(real)}) not DataFrame')
        self.assert_shape_consistent(expected.shape, real.shape)
        if not np.isnan(expected.shape[1]) \
                and expected.dtypes is not None:
            # ignore check when columns length is nan or dtypes undefined
            pd.testing.assert_index_equal(expected.dtypes.index,
                                          self.adapt_index_value(real.dtypes.index))

            if self._check_options['check_dtypes']:
                try:
                    for expected_dtype, real_dtype in zip(expected.dtypes, real.dtypes):
                        self.assert_dtype_consistent(expected_dtype, real_dtype)
                except AssertionError:
                    raise AssertionError('dtypes in metadata %r cannot cast to real dtype %r'
                                         % (expected.dtypes, real.dtypes))

        if self._check_options['check_columns_value'] and not np.isnan(expected.shape[1]):
            self.assert_index_value_consistent(expected.columns_value, real.columns)
        if self._check_options['check_index_value'] and not np.isnan(expected.shape[0]):
            self.assert_index_value_consistent(expected.index_value, real.index)

    def assert_series_consistent(self, expected, real):
        series_types = (pd.Series,)
        if cudf is not None:
            series_types += (cudf.Series,)

        if not isinstance(real, series_types):
            raise AssertionError(f'Type of real value ({type(real)}) not Series')
        self.assert_shape_consistent(expected.shape, real.shape)

        if self._check_options['check_series_name'] and expected.name != real.name:
            raise AssertionError(f'series name in metadata {expected.name} '
                                 f'is not equal to real name {real.name}')

        self.assert_dtype_consistent(expected.dtype, real.dtype)
        if self._check_options['check_index_value']:
            self.assert_index_value_consistent(expected.index_value, real.index)

    def assert_groupby_consistent(self, expected, real):
        from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
        from ..lib.groupby_wrapper import GroupByWrapper
        from ..dataframe.core import DATAFRAME_GROUPBY_TYPE, SERIES_GROUPBY_TYPE
        from ..dataframe.core import DATAFRAME_GROUPBY_CHUNK_TYPE, SERIES_GROUPBY_CHUNK_TYPE

        df_groupby_types = (DataFrameGroupBy,)
        series_groupby_types = (SeriesGroupBy,)

        try:
            from cudf.core.groupby.groupby import DataFrameGroupBy as CUDataFrameGroupBy, \
                SeriesGroupBy as CUSeriesGroupBy
            df_groupby_types += (CUDataFrameGroupBy,)
            series_groupby_types += (CUSeriesGroupBy,)
        except ImportError:
            pass

        if isinstance(real, GroupByWrapper):
            real = real.groupby_obj

        if isinstance(expected, (DATAFRAME_GROUPBY_TYPE, DATAFRAME_GROUPBY_CHUNK_TYPE)) \
                and isinstance(real, df_groupby_types):
            selection = getattr(real, '_selection', None)
            if not selection:
                self.assert_dataframe_consistent(expected, real.obj)
            else:
                self.assert_dataframe_consistent(expected, real.obj[selection])
        elif isinstance(expected, (SERIES_GROUPBY_TYPE, SERIES_GROUPBY_CHUNK_TYPE)) \
                and isinstance(real, series_groupby_types):
            self.assert_series_consistent(expected, real.obj)
        else:
            raise AssertionError('GroupBy type not consistent. Expecting %r but receive %r'
                                 % (type(expected), type(real)))

    def assert_index_consistent(self, expected, real):
        index_types = (pd.Index,)
        if cudf is not None:
            index_types += (cudf.Index,)

        if not isinstance(real, index_types):
            raise AssertionError(f'Type of real value ({type(real)}) not Index')
        self.assert_shape_consistent(expected.shape, real.shape)

        if self._check_options['check_series_name'] and expected.name != real.name:
            raise AssertionError('series name in metadata %r is not equal to real name %r'
                                 % (expected.name, real.name))

        self.assert_dtype_consistent(expected.dtype, real.dtype)
        self.assert_index_value_consistent(expected.index_value, real)

    def assert_categorical_consistent(self, expected, real):
        if not isinstance(real, pd.Categorical):
            raise AssertionError(f'Type of real value ({type(real)}) not Categorical')
        self.assert_dtype_consistent(expected.dtype, real.dtype)
        self.assert_shape_consistent(expected.shape, real.shape)
        self.assert_index_value_consistent(expected.categories_value, real.categories)

    def assert_object_consistent(self, expected, real):
        from ..tensor.core import TENSOR_TYPE
        from ..dataframe.core import DATAFRAME_TYPE, SERIES_TYPE, GROUPBY_TYPE, \
            INDEX_TYPE, CATEGORICAL_TYPE

        from ..tensor.core import TENSOR_CHUNK_TYPE
        from ..dataframe.core import DATAFRAME_CHUNK_TYPE, SERIES_CHUNK_TYPE, \
            GROUPBY_CHUNK_TYPE, INDEX_CHUNK_TYPE, CATEGORICAL_CHUNK_TYPE

        op = getattr(expected, 'op', None)
        if op and getattr(op, 'stage', None) == OperandStage.map:
            return

        if isinstance(expected, (TENSOR_TYPE, TENSOR_CHUNK_TYPE)):
            self.assert_tensor_consistent(expected, real)
        elif isinstance(expected, (DATAFRAME_TYPE, DATAFRAME_CHUNK_TYPE)):
            self.assert_dataframe_consistent(expected, real)
        elif isinstance(expected, (SERIES_TYPE, SERIES_CHUNK_TYPE)):
            self.assert_series_consistent(expected, real)
        elif isinstance(expected, (GROUPBY_TYPE, GROUPBY_CHUNK_TYPE)):
            self.assert_groupby_consistent(expected, real)
        elif isinstance(expected, (INDEX_TYPE, INDEX_CHUNK_TYPE)):
            self.assert_index_consistent(expected, real)
        elif isinstance(expected, (CATEGORICAL_TYPE, CATEGORICAL_CHUNK_TYPE)):
            self.assert_categorical_consistent(expected, real)
