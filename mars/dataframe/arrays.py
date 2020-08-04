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

import itertools
import operator
from copy import copy as copy_obj
from numbers import Integral
from typing import Type, Sequence

import numpy as np
import pandas as pd
from pandas._libs import lib
from pandas.api.types import pandas_dtype, is_scalar, is_array_like, is_string_dtype
from pandas.api.extensions import ExtensionArray, ExtensionDtype, register_extension_dtype
from pandas.core import ops
from pandas.core.algorithms import take
from pandas.compat import set_function_name
try:
    from pandas.arrays import StringArray as StringArrayBase
except ImportError:  # for pandas < 1.0
    StringArrayBase = ExtensionArray
try:
    from pandas.api.indexers import check_array_indexer
except ImportError:  # for pandas < 1.0
    check_array_indexer = lambda array, indexer: indexer
try:
    from pandas.api.extensions import no_default
except ImportError:
    no_default = object()
try:
    from pandas.core.ops import ARITHMETIC_BINOPS
except ImportError:
    ARITHMETIC_BINOPS = {
        "add",
        "sub",
        "mul",
        "pow",
        "mod",
        "floordiv",
        "truediv",
        "divmod",
        "radd",
        "rsub",
        "rmul",
        "rpow",
        "rmod",
        "rfloordiv",
        "rtruediv",
        "rdivmod",
    }


try:
    import pyarrow as pa
    pa_null = pa.NULL
except ImportError:  # pragma: no cover
    pa = None
    pa_null = None


@register_extension_dtype
class ArrowStringDtype(ExtensionDtype):
    """
    Extension dtype for arrow string data.

    .. warning::

       ArrowStringDtype is considered experimental. The implementation and
       parts of the API may change without warning.

       In particular, ArrowStringDtype.na_value may change to no longer be
       ``numpy.nan``.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> import mars.dataframe as md
    >>> md.ArrowStringDtype()
    ArrowStringDtype
    """

    type = str
    kind = "U"
    name = "arrow_string"
    na_value = pa_null

    @classmethod
    def construct_from_string(cls, string):
        if string == cls.name:
            return cls()
        else:
            raise TypeError("Cannot construct a '{}' "
                            "from '{}'".format(cls, string))

    @classmethod
    def construct_array_type(cls) -> "Type[ArrowStringArray]":
        return ArrowStringArray

    def __from_arrow__(self, array):
        return ArrowStringArray(array)


class ArrowStringArray(StringArrayBase):
    def __init__(self, values, copy=False):
        if isinstance(values, (pd.Index, pd.Series)):
            # for pandas Index and Series,
            # convert to PandasArray
            values = values.array

        if isinstance(values, type(self)):
            arrow_array = values._arrow_array
        elif isinstance(values, ExtensionArray):
            # if come from pandas object like index,
            # convert to pandas StringArray first,
            # validation will be done in construct
            arrow_array = pa.chunked_array([pa.array(values, from_pandas=True)])
        elif isinstance(values, pa.ChunkedArray):
            arrow_array = values
        elif isinstance(values, pa.StringArray):
            arrow_array = pa.chunked_array([values])
        else:
            arrow_array = pa.chunked_array([pa.array(values, type=pa.string())])

        if copy:
            arrow_array = copy_obj(arrow_array)

        self._arrow_array = arrow_array
        self._dtype = ArrowStringDtype()

        # for test purpose
        self._force_use_pandas = False

    @classmethod
    def from_scalars(cls, values):
        values = np.asarray(values)
        if not is_string_dtype(values):
            values = np.asarray(pd.Series(values).map(
                lambda x: x if pd.isna(x) else str(x)))
        arrow_array = pa.chunked_array([pa.array(values)])
        return cls(arrow_array)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        if not hasattr(scalars, 'dtype'):
            scalars = np.asarray(scalars)
        if isinstance(scalars, cls):
            if copy:
                scalars = scalars.copy()
            return scalars
        arrow_array = pa.chunked_array([pa.array(scalars).cast(pa.string())])
        return cls(arrow_array, copy=copy)

    @classmethod
    def _from_sequence_of_strings(cls, strings, dtype=None, copy=False):
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values)

    def __repr__(self):
        return "{}({})".format(type(self).__name__,
                               repr(self._arrow_array))

    @property
    def dtype(self):
        return self._dtype

    @property
    def nbytes(self) -> int:
        return sum(x.size
                   for chunk in self._arrow_array.chunks
                   for x in chunk.buffers()
                   if x is not None)

    @staticmethod
    def _can_process_slice_via_arrow(slc):
        if not isinstance(slc, slice):
            return False
        if slc.step is not None and slc.step != 1:
            return False
        if slc.start is not None and \
                not isinstance(slc.start, Integral):  # pragma: no cover
            return False
        if slc.stop is not None and \
                not isinstance(slc.stop, Integral):  # pragma: no cover
            return False
        return True

    def _values_for_factorize(self):
        arr = self.to_numpy()
        mask = self.isna()
        arr[mask] = -1
        return arr, -1

    def _values_for_argsort(self):
        return self.to_numpy()

    @staticmethod
    def _process_pos(pos, length, is_start):
        if pos is None:
            return 0 if is_start else length
        return pos + length if pos < 0 else pos

    def __getitem__(self, item):
        has_take = hasattr(self._arrow_array, 'take')
        if not self._force_use_pandas and has_take:
            if pd.api.types.is_scalar(item):
                item = item + len(self) if item < 0 else item
                return self._arrow_array.take([item]).to_pandas()[0]
            elif self._can_process_slice_via_arrow(item):
                length = len(self)
                start, stop = item.start, item.stop
                start = self._process_pos(start, length, True)
                stop = self._process_pos(stop, length, False)
                return ArrowStringArray(
                    self._arrow_array.slice(offset=start,
                                            length=stop - start))
            elif hasattr(item, 'dtype') and np.issubdtype(item.dtype, np.bool_):
                return ArrowStringArray(self._arrow_array.filter(
                    pa.array(item, from_pandas=True)))
            elif hasattr(item, 'dtype'):
                length = len(self)
                item = np.where(item < 0, item + length, item)
                return ArrowStringArray(self._arrow_array.take(item))

        array = np.asarray(self._arrow_array.to_pandas())
        return ArrowStringArray(array[item])

    def __setitem__(self, key, value):
        if isinstance(value, (pd.Index, pd.Series)):
            value = value.to_numpy()
        if isinstance(value, type(self)):
            value = value.to_numpy()

        key = check_array_indexer(self, key)
        scalar_key = is_scalar(key)
        scalar_value = is_scalar(value)
        if scalar_key and not scalar_value:
            raise ValueError("setting an array element with a sequence.")

        # validate new items
        if scalar_value:
            if pd.isna(value):
                value = None
            elif not isinstance(value, str):
                raise ValueError(
                    "Cannot set non-string value '{}' into a StringArray.".format(value)
                )
        else:
            if not is_array_like(value):
                value = np.asarray(value, dtype=object)
            if len(value) and not lib.is_string_array(value, skipna=True):
                raise ValueError("Must provide strings.")

        string_array = np.asarray(self._arrow_array.to_pandas())
        string_array[key] = value
        self._arrow_array = pa.chunked_array([pa.array(string_array)])

    def __len__(self):
        return len(self._arrow_array)

    def __array__(self, dtype=None):
        return self.to_numpy(dtype=dtype)

    def to_numpy(self, dtype=None, copy=False, na_value=no_default):
        array = np.asarray(self._arrow_array.to_pandas())
        if copy or na_value is not no_default:
            array = array.copy()
        if na_value is not no_default:
            array[self.isna()] = na_value
        return array

    def fillna(self, value=None, method=None, limit=None):
        chunks = []
        for chunk_array in self._arrow_array.chunks:
            array = chunk_array.to_pandas()
            result_array = array.fillna(value=value, method=method,
                                        limit=limit)
            chunks.append(pa.array(result_array, from_pandas=True))
        return ArrowStringArray(pa.chunked_array(chunks))

    def astype(self, dtype, copy=True):
        dtype = pandas_dtype(dtype)
        if isinstance(dtype, ArrowStringDtype):
            if copy:
                return self.copy()
            return self

        # try to slice 1 record to get the result dtype
        test_array = self._arrow_array.slice(0, 1).to_pandas()
        test_result_array = test_array.astype(dtype).array

        result_array = \
            type(test_result_array)(
                np.full(self.shape, test_result_array.dtype.na_value,
                        dtype=np.asarray(test_result_array).dtype))

        start = 0
        # use chunks to do astype
        for chunk_array in self._arrow_array.chunks:
            result_array[start: start + len(chunk_array)] = \
                chunk_array.to_pandas().astype(dtype).array
            start += len(chunk_array)
        return result_array

    def isna(self):
        if not self._force_use_pandas and hasattr(self._arrow_array, 'is_null'):
            return self._arrow_array.is_null().to_pandas().to_numpy()
        else:
            return pd.isna(self._arrow_array.to_pandas()).to_numpy()

    def take(self, indices, allow_fill=False, fill_value=None):
        if allow_fill is False:
            return ArrowStringArray(self[indices])

        string_array = self._arrow_array.to_pandas().to_numpy()

        replace = False
        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value
            replace = True

        result = take(string_array, indices, fill_value=fill_value,
                      allow_fill=allow_fill)
        if replace:
            # pyarrow cannot recognize pa.NULL
            result[result == self.dtype.na_value] = None
        return ArrowStringArray(result)

    def copy(self):
        return type(self)(copy_obj(self._arrow_array))

    @classmethod
    def _concat_same_type(
            cls, to_concat: Sequence["ArrowStringArray"]) -> "ArrowStringArray":
        chunks = list(itertools.chain.from_iterable(
            x._arrow_array.chunks for x in to_concat))
        if len(chunks) == 0:
            chunks = [pa.array([], type=pa.string())]
        return cls(pa.chunked_array(chunks))

    def value_counts(self, dropna=False):
        string_array = self._arrow_array.to_pandas()
        return ArrowStringArray(string_array.value_counts(dropna=dropna))

    def any(self, axis=0, out=None):
        return self._arrow_array.to_pandas().any(axis=axis, out=out)

    def all(self, axis=0, out=None):
        return self._arrow_array.to_pandas().all(axis=axis, out=out)

    # Overrride parent because we have different return types.
    @classmethod
    def _create_arithmetic_method(cls, op):
        # Note: this handles both arithmetic and comparison methods.
        def method(self, other):
            is_arithmetic = \
                True if op.__name__ in ARITHMETIC_BINOPS else False

            is_other_array = False
            if not is_scalar(other):
                is_other_array = True
                other = np.asarray(other)

            self_is_na = self.isna()
            other_is_na = pd.isna(other)
            mask = self_is_na | other_is_na

            chunks = []
            mask_chunks = []
            start = 0
            for chunk_array in self._arrow_array.chunks:
                chunk_array = np.asarray(chunk_array.to_pandas())
                end = start + len(chunk_array)
                chunk_mask = mask[start: end]
                chunk_valid = ~chunk_mask

                if is_arithmetic:
                    result = np.empty(chunk_array.shape, dtype=object)
                else:
                    result = np.zeros(chunk_array.shape, dtype=bool)

                chunk_other = other
                if is_other_array:
                    chunk_other = other[start: end]
                    chunk_other = chunk_other[chunk_valid]

                # calculate only for both not None
                result[chunk_valid] = op(chunk_array[chunk_valid],
                                         chunk_other)

                if is_arithmetic:
                    chunks.append(pa.array(result, type=pa.string(),
                                           from_pandas=True))
                else:
                    chunks.append(result)
                    mask_chunks.append(chunk_mask)

            if is_arithmetic:
                return ArrowStringArray(pa.chunked_array(chunks))
            else:
                try:
                    return pd.arrays.BooleanArray(np.concatenate(chunks),
                                                  np.concatenate(mask_chunks))
                except AttributeError:
                    return np.concatenate(chunks)

        return set_function_name(method, "__{}__".format(op.__name__), cls)

    @classmethod
    def _add_arithmetic_ops(cls):
        cls.__add__ = cls._create_arithmetic_method(operator.add)
        cls.__radd__ = cls._create_arithmetic_method(ops.radd)

        cls.__mul__ = cls._create_arithmetic_method(operator.mul)
        cls.__rmul__ = cls._create_arithmetic_method(ops.rmul)

    @classmethod
    def _add_comparison_ops(cls):
        cls.__eq__ = cls._create_comparison_method(operator.eq)
        cls.__ne__ = cls._create_comparison_method(operator.ne)
        cls.__lt__ = cls._create_comparison_method(operator.lt)
        cls.__gt__ = cls._create_comparison_method(operator.gt)
        cls.__le__ = cls._create_comparison_method(operator.le)
        cls.__ge__ = cls._create_comparison_method(operator.ge)

    _create_comparison_method = _create_arithmetic_method

    def __mars_tokenize__(self):
        return [memoryview(x) for chunk in self._arrow_array.chunks
                for x in chunk.buffers()
                if x is not None]


ArrowStringArray._add_arithmetic_ops()
ArrowStringArray._add_comparison_ops()
