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

import itertools
from collections.abc import Sequence
from typing import List

import numpy as np
try:
    from scipy.sparse.base import spmatrix
except ImportError:  # pragma: no cover
    spmatrix = None

from ... import opcodes as OperandDef
from ... import tensor as mt
from ...core import ENTITY_TYPE, TILEABLE_TYPE, recursive_tile
from ...core.context import get_context
from ...serialization.serializables import AnyField, ListField
from ...tensor.core import TensorOrder
from ...typing import TileableType
from ..operands import LearnOperand, LearnOperandMixin, OutputType
from ..utils import assert_all_finite
from .validation import check_array


def _unique_multiclass(y):
    if hasattr(y, '__array__') or hasattr(y, '__mars_tensor__'):
        return mt.unique(mt.asarray(y))
    else:
        return set(y)


def _unique_indicator(y):
    return mt.arange(
        check_array(y, accept_sparse=True).shape[1]
    )


_FN_UNIQUE_LABELS = {
    'binary': _unique_multiclass,
    'multiclass': _unique_multiclass,
    'multilabel-indicator': _unique_indicator,
}


class UniqueLabels(LearnOperand, LearnOperandMixin):
    _op_type_ = OperandDef.UNIQUE_LABELS

    ys = ListField('ys')

    def __call__(self, ys: List[TileableType]):
        self._output_types = [OutputType.tensor]
        inputs = [y for y in ys if isinstance(y, TILEABLE_TYPE)]
        return self.new_tileable(inputs, shape=(np.nan,),
                                 dtype=mt.tensor(ys[0]).dtype,
                                 order=TensorOrder.C_ORDER)

    @classmethod
    def tile(cls, op: "UniqueLabels"):
        ys = op.ys
        ctx = get_context()

        target_types = yield from recursive_tile(
            [type_of_target(x) for x in ys])
        # yield chunks of target_types for execution
        chunks = list(itertools.chain(*(t.chunks for t in target_types)))
        yield chunks

        ys_types = set([it.item() for it in
                        ctx.get_chunks_result([c.key for c in chunks])])
        if ys_types == {"binary", "multiclass"}:
            ys_types = {"multiclass"}

        if len(ys_types) > 1:
            raise ValueError("Mix type of y not allowed, got types %s" % ys_types)

        label_type = ys_types.pop()

        # Check consistency for the indicator format
        if label_type == "multilabel-indicator":
            check_arrays = []
            chunks = []
            for y in ys:
                arr = yield from recursive_tile(
                    check_array(y, accept_sparse=True))
                check_arrays.append(arr)
                chunks.extend(arr.chunks)
            yield check_arrays + chunks
            if len(set(arr.shape[1] for arr in check_arrays)) > 1:
                raise ValueError("Multi-label binary indicator input with "
                                 "different numbers of labels")

        # Get the unique set of labels
        _unique_labels = _FN_UNIQUE_LABELS.get(label_type, None)
        if not _unique_labels:
            raise ValueError("Unknown label type: %s" % repr(ys))

        labels = [_unique_labels(y) for y in ys]
        labels_chunks = []
        ys_labels = set()
        for label in labels:
            if isinstance(label, ENTITY_TYPE):
                label = yield from recursive_tile(label)
                labels_chunks.extend(label.chunks)
            else:
                ys_labels.update(label)
        yield labels_chunks
        ys_labels.update(itertools.chain.from_iterable(
            ctx.get_chunks_result([c.key for c in labels_chunks])))

        # Check that we don't mix string type with number type
        if (len(set(isinstance(label, str) for label in ys_labels)) > 1):
            raise ValueError("Mix of label input types (string and number)")

        return (yield from recursive_tile(mt.array(sorted(ys_labels))))


def unique_labels(*ys):
    """
    Extract an ordered array of unique labels.

    We don't allow:
        - mix of multilabel and multiclass (single label) targets
        - mix of label indicator matrix and anything else,
          because there are no explicit labels)
        - mix of label indicator matrices of different sizes
        - mix of string and integer labels

    At the moment, we also don't allow "multiclass-multioutput" input type.

    Parameters
    ----------
    *ys : array-likes

    Returns
    -------
    out : ndarray of shape (n_unique_labels,)
        An ordered array of unique labels.

    Examples
    --------
    >>> from mars.learn.utils.multiclass import unique_labels
    >>> unique_labels([3, 5, 5, 5, 7, 7]).execute()
    array([3, 5, 7])
    >>> unique_labels([1, 2, 3, 4], [2, 2, 3, 4]).execute()
    array([1, 2, 3, 4])
    >>> unique_labels([1, 2, 10], [5, 11]).execute()
    array([ 1,  2,  5, 10, 11])
    """
    if not ys:
        raise ValueError('No argument has been passed.')

    ys = list(ys)
    op = UniqueLabels(ys=ys)
    return op(ys)


class IsMultilabel(LearnOperand, LearnOperandMixin):
    _op_type_ = OperandDef.IS_MULTILABEL

    y = AnyField('y')

    def __call__(self, y):
        self._output_types = [OutputType.tensor]
        inputs = [y] if isinstance(y, ENTITY_TYPE) else []
        return self.new_tileable(inputs, shape=(), dtype=np.dtype(bool),
                                 order=TensorOrder.C_ORDER)

    @classmethod
    def _tile(cls, op: "IsMultilabel"):
        y = op.y
        ctx = get_context()

        y = yield from recursive_tile(mt.tensor(y))
        if any(np.isnan(s) for s in y.shape):
            yield y.chunks + [y]

        if not (hasattr(y, "shape") and y.ndim == 2 and y.shape[1] > 1):
            return False

        labels = yield from recursive_tile(mt.unique(y))
        yield labels.chunks + [labels]

        if len(labels) < 3:
            if y.dtype.kind in 'biu':
                return True
            if y.dtype.kind == 'f':
                is_integral_float = yield from recursive_tile(
                    mt.all(mt.equal(y.astype(int), y)))
                yield is_integral_float.chunks
                is_integral_float = ctx.get_chunks_result(
                    [is_integral_float.chunks[0].key])[0]
                if is_integral_float:
                    return True

        return False

    @classmethod
    def tile(cls, op: "IsMultilabel"):
        result = yield from cls._tile(op)
        return (yield from recursive_tile(mt.array(result)))


def is_multilabel(y):
    """
    Check if ``y`` is in a multilabel format.

    Parameters
    ----------
    y : numpy array of shape [n_samples]
        Target values.

    Returns
    -------
    out : bool,
        Return ``True``, if ``y`` is in a multilabel format, else ```False``.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> from mars.learn.utils.multiclass import is_multilabel
    >>> is_multilabel([0, 1, 0, 1]).execute()
    False
    >>> is_multilabel([[1], [0, 2], []]).execute()
    False
    >>> is_multilabel(mt.array([[1, 0], [0, 0]])).execute()
    True
    >>> is_multilabel(mt.array([[1], [0], [0]])).execute()
    False
    >>> is_multilabel(mt.array([[1, 0, 0]])).execute()
    True
    """
    if not isinstance(y, ENTITY_TYPE):
        if hasattr(y, '__array__') or isinstance(y, Sequence):
            y = np.asarray(y)
        if hasattr(y, 'shape'):
            yt = y = mt.asarray(y)
        else:
            yt = None
    else:
        yt = y = mt.tensor(y)

    op = IsMultilabel(y=y)
    return op(yt)


class TypeOfTarget(LearnOperand, LearnOperandMixin):
    _op_type_ = OperandDef.TYPE_OF_TARGET

    y = AnyField('y')

    def __call__(self, y: TileableType):
        self._output_types = [OutputType.tensor]
        inputs = [y] if isinstance(y, ENTITY_TYPE) else []
        return self.new_tileable(inputs, shape=(), order=TensorOrder.C_ORDER,
                                 dtype=np.dtype(object))

    @classmethod
    def _tile(cls, op: "TypeOfTarget"):
        y = op.y
        ctx = get_context()

        multilabel = yield from recursive_tile(is_multilabel(y))
        yield multilabel.chunks
        multilabel = ctx.get_chunks_result([multilabel.chunks[0].key])[0]
        if multilabel:
            return 'multilabel-indicator'

        y = yield from recursive_tile(mt.tensor(y))
        executed = False
        if any(np.isnan(s) for s in y.shape):
            # trigger execution
            yield y.chunks + [y]
            executed = True

        # Invalid inputs
        if y.ndim > 2:
            return 'unknown'
        if y.dtype == object and len(y):
            # [[[1, 2]]] or [obj_1] and not ["label_1"]
            if isinstance(y, ENTITY_TYPE):
                if not executed:
                    yield y.chunks + [y]
                first_val = ctx.get_chunks_result([y.chunks[0].key])[0].flat[0]
            else:
                first_val = y.flat[0]
            if not isinstance(first_val, str):
                return 'unknown'

        if y.ndim == 2 and y.shape[1] == 0:
            return 'unknown'  # [[]]

        if y.ndim == 2 and y.shape[1] > 1:
            suffix = "-multioutput"  # [[1, 2], [1, 2]]
        else:
            suffix = ""  # [1, 2, 3] or [[1], [2], [3]]

        if y.dtype.kind == 'f':
            # check float and contains non-integer float values
            contain_float_values = yield from recursive_tile(
                mt.any(y != y.astype(int)))
            yield contain_float_values.chunks
            contain_float_values = ctx.get_chunks_result(
                [contain_float_values.chunks[0].key])[0]
            # [.1, .2, 3] or [[.1, .2, 3]] or [[1., .2]] and not [1., 2., 3.]
            if contain_float_values:
                yield from recursive_tile(assert_all_finite(y))
                return 'continuous' + suffix

        unique_y = yield from recursive_tile(mt.unique(y))
        yield unique_y.chunks + [unique_y]
        if (len(unique_y) > 2) or (y.ndim >= 2 and len(y[0]) > 1):
            return 'multiclass' + suffix  # [1, 2, 3] or [[1., 2., 3]] or [[1, 2]]
        else:
            return 'binary'  # [1, 2] or [["a"], ["b"]]

    @classmethod
    def tile(cls, op: "TypeOfTarget"):
        result = yield from cls._tile(op)
        return (yield from recursive_tile(mt.array(result)))


def type_of_target(y):
    """
    Determine the type of data indicated by the target.

    Note that this type is the most specific type that can be inferred.
    For example:

        * ``binary`` is more specific but compatible with ``multiclass``.
        * ``multiclass`` of integers is more specific but compatible with
          ``continuous``.
        * ``multilabel-indicator`` is more specific but compatible with
          ``multiclass-multioutput``.

    Parameters
    ----------
    y : array-like

    Returns
    -------
    target_type : string
        One of:

        * 'continuous': `y` is an array-like of floats that are not all
          integers, and is 1d or a column vector.
        * 'continuous-multioutput': `y` is a 2d tensor of floats that are
          not all integers, and both dimensions are of size > 1.
        * 'binary': `y` contains <= 2 discrete values and is 1d or a column
          vector.
        * 'multiclass': `y` contains more than two discrete values, is not a
          sequence of sequences, and is 1d or a column vector.
        * 'multiclass-multioutput': `y` is a 2d tensor that contains more
          than two discrete values, is not a sequence of sequences, and both
          dimensions are of size > 1.
        * 'multilabel-indicator': `y` is a label indicator matrix, a tensor
          of two dimensions with at least two columns, and at most 2 unique
          values.
        * 'unknown': `y` is array-like but none of the above, such as a 3d
          tensor, sequence of sequences, or a tensor of non-sequence objects.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> from mars.learn.utils.multiclass import type_of_target
    >>> type_of_target([0.1, 0.6]).execute()
    'continuous'
    >>> type_of_target([1, -1, -1, 1]).execute()
    'binary'
    >>> type_of_target(['a', 'b', 'a']).execute()
    'binary'
    >>> type_of_target([1.0, 2.0]).execute()
    'binary'
    >>> type_of_target([1, 0, 2]).execute()
    'multiclass'
    >>> type_of_target([1.0, 0.0, 3.0]).execute()
    'multiclass'
    >>> type_of_target(['a', 'b', 'c']).execute()
    'multiclass'
    >>> type_of_target(mt.array([[1, 2], [3, 1]])).execute()
    'multiclass-multioutput'
    >>> type_of_target([[1, 2]]).execute()
    'multiclass-multioutput'
    >>> type_of_target(mt.array([[1.5, 2.0], [3.0, 1.6]])).execute()
    'continuous-multioutput'
    >>> type_of_target(mt.array([[0, 1], [1, 1]])).execute()
    'multilabel-indicator'
    """
    valid_types = (Sequence, spmatrix) if spmatrix is not None else (Sequence,)
    valid = ((isinstance(y, valid_types) or
              hasattr(y, '__array__') or hasattr(y, '__mars_tensor__'))
             and not isinstance(y, str))

    if not valid:
        raise ValueError(f'Expected array-like (array or non-string sequence), got {y}')

    sparse_pandas = (y.__class__.__name__ in ['SparseSeries', 'SparseArray'])
    if sparse_pandas:  # pragma: no cover
        raise ValueError("y cannot be class 'SparseSeries' or 'SparseArray'")

    if isinstance(y, ENTITY_TYPE):
        y = mt.tensor(y)

    op = TypeOfTarget(y=y)
    return op(y)


def check_classification_targets(y):
    """
    Ensure that target y is of a non-regression type.

    Only the following target types (as defined in type_of_target) are allowed:
        'binary', 'multiclass', 'multiclass-multioutput',
        'multilabel-indicator', 'multilabel-sequences'

    Parameters
    ----------
    y : array-like
    """
    y_type = type_of_target(y)

    def check(t):
        if t not in ['binary', 'multiclass', 'multiclass-multioutput',
                          'multilabel-indicator', 'multilabel-sequences']:
            raise ValueError("Unknown label type: %r" % y_type)
        return t

    y_type = y_type.map_chunk(check, dtype=y_type.dtype)
    return y_type
