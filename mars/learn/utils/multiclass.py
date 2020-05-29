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

from collections.abc import Sequence

import numpy as np
try:
    from scipy.sparse.base import spmatrix
except ImportError:  # pragma: no cover
    spmatrix = None

from ... import opcodes as OperandDef
from ... import tensor as mt
from ...core import Base, Entity
from ...serialize import KeyField, BoolField, TupleField, DataTypeField, AnyField, ListField
from ...tensor.operands import TensorOrder
from ...tiles import TilesError
from ...utils import recursive_tile
from ..operands import LearnOperand, LearnOperandMixin, OutputType
from ..utils import assert_all_finite


class IsMultilabel(LearnOperand, LearnOperandMixin):
    _op_type_ = OperandDef.IS_MULTILABEL

    _y = AnyField('y')
    _unique_y = KeyField('unique_y')
    # for chunk
    _is_y_sparse = BoolField('is_y_sparse')

    def __init__(self, y=None, unique_y=None, is_y_sparse=None, **kw):
        super().__init__(_y=y, _unique_y=unique_y,
                         _is_y_sparse=is_y_sparse, **kw)
        self._output_types = [OutputType.tensor]

    @property
    def y(self):
        return self._y

    @property
    def unique_y(self):
        return self._unique_y

    @property
    def is_y_sparse(self):
        return self._is_y_sparse

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if isinstance(self._y, (Base, Entity)):
            self._y = self._inputs[0]
        if self._unique_y is not None:
            self._unique_y = self._inputs[-1]

    def __call__(self, y, y_unique=None):
        inputs = [y] if isinstance(y, (Base, Entity)) else []
        if y_unique is not None:
            inputs.append(y_unique)
        return self.new_tileable(inputs, shape=(), dtype=np.dtype(bool),
                                 order=TensorOrder.C_ORDER)

    @classmethod
    def tile(cls, op):
        y = op.y
        out = op.outputs[0]

        if not (hasattr(y, 'shape') and y.ndim == 2 and y.shape[1] > 1):
            result = mt.array(False)._inplace_tile()
            return [result]
        else:
            unique_y = op.unique_y
            assert len(unique_y.chunks) == 1
            unique_y_chunk = unique_y.chunks[0]
            chunk_op = IsMultilabel(unique_y=unique_y_chunk,
                                    is_y_sparse=y.issparse())
            chunk = chunk_op.new_chunk([unique_y_chunk], dtype=out.dtype,
                                       order=out.order, index=(0,),
                                       shape=())

            new_op = op.copy()
            params = out.params
            params['nsplits'] = ()
            params['chunks'] = [chunk]
            return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    def execute(cls, ctx, op):
        unique_y = ctx[op.unique_y.key]

        if op.is_y_sparse:
            # sparse
            result = (unique_y.size in (0, 1) and
                      (unique_y.dtype.kind in 'biu' or  # bool, int, uint
                       _is_integral_float(unique_y)))
        else:
            # dense
            labels = unique_y
            result = len(labels) < 3 and (unique_y.dtype.kind in 'biu' or  # bool, int, uint
                                          _is_integral_float(labels))

        ctx[op.outputs[0].key] = result


def _is_integral_float(y):
    return y.dtype.kind == 'f' and np.all(y.astype(int) == y)


def is_multilabel(y):
    """ Check if ``y`` is in a multilabel format.

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
    if not isinstance(y, (Base, Entity)):
        if hasattr(y, '__array__') or isinstance(y, Sequence):
            y = np.asarray(y)
        if hasattr(y, 'shape'):
            yt = y = mt.asarray(y)
        else:
            yt = None
    else:
        yt = y = mt.tensor(y)

    if hasattr(y, 'dtype') and y.dtype != np.object_:
        unique_y = mt.unique(y, aggregate_size=1)
    else:
        unique_y = None
    op = IsMultilabel(y=y, unique_y=unique_y)
    return op(yt, unique_y)


class TypeOfTarget(LearnOperand, LearnOperandMixin):
    __slots__ = ('_unique_y_chunk', '_check_all_finite_chunk')
    _op_type_ = OperandDef.TYPE_OF_TARGET

    _y = AnyField('y')
    # for chunks
    _is_multilabel = KeyField('is_multilabel')
    _first_value = KeyField('first_value')
    _check_float = KeyField('check_float')
    _assert_all_finite = KeyField('assert_all_finite')
    _unique_y = KeyField('unique_y')
    _y_shape = TupleField('y_shape')
    _y_dtype = DataTypeField('y_dtype')
    _checked_targets = ListField('checked_targets')

    def __init__(self, y=None, is_multilabel=None, first_value=None,
                 check_float=None, assert_all_finite=None,
                 unique_y=None, y_shape=None, y_dtype=None,
                 checked_targets=None, **kw):
        super().__init__(_y=y, _is_multilabel=is_multilabel,
                         _first_value=first_value, _check_float=check_float,
                         _assert_all_finite=assert_all_finite,
                         _unique_y=unique_y, _y_shape=y_shape,
                         _y_dtype=y_dtype, _checked_targets=checked_targets, **kw)
        self._output_types = [OutputType.tensor]

    @property
    def y(self):
        return self._y

    @property
    def is_multilabel(self):
        return self._is_multilabel

    @property
    def first_value(self):
        return self._first_value

    @property
    def check_float(self):
        return self._check_float

    @property
    def assert_all_finite(self):
        return self._assert_all_finite

    @property
    def unique_y(self):
        return self._unique_y

    @property
    def y_shape(self):
        return self._y_shape

    @property
    def y_dtype(self):
        return self._y_dtype

    @property
    def checked_targets(self):
        return self._checked_targets

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs)
        for attr in ['_y', '_is_multilabel', '_first_value',
                     '_check_float', '_assert_all_finite',
                     '_unique_y']:
            v = getattr(self, attr)
            if isinstance(v, (Base, Entity)):
                setattr(self, attr, next(inputs_iter))

    def __call__(self, y):
        inputs = [y] if isinstance(y, (Base, Entity)) else []
        return self.new_tileable(inputs, shape=(), order=TensorOrder.C_ORDER,
                                 dtype=np.dtype(object))

    @classmethod
    def tile(cls, op):
        out = op.outputs[0]
        y = op.y

        chunk_inputs = []
        is_multilabel_chunk = recursive_tile(is_multilabel(y)).chunks[0]
        chunk_inputs.append(is_multilabel_chunk)

        if not isinstance(y, (Base, Entity)):
            if hasattr(y, '__array__'):
                y = np.asarray(y)
            y = mt.asarray(y)
        if np.isnan(y.size):  # pragma: no cover
            raise TilesError('y has unknown shape')

        chunk_op = TypeOfTarget(is_multilabel=is_multilabel_chunk,
                                y_shape=y.shape, y_dtype=y.dtype)

        if y.ndim <= 2 and y.size > 0 and y.dtype == object:
            first_value_chunk = recursive_tile(y[(0,) * y.ndim]).chunks[0]
            chunk_inputs.append(first_value_chunk)
            chunk_op._first_value = first_value_chunk

        if y.dtype.kind == 'f':
            check_float_chunk = recursive_tile(mt.any(y != y.astype(int))).chunks[0]
            chunk_inputs.append(check_float_chunk)
            chunk_op._check_float = check_float_chunk

            assert_all_finite_chunk = recursive_tile(assert_all_finite(y)).chunks[0]
            chunk_inputs.append(assert_all_finite_chunk)
            chunk_op._assert_all_finite = assert_all_finite_chunk

        if y.size > 0:
            unique_y_chunk = recursive_tile(mt.unique(y, aggregate_size=1)).chunks[0]
            chunk_inputs.append(unique_y_chunk)
            chunk_op._unique_y = unique_y_chunk

        chunk = chunk_op.new_chunk(chunk_inputs, dtype=out.dtype,
                                   shape=out.shape, order=out.order, index=())
        params = out.params
        params['nsplits'] = ()
        params['chunks'] = [chunk]
        new_op = op.copy()
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    def _execute(cls, ctx, op):
        is_multilabel_ = ctx[op.is_multilabel.key]
        shape = op.y_shape
        ndim = len(shape)
        dtype = op.y_dtype

        if is_multilabel_:
            return 'multilabel-indicator'

        if ndim > 2 or (dtype == object and shape[0] and
                        not isinstance(ctx[op.first_value.key], str)):
            return 'unknown'  # [[[1, 2]]] or [obj_1] and not ["label_1"]

        if ndim == 2 and shape[1] == 0:
            return 'unknown'  # [[]]

        if ndim == 2 and shape[1] > 1:
            suffix = '-multioutput'  # [[1, 2], [1, 2]]
        else:
            suffix = ""  # [1, 2, 3] or [[1], [2], [3]]

        # check float and contains non-integer float values
        if dtype.kind == 'f' and ctx[op.check_float.key]:
            # [.1, .2, 3] or [[.1, .2, 3]] or [[1., .2]] and not [1., 2., 3.]
            assert ctx[op.assert_all_finite.key]
            return 'continuous' + suffix

        if op.unique_y is not None:
            unique_y_len = len(ctx[op.unique_y.key])
        else:
            # y.size == 0
            unique_y_len = 0
        if (unique_y_len > 2) or (ndim >= 2 and shape[1] > 1):
            return 'multiclass' + suffix   # [1, 2, 3] or [[1., 2., 3]] or [[1, 2]]
        else:
            return 'binary'  # [1, 2] or [["a"], ["b"]]

    @classmethod
    def execute(cls, ctx, op):
        target = cls._execute(ctx, op)
        if op.checked_targets is not None and len(op.checked_targets) > 0:
            if target not in op.checked_targets:
                raise ValueError('Unknown label type: {}'.format(target))
        ctx[op.outputs[0].key] = target


def type_of_target(y):
    """Determine the type of data indicated by the target.

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
    valid = ((isinstance(y, valid_types) or hasattr(y, '__array__'))
             and not isinstance(y, str))

    if not valid:
        raise ValueError('Expected array-like (array or non-string sequence), '
                         'got %r' % y)

    sparse_pandas = (y.__class__.__name__ in ['SparseSeries', 'SparseArray'])
    if sparse_pandas:  # pragma: no cover
        raise ValueError("y cannot be class 'SparseSeries' or 'SparseArray'")

    if isinstance(y, (Base, Entity)):
        y = mt.tensor(y)

    op = TypeOfTarget(y=y)
    return op(y)


def check_classification_targets(y):
    """Ensure that target y is of a non-regression type.

    Only the following target types (as defined in type_of_target) are allowed:
        'binary', 'multiclass', 'multiclass-multioutput',
        'multilabel-indicator', 'multilabel-sequences'

    Parameters
    ----------
    y : array-like
    """
    y_type = type_of_target(y)
    y_type.op._checked_targets = ['binary', 'multiclass', 'multiclass-multioutput',
                                  'multilabel-indicator', 'multilabel-sequences']
    return y_type
