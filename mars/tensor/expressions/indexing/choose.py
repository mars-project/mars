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

import numpy as np

from .... import opcodes as OperandDef
from ....serialize import ValueType, KeyField, ListField, StringField
from ..utils import broadcast_shape
from ..core import TensorOperand, TensorOperandMixin
from ..datasource import tensor as astensor


class TensorChoose(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.CHOOSE

    _a = KeyField('a')
    _choices = ListField('choices', ValueType.key)
    _mode = StringField('mode')

    def __init__(self, mode=None, dtype=None, **kw):
        super(TensorChoose, self).__init__(_mode=mode, _dtype=dtype, **kw)

    def __setattr__(self, key, value):
        if key == '_mode' and value not in ('raise', 'wrap', 'clip'):
            raise ValueError('mode should be raise, wrap or clip')

        super(TensorChoose, self).__setattr__(key, value)

    @property
    def a(self):
        return self._a

    @property
    def choices(self):
        return self._choices

    @property
    def mode(self):
        return self._mode

    def _set_inputs(self, inputs):
        super(TensorChoose, self)._set_inputs(inputs)
        self._a = self._inputs[0]
        self._choices = self._inputs[1:]

    def __call__(self, a, choices):
        inputs = [a] + choices
        shape = broadcast_shape(a.shape, *[c.shape for c in choices])
        return self.new_tensor(inputs, shape)

    @classmethod
    def tile(cls, op):
        from ..arithmetic.core import TensorElementWise

        return TensorElementWise.tile(op)


def choose(a, choices, mode='raise'):
    """
    Construct a tensor from an index tensor and a set of tensors to choose from.

    First of all, if confused or uncertain, definitely look at the Examples -
    in its full generality, this function is less simple than it might
    seem from the following code description (below ndi =
    `mt.lib.index_tricks`):

    ``mt.choose(a,c) == mt.array([c[a[I]][I] for I in ndi.ndindex(a.shape)])``.

    But this omits some subtleties.  Here is a fully general summary:

    Given an "index" tensor (`a`) of integers and a sequence of `n` tensors
    (`choices`), `a` and each choice tensor are first broadcast, as necessary,
    to tensors of a common shape; calling these *Ba* and *Bchoices[i], i =
    0,...,n-1* we have that, necessarily, ``Ba.shape == Bchoices[i].shape``
    for each `i`.  Then, a new array with shape ``Ba.shape`` is created as
    follows:

    * if ``mode=raise`` (the default), then, first of all, each element of
      `a` (and thus `Ba`) must be in the range `[0, n-1]`; now, suppose that
      `i` (in that range) is the value at the `(j0, j1, ..., jm)` position
      in `Ba` - then the value at the same position in the new array is the
      value in `Bchoices[i]` at that same position;

    * if ``mode=wrap``, values in `a` (and thus `Ba`) may be any (signed)
      integer; modular arithmetic is used to map integers outside the range
      `[0, n-1]` back into that range; and then the new array is constructed
      as above;

    * if ``mode=clip``, values in `a` (and thus `Ba`) may be any (signed)
      integer; negative integers are mapped to 0; values greater than `n-1`
      are mapped to `n-1`; and then the new tensor is constructed as above.

    Parameters
    ----------
    a : int tensor
        This tensor must contain integers in `[0, n-1]`, where `n` is the number
        of choices, unless ``mode=wrap`` or ``mode=clip``, in which cases any
        integers are permissible.
    choices : sequence of tensors
        Choice tensors. `a` and all of the choices must be broadcastable to the
        same shape.  If `choices` is itself a tensor (not recommended), then
        its outermost dimension (i.e., the one corresponding to
        ``choices.shape[0]``) is taken as defining the "sequence".
    out : tensor, optional
        If provided, the result will be inserted into this tensor. It should
        be of the appropriate shape and dtype.
    mode : {'raise' (default), 'wrap', 'clip'}, optional
        Specifies how indices outside `[0, n-1]` will be treated:

          * 'raise' : an exception is raised
          * 'wrap' : value becomes value mod `n`
          * 'clip' : values < 0 are mapped to 0, values > n-1 are mapped to n-1

    Returns
    -------
    merged_array : Tensor
        The merged result.

    Raises
    ------
    ValueError: shape mismatch
        If `a` and each choice tensor are not all broadcastable to the same
        shape.

    See Also
    --------
    Tensor.choose : equivalent method

    Notes
    -----
    To reduce the chance of misinterpretation, even though the following
    "abuse" is nominally supported, `choices` should neither be, nor be
    thought of as, a single tensor, i.e., the outermost sequence-like container
    should be either a list or a tuple.

    Examples
    --------

    >>> import mars.tensor as mt

    >>> choices = [[0, 1, 2, 3], [10, 11, 12, 13],
    ...   [20, 21, 22, 23], [30, 31, 32, 33]]
    >>> mt.choose([2, 3, 1, 0], choices
    ... # the first element of the result will be the first element of the
    ... # third (2+1) "array" in choices, namely, 20; the second element
    ... # will be the second element of the fourth (3+1) choice array, i.e.,
    ... # 31, etc.
    ... ).execute()
    array([20, 31, 12,  3])
    >>> mt.choose([2, 4, 1, 0], choices, mode='clip').execute() # 4 goes to 3 (4-1)
    array([20, 31, 12,  3])
    >>> # because there are 4 choice arrays
    >>> mt.choose([2, 4, 1, 0], choices, mode='wrap').execute() # 4 goes to (4 mod 4)
    array([20,  1, 12,  3])
    >>> # i.e., 0

    A couple examples illustrating how choose broadcasts:

    >>> a = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    >>> choices = [-10, 10]
    >>> mt.choose(a, choices).execute()
    array([[ 10, -10,  10],
           [-10,  10, -10],
           [ 10, -10,  10]])

    >>> # With thanks to Anne Archibald
    >>> a = mt.array([0, 1]).reshape((2,1,1))
    >>> c1 = mt.array([1, 2, 3]).reshape((1,3,1))
    >>> c2 = mt.array([-1, -2, -3, -4, -5]).reshape((1,1,5))
    >>> mt.choose(a, (c1, c2)).execute() # result is 2x3x5, res[0,:,:]=c1, res[1,:,:]=c2
    array([[[ 1,  1,  1,  1,  1],
            [ 2,  2,  2,  2,  2],
            [ 3,  3,  3,  3,  3]],
           [[-1, -2, -3, -4, -5],
            [-1, -2, -3, -4, -5],
            [-1, -2, -3, -4, -5]]])

    """
    a = astensor(a, dtype='i8')
    choices = [astensor(c) for c in choices]

    dtype = np.result_type(*[c.dtype for c in choices])
    op = TensorChoose(mode=mode, dtype=dtype)
    return op(a, choices)
