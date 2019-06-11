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
from .core import TensorOutBinOpMixin, TensorOutBinOp


class TensorModf(TensorOutBinOp, TensorOutBinOpMixin):
    _op_type_ = OperandDef.MODF

    def __init__(self, casting='same_kind', dtype=None, sparse=False, **kw):
        super(TensorModf, self).__init__(_casting=casting, _dtype=dtype,
                                         _sparse=sparse, **kw)

    @property
    def _fun(self):
        return np.modf


def modf(x, out1=None, out2=None, out=None, where=None, **kwargs):
    """
    Return the fractional and integral parts of a tensor, element-wise.

    The fractional and integral parts are negative if the given number is
    negative.

    Parameters
    ----------
    x : array_like
        Input tensor.
    out : Tensor, None, or tuple of Tensor and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated tensor is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        Values of True indicate to calculate the ufunc at that position, values
        of False indicate to leave the value in the output alone.
    **kwargs

    Returns
    -------
    y1 : Tensor
        Fractional part of `x`.
    y2 : Tensor
        Integral part of `x`.

    Notes
    -----
    For integer input the return values are floats.

    See Also
    --------
    divmod : ``divmod(x, 1)`` is equivalent to ``modf`` with the return values
             switched, except it always has a positive remainder.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> from mars.session import new_session

    >>> sess = new_session().as_default()
    >>> sess.run(mt.modf([0, 3.5]))
    (array([ 0. ,  0.5]), array([ 0.,  3.]))
    >>> sess.run(mt.modf(-0.5))
    (-0.5, -0)
    """
    op = TensorModf(**kwargs)
    return op(x, out1=out1, out2=out2, out=out, where=where)
