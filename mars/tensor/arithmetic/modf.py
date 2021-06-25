#!/usr/bin/env python
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

import numpy as np

from ... import opcodes as OperandDef
from ..array_utils import as_same_device, device
from ..datasource import tensor as astensor
from .core import TensorOutBinOp


class TensorModf(TensorOutBinOp):
    _op_type_ = OperandDef.MODF

    def __init__(self, casting='same_kind', dtype=None, sparse=False, **kw):
        super().__init__(_casting=casting, _dtype=dtype, _sparse=sparse, **kw)

    @property
    def _fun(self):
        return np.modf

    @classmethod
    def execute(cls, ctx, op):
        inputs, device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            kw = {'casting': op.casting}

            inputs_iter = iter(inputs)
            input = next(inputs_iter)
            if op.out1 is not None:
                out1 = next(inputs_iter)
            else:
                out1 = None
            if op.out2 is not None:
                out2 = next(inputs_iter)
            else:
                out2 = None
            if op.where is not None:
                where = kw['where'] = next(inputs_iter)
            else:
                where = None
            kw['order'] = op.order

            try:
                args = [input]
                if out1 is not None:
                    args.append(out1.copy())
                if out2 is not None:
                    args.append(out2.copy())
                y1, y2 = xp.modf(*args, **kw)
            except TypeError:
                if where is None:
                    raise
                y1, y2 = xp.modf(input)
                y1, y2 = xp.where(where, y1, out1), xp.where(where, y2, out2)

            for c, res in zip(op.outputs, (y1, y2)):
                ctx[c.key] = res


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

    >>> mt.modf([0, 3.5]).execute()
    (array([ 0. ,  0.5]), array([ 0.,  3.]))
    >>> mt.modf(-0.5).execute()
    (-0.5, -0)
    """
    x = astensor(x)
    op = TensorModf(**kwargs)
    return op(x, out1=out1, out2=out2, out=out, where=where)
