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

import operator
from functools import wraps
try:
    from collections.abc import Container, Iterable, Sequence
except ImportError:  # pragma: no cover
    from collections import Container, Iterable, Sequence

from ...compat import reduce
from ..array_utils import get_array_module


def get_axis(axis):
    return tuple(axis) if axis is not None else axis


def get_arg_axis(axis, ndim):
    return None if len(axis) == ndim or ndim == 1 else axis[0]


def keepdims_wrapper(a_callable):
    @wraps(a_callable)
    def keepdims_wrapped_callable(x, axis=None, keepdims=None, *args, **kwargs):
        xp = get_array_module(x)
        if xp == np:
            func = a_callable
        else:
            func = getattr(xp, a_callable.__name__)

        r = func(x, axis=axis, *args, **kwargs)

        if not keepdims:
            return xp.asarray(r)

        axes = axis

        if axes is None:
            axes = range(x.ndim)

        if not isinstance(axes, (Container, Iterable, Sequence)):
            axes = [axes]

        if r.ndim != x.ndim:
            r_slice = tuple()
            for each_axis in range(x.ndim):
                if each_axis in axes:
                    r_slice += (None,)
                else:
                    r_slice += (slice(None),)

            r = r[r_slice]

        return r

    return keepdims_wrapped_callable


_sum = keepdims_wrapper(np.sum)
_nansum = keepdims_wrapper(np.nansum)


def _numel(x, **kwargs):
    xp = get_array_module(x)
    return _sum(xp.ones_like(x), **kwargs)


def _nannumel(x, **kwargs):
    x_size = reduce(operator.mul, x.shape)
    xp = get_array_module(x)
    return x_size - _sum(xp.isnan(x), **kwargs)


