#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from .split import _split


def array_split(a, indices_or_sections, axis=0):
    """
    Split a tensor into multiple sub-tensors.

    Please refer to the ``split`` documentation.  The only difference
    between these functions is that ``array_split`` allows
    `indices_or_sections` to be an integer that does *not* equally
    divide the axis. For a tensor of length l that should be split
    into n sections, it returns l % n sub-arrays of size l//n + 1
    and the rest of size l//n.

    See Also
    --------
    split : Split tensor into multiple sub-tensors of equal size.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> x = mt.arange(8.0)
    >>> mt.array_split(x, 3).execute()
        [array([ 0.,  1.,  2.]), array([ 3.,  4.,  5.]), array([ 6.,  7.])]

    >>> x = mt.arange(7.0)
    >>> mt.array_split(x, 3).execute()
        [array([ 0.,  1.,  2.]), array([ 3.,  4.]), array([ 5.,  6.])]

    """
    return _split(a, indices_or_sections, axis=axis)
