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

from ..utils import broadcast_shape
from ..datasource import tensor as astensor
from .broadcast_to import broadcast_to


def broadcast_arrays(*args, **kwargs):
    """
    Broadcast any number of arrays against each other.

    Parameters
    ----------
    `*args` : array_likes
        The tensors to broadcast.

    Returns
    -------
    broadcasted : list of tensors

    Examples
    --------
    >>> import mars.tensor as mt
    >>> from mars.session import new_session

    >>> sess = new_session().as_default()
    >>> x = mt.array([[1,2,3]])
    >>> y = mt.array([[1],[2],[3]])
    >>> sess.run(mt.broadcast_arrays(x, y))
    [array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]]), array([[1, 1, 1],
           [2, 2, 2],
           [3, 3, 3]])]

    """
    if kwargs:
        raise TypeError('broadcast_arrays() got an unexpected keyword '
                        'argument {!r}'.format(next(iter(kwargs.keys()))))

    args = [astensor(arg) for arg in args]

    shape = broadcast_shape(*[arg.shape for arg in args])
    return [broadcast_to(a, shape) for a in args]
