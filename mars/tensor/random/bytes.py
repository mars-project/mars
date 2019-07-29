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


def bytes(random_state, length):
    """
    Return random bytes.

    Parameters
    ----------
    length : int
        Number of random bytes.

    Returns
    -------
    out : str
        String of length `length`.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.random.bytes(10)
    ' eh\x85\x022SZ\xbf\xa4' #random
    """
    return random_state._random_state.bytes(length)
