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

from math import log

from ... import tensor as mt
from ..core import TENSOR_TYPE
from ..datasource import tensor as astensor
from ..special import entr, rel_entr


def entropy(pk, qk=None, base=None):
    """Calculate the entropy of a distribution for given probability values.

    If only probabilities `pk` are given, the entropy is calculated as
    ``S = -sum(pk * log(pk), axis=0)``.

    If `qk` is not None, then compute the Kullback-Leibler divergence
    ``S = sum(pk * log(pk / qk), axis=0)``.

    This routine will normalize `pk` and `qk` if they don't sum to 1.

    Parameters
    ----------
    pk : sequence
        Defines the (discrete) distribution. ``pk[i]`` is the (possibly
        unnormalized) probability of event ``i``.
    qk : sequence, optional
        Sequence against which the relative entropy is computed. Should be in
        the same format as `pk`.
    base : float, optional
        The logarithmic base to use, defaults to ``e`` (natural logarithm).

    Returns
    -------
    S : Tensor
        The calculated entropy.

    """
    pk = astensor(pk)
    pk = 1.0 * pk / mt.sum(pk, axis=0)
    if qk is None:
        vec = entr(pk)
    else:
        qk = astensor(qk)
        if len(qk) != len(pk):
            raise ValueError("qk and pk must have same length.")
        qk = 1.0 * qk / mt.sum(qk, axis=0)
        vec = rel_entr(pk, qk)
    S = mt.sum(vec, axis=0)
    if base is not None:
        if isinstance(base, TENSOR_TYPE):
            S /= mt.log(base)
        else:
            S /= log(base)
    return S
