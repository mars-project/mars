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

try:
    from scipy.special import gammaln as scipy_gammaln
except ImportError:  # pragma: no cover
    scipy_gammaln = None

from ... import opcodes as OperandDef
from ...utils import require_not_none
from ..arithmetic.utils import arithmetic_operand
from ..utils import infer_dtype
from .core import TensorSpecialUnaryOp


@arithmetic_operand(sparse_mode='unary')
class TensorGammaln(TensorSpecialUnaryOp):
    _op_type_ = OperandDef.GAMMALN
    _func_name = 'gammaln'


@require_not_none(scipy_gammaln)
@infer_dtype(scipy_gammaln)
def gammaln(x, out=None, where=None, **kwargs):
    """
    Logarithm of the absolute value of the Gamma function.

    Parameters
    ----------
    x : array-like
        Values on the real line at which to compute ``gammaln``
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
    gammaln : Tensor
        Values of ``gammaln`` at x.

    See Also
    --------
    gammasgn : sign of the gamma function
    loggamma : principal branch of the logarithm of the gamma function

    Notes
    -----
    When used in conjunction with `gammasgn`, this function is useful
    for working in logspace on the real axis without having to deal with
    complex numbers, via the relation ``exp(gammaln(x)) = gammasgn(x)*gamma(x)``.

    For complex-valued log-gamma, use `loggamma` instead of `gammaln`.
    """
    op = TensorGammaln(**kwargs)
    return op(x, out=out, where=where)
