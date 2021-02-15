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

from ...utils import require_not_none
from ..arithmetic.utils import arithmetic_operand
from ..utils import infer_dtype
from .core import spspecial, TensorSpecialUnaryOp, TensorSpecialMultiOp, TensorSpecialBinOp


class NoOrderSpecialMixin:
    @classmethod
    def _get_func(cls, xp):
        func = super()._get_func(xp)

        def _wrapped(*args, **kw):
            kw.pop('order', None)
            return func(*args, **kw)

        return _wrapped


@arithmetic_operand(sparse_mode='unary')
class TensorGamma(TensorSpecialUnaryOp):
    _func_name = 'gamma'


@require_not_none(spspecial.gamma)
@infer_dtype(spspecial.gamma)
def gamma(x, **kwargs):
    op = TensorGamma(**kwargs)
    return op(x)


@arithmetic_operand(sparse_mode='unary')
class TensorGammaln(TensorSpecialUnaryOp):
    _func_name = 'gammaln'


@require_not_none(spspecial.gammaln)
@infer_dtype(spspecial.gammaln)
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


@arithmetic_operand(sparse_mode='unary')
class TensorLogGamma(TensorSpecialUnaryOp):
    _func_name = 'loggamma'


@require_not_none(spspecial.loggamma)
@infer_dtype(spspecial.loggamma)
def loggamma(x, **kwargs):
    op = TensorLogGamma(**kwargs)
    return op(x)


@arithmetic_operand(sparse_mode='unary')
class TensorGammaSgn(TensorSpecialUnaryOp):
    _func_name = 'gammasgn'


@require_not_none(spspecial.gammasgn)
@infer_dtype(spspecial.gammasgn)
def gammasgn(x, **kwargs):
    op = TensorGammaSgn(**kwargs)
    return op(x)


@arithmetic_operand(sparse_mode='binary_and')
class TensorGammaInc(TensorSpecialBinOp):
    _func_name = 'gammainc'


@require_not_none(spspecial.gammainc)
@infer_dtype(spspecial.gammainc)
def gammainc(a, b, **kwargs):
    op = TensorGammaInc(**kwargs)
    return op(a, b)


@arithmetic_operand(sparse_mode='binary_and')
class TensorGammaIncInv(TensorSpecialBinOp):
    _func_name = 'gammaincinv'


@require_not_none(spspecial.gammaincinv)
@infer_dtype(spspecial.gammaincinv)
def gammaincinv(a, b, **kwargs):
    op = TensorGammaIncInv(**kwargs)
    return op(a, b)


@arithmetic_operand(sparse_mode='binary_and')
class TensorGammaIncc(TensorSpecialBinOp):
    _func_name = 'gammaincc'


@require_not_none(spspecial.gammainc)
@infer_dtype(spspecial.gammainc)
def gammaincc(a, b, **kwargs):
    op = TensorGammaIncc(**kwargs)
    return op(a, b)


@arithmetic_operand(sparse_mode='binary_and')
class TensorGammaInccInv(TensorSpecialBinOp):
    _func_name = 'gammainccinv'


@require_not_none(spspecial.gammainccinv)
@infer_dtype(spspecial.gammainccinv)
def gammainccinv(a, b, **kwargs):
    op = TensorGammaInccInv(**kwargs)
    return op(a, b)


@arithmetic_operand(sparse_mode='binary_and')
class TensorBeta(TensorSpecialBinOp):
    _func_name = 'beta'


@require_not_none(spspecial.beta)
@infer_dtype(spspecial.beta)
def beta(a, b, out=None, **kwargs):
    op = TensorBeta(**kwargs)
    return op(a, b, out=out)


@arithmetic_operand(sparse_mode='binary_and')
class TensorBetaLn(TensorSpecialBinOp):
    _func_name = 'betaln'


@require_not_none(spspecial.betaln)
@infer_dtype(spspecial.betaln)
def betaln(a, b, out=None, **kwargs):
    op = TensorBetaLn(**kwargs)
    return op(a, b, out=out)


class TensorBetaInc(TensorSpecialMultiOp):
    _ARG_COUNT = 3
    _func_name = 'betainc'


@require_not_none(spspecial.betainc)
@infer_dtype(spspecial.betainc)
def betainc(a, b, x, out=None, **kwargs):
    op = TensorBetaInc(**kwargs)
    return op(a, b, x, out=out)


class TensorBetaIncInv(TensorSpecialMultiOp):
    _ARG_COUNT = 3
    _func_name = 'betaincinv'


@require_not_none(spspecial.betaincinv)
@infer_dtype(spspecial.betaincinv)
def betaincinv(a, b, y, out=None, **kwargs):
    op = TensorBetaIncInv(**kwargs)
    return op(a, b, y, out=out)


@arithmetic_operand(sparse_mode='unary')
class TensorPsi(TensorSpecialUnaryOp):
    _func_name = 'psi'


@require_not_none(spspecial.psi)
@infer_dtype(spspecial.psi)
def psi(x, out=None, **kwargs):
    op = TensorPsi(**kwargs)
    return op(x, out=out)


@arithmetic_operand(sparse_mode='unary')
class TensorRGamma(TensorSpecialUnaryOp):
    _func_name = 'rgamma'


@require_not_none(spspecial.rgamma)
@infer_dtype(spspecial.rgamma)
def rgamma(x, out=None, **kwargs):
    op = TensorRGamma(**kwargs)
    return op(x, out=out)


@arithmetic_operand(sparse_mode='binary_and')
class TensorPolyGamma(NoOrderSpecialMixin, TensorSpecialBinOp):
    _func_name = 'polygamma'


@require_not_none(spspecial.polygamma)
@infer_dtype(spspecial.polygamma)
def polygamma(a, b, **kwargs):
    op = TensorPolyGamma(**kwargs)
    return op(a, b)


@arithmetic_operand(sparse_mode='binary_and')
class TensorMultiGammaLn(NoOrderSpecialMixin, TensorSpecialBinOp):
    _func_name = 'multigammaln'


@require_not_none(spspecial.multigammaln)
@infer_dtype(spspecial.multigammaln)
def multigammaln(a, b, **kwargs):
    op = TensorMultiGammaLn(**kwargs)
    return op(a, b)


@arithmetic_operand(sparse_mode='unary')
class TensorDiGamma(TensorSpecialUnaryOp):
    _func_name = 'digamma'


@require_not_none(spspecial.digamma)
@infer_dtype(spspecial.digamma)
def digamma(x, out=None, **kwargs):
    op = TensorDiGamma(**kwargs)
    return op(x, out=out)


@arithmetic_operand(sparse_mode='binary_and')
class TensorPoch(TensorSpecialBinOp):
    _func_name = 'poch'


@require_not_none(spspecial.poch)
@infer_dtype(spspecial.poch)
def poch(a, b, **kwargs):
    op = TensorPoch(**kwargs)
    return op(a, b)
