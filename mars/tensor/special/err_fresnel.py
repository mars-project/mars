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

import scipy.special as spspecial

from ..arithmetic.utils import arithmetic_operand
from ..utils import infer_dtype, implement_scipy
from .core import (
    TensorSpecialUnaryOp,
    TensorSpecialMultiOp,
    TensorTupleOp,
    _register_special_op,
)


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorErf(TensorSpecialUnaryOp):
    _func_name = "erf"


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorErfc(TensorSpecialUnaryOp):
    _func_name = "erfc"


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorErfcx(TensorSpecialUnaryOp):
    _func_name = "erfcx"


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorErfi(TensorSpecialUnaryOp):
    _func_name = "erfi"


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorErfinv(TensorSpecialUnaryOp):
    _func_name = "erfinv"


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorErfcinv(TensorSpecialUnaryOp):
    _func_name = "erfcinv"


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorWofz(TensorSpecialUnaryOp):
    _func_name = "wofz"


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorDawsn(TensorSpecialUnaryOp):
    _func_name = "dawsn"


@_register_special_op
class TensorFresnel(TensorTupleOp):
    _func_name = "fresnel"
    _n_outputs = 2


@_register_special_op
class TensorModFresnelP(TensorTupleOp):
    _func_name = "modfresnelp"
    _n_outputs = 2


@_register_special_op
class TensorModFresnelM(TensorTupleOp):
    _func_name = "modfresnelm"
    _n_outputs = 2


@_register_special_op
class TensorVoigtProfile(TensorSpecialMultiOp):
    _ARG_COUNT = 3
    _func_name = "voigt_profile"


@implement_scipy(spspecial.erf)
@infer_dtype(spspecial.erf)
def erf(x, out=None, where=None, **kwargs):
    """
    Returns the error function of complex argument.

    It is defined as ``2/sqrt(pi)*integral(exp(-t**2), t=0..z)``.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    res : Tensor
        The values of the error function at the given points `x`.

    See Also
    --------
    erfc, erfinv, erfcinv, wofz, erfcx, erfi

    Notes
    -----
    The cumulative of the unit normal distribution is given by
    ``Phi(z) = 1/2[1 + erf(z/sqrt(2))]``.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Error_function
    .. [2] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover,
        1972. http://www.math.sfu.ca/~cbm/aands/page_297.htm
    .. [3] Steven G. Johnson, Faddeeva W function implementation.
       http://ab-initio.mit.edu/Faddeeva

    Examples
    --------
    >>> import mars.tensor as mt
    >>> from mars.tensor import special
    >>> import matplotlib.pyplot as plt
    >>> x = mt.linspace(-3, 3)
    >>> plt.plot(x, special.erf(x))
    >>> plt.xlabel('$x$')
    >>> plt.ylabel('$erf(x)$')
    >>> plt.show()
    """
    op = TensorErf(**kwargs)
    return op(x, out=out, where=where)


@implement_scipy(spspecial.erfc)
@infer_dtype(spspecial.erfc)
def erfc(x, out=None, where=None, **kwargs):
    op = TensorErfc(**kwargs)
    return op(x, out=out, where=where)


@implement_scipy(spspecial.erfcx)
@infer_dtype(spspecial.erfcx)
def erfcx(x, out=None, where=None, **kwargs):
    op = TensorErfcx(**kwargs)
    return op(x, out=out, where=where)


@implement_scipy(spspecial.erfi)
@infer_dtype(spspecial.erfi)
def erfi(x, out=None, where=None, **kwargs):
    op = TensorErfi(**kwargs)
    return op(x, out=out, where=where)


@implement_scipy(spspecial.erfinv)
@infer_dtype(spspecial.erfinv)
def erfinv(x, out=None, where=None, **kwargs):
    op = TensorErfinv(**kwargs)
    return op(x, out=out, where=where)


@implement_scipy(spspecial.erfcinv)
@infer_dtype(spspecial.erfcinv)
def erfcinv(x, out=None, where=None, **kwargs):
    op = TensorErfcinv(**kwargs)
    return op(x, out=out, where=where)


@implement_scipy(spspecial.wofz)
@infer_dtype(spspecial.wofz)
def wofz(x, out=None, where=None, **kwargs):
    op = TensorWofz(**kwargs)
    return op(x, out=out, where=where)


@implement_scipy(spspecial.dawsn)
@infer_dtype(spspecial.dawsn)
def dawsn(x, out=None, where=None, **kwargs):
    op = TensorDawsn(**kwargs)
    return op(x, out=out, where=where)


@implement_scipy(spspecial.fresnel)
@infer_dtype(spspecial.fresnel, multi_outputs=True)
def fresnel(x, out=None, **kwargs):
    op = TensorFresnel(**kwargs)
    return op(x, out=out)


@implement_scipy(spspecial.modfresnelp)
@infer_dtype(spspecial.modfresnelp, multi_outputs=True)
def modfresnelp(x, out=None, **kwargs):
    op = TensorModFresnelP(**kwargs)
    return op(x, out=out)


@implement_scipy(spspecial.modfresnelm)
@infer_dtype(spspecial.modfresnelm, multi_outputs=True)
def modfresnelm(x, out=None, **kwargs):
    op = TensorModFresnelM(**kwargs)
    return op(x, out=out)


@implement_scipy(spspecial.voigt_profile)
@infer_dtype(spspecial.voigt_profile)
def voigt_profile(x, sigma, gamma, **kwargs):
    op = TensorVoigtProfile(**kwargs)
    return op(x, sigma, gamma)
