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
from .core import TensorSpecialUnaryOp, TensorSpecialBinOp, \
    _register_special_op


@_register_special_op
@arithmetic_operand(sparse_mode='unary')
class TensorEntr(TensorSpecialUnaryOp):
    _func_name = 'entr'


@implement_scipy(spspecial.entr)
@infer_dtype(spspecial.entr)
def entr(x, out=None, where=None, **kwargs):
    r"""
    Elementwise function for computing entropy.

    .. math:: \text{entr}(x) = \begin{cases} - x \log(x) & x > 0  \\ 0 & x = 0 \\ -\infty & \text{otherwise} \end{cases}

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    res : Tensor
        The value of the elementwise entropy function at the given points `x`.

    See Also
    --------
    kl_div, rel_entr

    Notes
    -----
    This function is concave.
    """
    op = TensorEntr(**kwargs)
    return op(x, out=out, where=where)


@_register_special_op
class TensorRelEntr(TensorSpecialBinOp):
    _func_name = 'rel_entr'

    @classmethod
    def _is_sparse(cls, x1, x2):
        if hasattr(x1, 'issparse') and x1.issparse():
            return True
        return False


@implement_scipy(spspecial.rel_entr)
@infer_dtype(spspecial.rel_entr)
def rel_entr(x, y, out=None, where=None, **kwargs):
    r"""
    Elementwise function for computing relative entropy.

    .. math::

        \mathrm{rel\_entr}(x, y) =
            \begin{cases}
                x \log(x / y) & x > 0, y > 0 \\
                0 & x = 0, y \ge 0 \\
                \infty & \text{otherwise}
            \end{cases}

    Parameters
    ----------
    x, y : array_like
        Input arrays
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Relative entropy of the inputs

    See Also
    --------
    entr, kl_div

    Notes
    -----
    This function is jointly convex in x and y.

    The origin of this function is in convex programming; see
    [1]_. Given two discrete probability distributions :math:`p_1,
    \ldots, p_n` and :math:`q_1, \ldots, q_n`, to get the relative
    entropy of statistics compute the sum

    .. math::

        \sum_{i = 1}^n \mathrm{rel\_entr}(p_i, q_i).

    See [2]_ for details.

    References
    ----------
    .. [1] Grant, Boyd, and Ye, "CVX: Matlab Software for Disciplined Convex
        Programming", http://cvxr.com/cvx/
    .. [2] Kullback-Leibler divergence,
        https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    """
    op = TensorRelEntr(**kwargs)
    return op(x, y, out=out, where=where)


@_register_special_op
class TensorKlDiv(TensorSpecialBinOp):
    _func_name = 'kl_div'

    @classmethod
    def _is_sparse(cls, x1, x2):
        if hasattr(x1, 'issparse') and x1.issparse():
            return True
        return False


@implement_scipy(spspecial.kl_div)
@infer_dtype(spspecial.kl_div)
def kl_div(x, y, out=None, where=None, **kwargs):
    r"""
    Elementwise function for computing relative entropy.

    .. math::

        \mathrm{rel\_entr}(x, y) =
            \begin{cases}
                x \log(x / y) & x > 0, y > 0 \\
                0 & x = 0, y \ge 0 \\
                \infty & \text{otherwise}
            \end{cases}

    Parameters
    ----------
    x, y : array_like
        Input arrays
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Relative entropy of the inputs

    See Also
    --------
    entr, kl_div

    Notes
    -----
    This function is jointly convex in x and y.

    The origin of this function is in convex programming; see
    [1]_. Given two discrete probability distributions :math:`p_1,
    \ldots, p_n` and :math:`q_1, \ldots, q_n`, to get the relative
    entropy of statistics compute the sum

    .. math::

        \sum_{i = 1}^n \mathrm{rel\_entr}(p_i, q_i).

    See [2]_ for details.

    References
    ----------
    .. [1] Grant, Boyd, and Ye, "CVX: Matlab Software for Disciplined Convex
        Programming", http://cvxr.com/cvx/
    .. [2] Kullback-Leibler divergence,
        https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    """
    op = TensorKlDiv(**kwargs)
    return op(x, y, out=out, where=where)
