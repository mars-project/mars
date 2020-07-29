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
    from scipy.special import rel_entr as scipy_rel_entr
except ImportError:  # pragma: no cover
    scipy_rel_entr = None

from ... import opcodes as OperandDef
from ...utils import require_not_none
from ..utils import infer_dtype
from .core import TensorSpecialBinOp


class TensorRelEntr(TensorSpecialBinOp):
    _op_type_ = OperandDef.REL_ENTR
    _func_name = 'rel_entr'

    def _is_sparse(cls, x1, x2):
        if hasattr(x1, 'issparse') and x1.issparse():
            return True
        return False


@require_not_none(scipy_rel_entr)
@infer_dtype(scipy_rel_entr)
def rel_entr(x1, x2, out=None, where=None, **kwargs):
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
    return op(x1, x2, out=out, where=where)
