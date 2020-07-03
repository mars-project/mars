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
    from scipy.special import entr as scipy_entr
except ImportError:  # pragma: no cover
    scipy_entr = None

from ... import opcodes as OperandDef
from ...utils import require_not_none
from ..arithmetic.utils import arithmetic_operand
from ..utils import infer_dtype
from .core import TensorSpecialUnaryOp


@arithmetic_operand(sparse_mode='unary')
class TensorEntr(TensorSpecialUnaryOp):
    _op_type_ = OperandDef.ENTR
    _func_name = 'entr'


@require_not_none(scipy_entr)
@infer_dtype(scipy_entr)
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
