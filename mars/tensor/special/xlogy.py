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
    from scipy.special import xlogy as scipy_xlogy
except ImportError:  # pragma: no cover
    scipy_xlogy = None

from ... import opcodes
from ...utils import require_not_none
from ..utils import infer_dtype
from .core import TensorSpecialBinOp


class TensorXLogY(TensorSpecialBinOp):
    _op_type_ = opcodes.XLOGY
    _func_name = 'xlogy'

    @classmethod
    def _is_sparse(cls, x1, x2):
        if hasattr(x1, 'issparse') and x1.issparse():
            return True
        return False


@require_not_none(scipy_xlogy)
@infer_dtype(scipy_xlogy)
def xlogy(x1, x2, out=None, where=None, **kwargs):
    r"""
    Compute ``x*log(y)`` so that the result is 0 if ``x = 0``.

    Parameters
    ----------
    x : array_like
        Multiplier
    y : array_like
        Argument

    Returns
    -------
    z : array_like
        Computed x*log(y)
    """
    op = TensorXLogY(**kwargs)
    return op(x1, x2, out=out, where=where)
