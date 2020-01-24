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

from ...core import build_mode
from ..core import DATAFRAME_TYPE
from ..utils import wrap_notimplemented_exception
from ..ufunc.tensor import register_tensor_unary_ufunc
from .abs import abs, DataFrameAbs
from .add import add, radd, DataFrameAdd
from .subtract import subtract, rsubtract, DataFrameSubtract
from .floordiv import floordiv, rfloordiv, DataFrameFloorDiv
from .truediv import truediv, rtruediv, DataFrameTrueDiv
from .equal import eq, DataFrameEqual
from .not_equal import ne, DataFrameNotEqual
from .less import lt, DataFrameLess
from .greater import gt, DataFrameGreater
from .less_equal import le, DataFrameLessEqual
from .greater_equal import ge, DataFrameGreaterEqual
from .log import DataFrameLog
from .log2 import DataFrameLog2
from .log10 import DataFrameLog10
from .sin import DataFrameSin
from .cos import DataFrameCos
from .tan import DataFrameTan
from .sinh import DataFrameSinh
from .cosh import DataFrameCosh
from .tanh import DataFrameTanh
from .arcsin import DataFrameArcsin
from .arccos import DataFrameArccos
from .arctan import DataFrameArctan
from .arcsinh import DataFrameArcsinh
from .arccosh import DataFrameArccosh
from .arctanh import DataFrameArctanh
from .radians import DataFrameRadians
from .degrees import DataFrameDegrees
from .ceil import DataFrameCeil
from .floor import DataFrameFloor
from .around import DataFrameAround, round
from .exp import DataFrameExp
from .exp2 import DataFrameExp2
from .expm1 import DataFrameExpm1


def _wrap_eq():
    def call(df, other, **kw):
        if build_mode().is_build_mode:
            return df._equals(other)
        return _wrap_comparison(eq)(df, other, **kw)
    return call


def _wrap_comparison(func):
    def call(df, other, **kw):
        if isinstance(df, DATAFRAME_TYPE) and isinstance(other, DATAFRAME_TYPE):
            # index and columns should be identical
            for index_type in ['index_value', 'columns_value']:
                left, right = getattr(df, index_type), getattr(other, index_type)
                if left.has_value() and right.has_value():
                    # if df and other's index or columns has value
                    index_eq = left.to_pandas().equals(right.to_pandas())
                else:
                    index_eq = left.key == right.key
                if not index_eq:
                    raise ValueError('Can only compare '
                                     'identically-labeled DataFrame object')
        return wrap_notimplemented_exception(func)(df, other, **kw)
    return call


def _install():
    from ..core import DATAFRAME_TYPE, SERIES_TYPE

    # register mars unary ufuncs
    unary_ops = [
        DataFrameAbs, DataFrameLog, DataFrameLog2, DataFrameLog10,
        DataFrameSin, DataFrameCos, DataFrameTan,
        DataFrameSinh, DataFrameCosh, DataFrameTanh,
        DataFrameArcsin, DataFrameArccos, DataFrameArctan,
        DataFrameArcsinh, DataFrameArccosh, DataFrameArctanh,
        DataFrameRadians, DataFrameDegrees,
        DataFrameCeil, DataFrameFloor, DataFrameAround,
        DataFrameExp, DataFrameExp2, DataFrameExpm1,
    ]
    for unary_op in unary_ops:
        register_tensor_unary_ufunc(unary_op)

    for entity in DATAFRAME_TYPE + SERIES_TYPE:
        setattr(entity, 'abs', abs)
        setattr(entity, 'round', round)

        setattr(entity, '__add__', wrap_notimplemented_exception(add))
        setattr(entity, '__radd__', wrap_notimplemented_exception(radd))
        setattr(entity, 'add', add)
        setattr(entity, 'radd', radd)

        setattr(entity, '__sub__', wrap_notimplemented_exception(subtract))
        setattr(entity, '__rsub__', wrap_notimplemented_exception(rsubtract))
        setattr(entity, 'sub', subtract)
        setattr(entity, 'rsub', rsubtract)

        setattr(entity, '__floordiv__', wrap_notimplemented_exception(floordiv))
        setattr(entity, '__rfloordiv__', wrap_notimplemented_exception(rfloordiv))
        setattr(entity, '__truediv__', wrap_notimplemented_exception(truediv))
        setattr(entity, '__rtruediv__', wrap_notimplemented_exception(rtruediv))
        setattr(entity, '__div__', wrap_notimplemented_exception(truediv))
        setattr(entity, '__rdiv__', wrap_notimplemented_exception(rtruediv))
        setattr(entity, 'floordiv', floordiv)
        setattr(entity, 'rfloordiv', rfloordiv)
        setattr(entity, 'truediv', truediv)
        setattr(entity, 'rtruediv', rtruediv)
        setattr(entity, 'div', truediv)
        setattr(entity, 'rdiv', rtruediv)

        setattr(entity, '__eq__', _wrap_eq())
        setattr(entity, 'eq', eq)
        setattr(entity, '__ne__', _wrap_comparison(ne))
        setattr(entity, 'ne', ne)
        setattr(entity, '__lt__', _wrap_comparison(lt))
        setattr(entity, 'lt', lt)
        setattr(entity, '__gt__', _wrap_comparison(gt))
        setattr(entity, 'gt', gt)
        setattr(entity, '__ge__', _wrap_comparison(ge)),
        setattr(entity, 'ge', ge)
        setattr(entity, '__le__', _wrap_comparison(le))
        setattr(entity, 'le', le)


_install()
del _install
