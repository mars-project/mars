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

import functools

from ..core import DATAFRAME_TYPE, is_build_mode
from ..utils import wrap_notimplemented_exception
from ..ufunc.tensor import register_tensor_ufunc
from .abs import abs_, DataFrameAbs
from .add import add, radd, DataFrameAdd
from .bitwise_and import bitand, rbitand, DataFrameAnd
from .bitwise_or import bitor, rbitor, DataFrameOr
from .bitwise_xor import bitxor, rbitxor, DataFrameXor
from .subtract import subtract, rsubtract, DataFrameSubtract
from .multiply import mul, rmul, DataFrameMul
from .floordiv import floordiv, rfloordiv, DataFrameFloorDiv
from .truediv import truediv, rtruediv, DataFrameTrueDiv
from .mod import mod, rmod, DataFrameMod
from .power import power, rpower, DataFramePower
from .equal import eq, DataFrameEqual
from .not_equal import ne, DataFrameNotEqual
from .less import lt, DataFrameLess
from .greater import gt, DataFrameGreater
from .less_equal import le, DataFrameLessEqual
from .greater_equal import ge, DataFrameGreaterEqual
from .invert import invert, DataFrameNot
from .is_ufuncs import DataFrameIsNan, DataFrameIsInf, DataFrameIsFinite
from .negative import DataFrameNegative, negative
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
from .around import DataFrameAround, around
from .sqrt import DataFrameSqrt
from .exp import DataFrameExp
from .exp2 import DataFrameExp2
from .expm1 import DataFrameExpm1
from .dot import dot


def _wrap_eq():
    @functools.wraps(eq)
    def call(df, other, **kw):
        if is_build_mode():
            return df._equals(other)
        return _wrap_comparison(eq)(df, other, **kw)
    return call


def _wrap_comparison(func):
    @functools.wraps(func)
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
    from ..core import DATAFRAME_TYPE, SERIES_TYPE, INDEX_TYPE

    def _register_method(cls, name, func, wrapper=None):
        if wrapper is None:
            @functools.wraps(func)
            def wrapper(df, *args, **kwargs):
                return func(df, *args, **kwargs)

        try:
            if issubclass(cls, DATAFRAME_TYPE):
                wrapper.__doc__ = func.__frame_doc__
            elif issubclass(cls, SERIES_TYPE):
                wrapper.__doc__ = func.__series_doc__
            else:
                wrapper = func
        except AttributeError:
            wrapper = func

        wrapper.__name__ = func.__name__
        setattr(cls, name, wrapper)

    def _register_bin_method(cls, name, func):
        def call_df_fill(df, other, axis='columns', level=None, fill_value=None):
            return func(df, other, axis=axis, level=level, fill_value=fill_value)

        def call_df_no_fill(df, other, axis='columns', level=None):
            return func(df, other, axis=axis, level=level)

        def call_series_fill(df, other, level=None, fill_value=None, axis=0):
            return func(df, other, axis=axis, level=level, fill_value=fill_value)

        def call_series_no_fill(df, other, level=None, axis=0):
            return func(df, other, axis=axis, level=level)

        if issubclass(cls, DATAFRAME_TYPE):
            call = call_df_fill if 'fill_value' in func.__code__.co_varnames else call_df_no_fill
        elif issubclass(cls, SERIES_TYPE):
            call = call_series_fill if 'fill_value' in func.__code__.co_varnames else call_series_no_fill
        else:
            call = None
        return _register_method(cls, name, func, wrapper=call)

    # register mars tensor ufuncs
    ufunc_ops = [
        # unary
        DataFrameAbs, DataFrameLog, DataFrameLog2, DataFrameLog10,
        DataFrameSin, DataFrameCos, DataFrameTan,
        DataFrameSinh, DataFrameCosh, DataFrameTanh,
        DataFrameArcsin, DataFrameArccos, DataFrameArctan,
        DataFrameArcsinh, DataFrameArccosh, DataFrameArctanh,
        DataFrameRadians, DataFrameDegrees,
        DataFrameCeil, DataFrameFloor, DataFrameAround,
        DataFrameExp, DataFrameExp2, DataFrameExpm1,
        DataFrameSqrt, DataFrameNot, DataFrameIsNan,
        DataFrameIsInf, DataFrameIsFinite, DataFrameNegative,
        # binary
        DataFrameAdd, DataFrameEqual, DataFrameFloorDiv,
        DataFrameGreater, DataFrameGreaterEqual, DataFrameLess,
        DataFrameLessEqual, DataFrameAnd, DataFrameOr, DataFrameXor,
        DataFrameMod, DataFrameMul, DataFrameNotEqual, DataFramePower,
        DataFrameSubtract, DataFrameTrueDiv,
    ]
    for ufunc_op in ufunc_ops:
        register_tensor_ufunc(ufunc_op)

    for entity in DATAFRAME_TYPE + SERIES_TYPE:
        setattr(entity, '__abs__', abs_)
        setattr(entity, 'abs', abs_)
        _register_method(entity, 'round', around)
        setattr(entity, '__invert__', invert)

        setattr(entity, '__add__', wrap_notimplemented_exception(add))
        setattr(entity, '__radd__', wrap_notimplemented_exception(radd))
        _register_bin_method(entity, 'add', add)
        _register_bin_method(entity, 'radd', radd)

        setattr(entity, '__sub__', wrap_notimplemented_exception(subtract))
        setattr(entity, '__rsub__', wrap_notimplemented_exception(rsubtract))
        _register_bin_method(entity, 'sub', subtract)
        _register_bin_method(entity, 'rsub', rsubtract)

        setattr(entity, '__mul__', wrap_notimplemented_exception(mul))
        setattr(entity, '__rmul__', wrap_notimplemented_exception(rmul))
        _register_bin_method(entity, 'mul', mul)
        _register_bin_method(entity, 'multiply', mul)
        _register_bin_method(entity, 'rmul', rmul)

        setattr(entity, '__floordiv__', wrap_notimplemented_exception(floordiv))
        setattr(entity, '__rfloordiv__', wrap_notimplemented_exception(rfloordiv))
        setattr(entity, '__truediv__', wrap_notimplemented_exception(truediv))
        setattr(entity, '__rtruediv__', wrap_notimplemented_exception(rtruediv))
        setattr(entity, '__div__', wrap_notimplemented_exception(truediv))
        setattr(entity, '__rdiv__', wrap_notimplemented_exception(rtruediv))
        _register_bin_method(entity, 'floordiv', floordiv)
        _register_bin_method(entity, 'rfloordiv', rfloordiv)
        _register_bin_method(entity, 'truediv', truediv)
        _register_bin_method(entity, 'rtruediv', rtruediv)
        _register_bin_method(entity, 'div', truediv)
        _register_bin_method(entity, 'rdiv', rtruediv)

        setattr(entity, '__mod__', wrap_notimplemented_exception(mod))
        setattr(entity, '__rmod__', wrap_notimplemented_exception(rmod))
        _register_bin_method(entity, 'mod', mod)
        _register_bin_method(entity, 'rmod', rmod)

        setattr(entity, '__pow__', wrap_notimplemented_exception(power))
        setattr(entity, '__rpow__', wrap_notimplemented_exception(rpower))
        _register_bin_method(entity, 'pow', power)
        _register_bin_method(entity, 'rpow', rpower)

        setattr(entity, '__eq__', _wrap_eq())
        setattr(entity, '__ne__', _wrap_comparison(ne))
        setattr(entity, '__lt__', _wrap_comparison(lt))
        setattr(entity, '__gt__', _wrap_comparison(gt))
        setattr(entity, '__ge__', _wrap_comparison(ge))
        setattr(entity, '__le__', _wrap_comparison(le))
        _register_bin_method(entity, 'eq', eq)
        _register_bin_method(entity, 'ne', ne)
        _register_bin_method(entity, 'lt', lt)
        _register_bin_method(entity, 'gt', gt)
        _register_bin_method(entity, 'ge', ge)
        _register_bin_method(entity, 'le', le)

        setattr(entity, '__matmul__', dot)
        _register_method(entity, 'dot', dot)

        setattr(entity, '__and__', wrap_notimplemented_exception(bitand))
        setattr(entity, '__rand__', wrap_notimplemented_exception(rbitand))

        setattr(entity, '__or__', wrap_notimplemented_exception(bitor))
        setattr(entity, '__ror__', wrap_notimplemented_exception(rbitor))

        setattr(entity, '__xor__', wrap_notimplemented_exception(bitxor))
        setattr(entity, '__rxor__', wrap_notimplemented_exception(rbitxor))

        setattr(entity, '__neg__', wrap_notimplemented_exception(negative))

    for entity in INDEX_TYPE:
        setattr(entity, '__eq__', _wrap_eq())


_install()
del _install
