#!/usr/bin/env python
# -*- coding: utf-8 -*-
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


from .datasource import tensor, array, asarray, ascontiguousarray, asfortranarray, \
    scalar, empty, empty_like, ones, ones_like, zeros, zeros_like, \
    full, full_like, arange, diag, diagflat, eye, identity, linspace, \
    meshgrid, indices, tril, triu, fromtiledb, fromtiledb as from_tiledb, from_dataframe, \
    fromhdf5, fromhdf5 as from_hdf5, fromzarr, fromzarr as from_zarr
from .datastore import totiledb, totiledb as to_tiledb, tohdf5, tohdf5 as to_hdf5, \
    tozarr, tozarr as to_zarr  # pylint: disable=reimported
from .base import result_type, ndim, copyto, transpose, where, broadcast_to, broadcast_arrays, \
    expand_dims, rollaxis, swapaxes, moveaxis, ravel, atleast_1d, atleast_2d, atleast_3d, argwhere, \
    array_split, split, hsplit, vsplit, dsplit, roll, squeeze, diff, ediff1d, \
    flip, flipud, fliplr, repeat, tile, isin, searchsorted, unique, \
    sort, argsort, partition, argpartition, topk, argtopk, copy, trapz, shape, \
    insert, delete, in1d
from .arithmetic import add, subtract, multiply, divide, truediv as true_divide, \
    floordiv as floor_divide, mod, power, float_power, fmod, sqrt, \
    around, round_, round_ as round, logaddexp, logaddexp2, negative, positive, \
    absolute, fabs, absolute as abs, rint, sign, degrees, radians, conj, conjugate, exp, exp2, \
    log, log2, log10, expm1, log1p, square, cbrt, reciprocal, \
    equal, not_equal, less, less_equal, greater, greater_equal, sin, cos, tan, \
    arcsin, arccos, arctan, arctan2, hypot, sinh, cosh, tanh, arcsinh, arccosh, arctanh, \
    deg2rad, rad2deg, bitand as bitwise_and, bitor as bitwise_or, bitxor as bitwise_xor, \
    invert, invert as bitwise_not, lshift as left_shift, rshift as right_shift, \
    logical_and, logical_or, logical_xor, logical_not, \
    maximum, minimum, floor, ceil, trunc, remainder, fmax, fmin, isfinite, isinf, isnan, \
    signbit, copysign, nextafter, spacing, clip, isclose, ldexp, frexp, modf, angle, \
    isreal, iscomplex, real, imag, fix, i0, sinc, nan_to_num, tree_add, tree_multiply
from .statistics import average, cov, corrcoef, digitize, ptp, \
    histogram_bin_edges, histogram, quantile, percentile, median
from .linalg.tensordot import tensordot
from .linalg.dot import dot
from .linalg.inner import inner, innerproduct
from .linalg.vdot import vdot
from .linalg.matmul import matmul
from .reduction import sum, nansum, prod, prod as product, nanprod, \
    max, max as amax, nanmax, min, min as amin, nanmin, all, any, mean, nanmean, \
    argmax, nanargmax, argmin, nanargmin, cumsum, cumprod, \
    var, std, nanvar, nanstd, nancumsum, nancumprod, count_nonzero, allclose, array_equal
from .reshape import reshape
from .merge import concatenate, stack, hstack, vstack, dstack, column_stack, union1d, block
from .indexing import take, compress, extract, choose, unravel_index, \
    nonzero, flatnonzero, fill_diagonal
from .rechunk import rechunk
from .einsum import einsum
from .images import imread
# noinspection PyUnresolvedReferences
from .lib.index_tricks import mgrid, ogrid, ndindex, r_, c_

from . import random
from . import fft
from . import linalg
from . import lib
from . import special
from . import stats

# types
from .core import Tensor
# noinspection PyUnresolvedReferences
from ..core import ExecutableTuple

# noinspection PyUnresolvedReferences
from numpy import newaxis, AxisError, inf, Inf, NINF, nan, NAN, NaN, pi, e, \
    errstate, geterr, seterr
# import numpy types
# noinspection PyUnresolvedReferences
from numpy import dtype, object, number, inexact, floating, complexfloating, \
    integer, signedinteger, unsignedinteger, character, generic, flexible, \
    int, int_, bool, bool_, float_, cfloat, bytes_, unicode_, void, object_, \
    intc, intp, int8, int16, int32, int64, uint8, uint16, uint32, uint64, uint, \
    float, float16, float32, float64, double, complex64, complex128, datetime64, timedelta64
# noinspection PyUnresolvedReferences
from numpy import finfo

# register fuse op and fetch op
from .fuse import TensorFuseChunk, TensorCpFuseChunk, TensorNeFuseChunk
from .fetch import TensorFetch, TensorFetchShuffle
from . import ufunc
del TensorFuseChunk, TensorCpFuseChunk, TensorNeFuseChunk, TensorFetch, TensorFetchShuffle, ufunc
