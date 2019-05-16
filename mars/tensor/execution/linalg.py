#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

import numpy as np

from .array import as_same_device, device, is_sparse_module, cp


def _qr(ctx, chunk):
    (a,), device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.device, ret_extra=True)

    with device(device_id):
        q, r = xp.linalg.qr(a)
        qc, rc = chunk.op.outputs
        ctx[qc.key] = q
        ctx[rc.key] = r


def _svd(ctx, chunk):
    (a,), device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.device, ret_extra=True)

    with device(device_id):
        u, s, v = xp.linalg.svd(a, full_matrices=False)
        uc, sc, vc = chunk.op.outputs
        ctx[uc.key] = u
        ctx[sc.key] = s
        ctx[vc.key] = v


def _cholesky(ctx, chunk):
    (a,), device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.device, ret_extra=True)

    with device(device_id):
        if xp is np:
            try:
                import scipy.linalg

                ctx[chunk.key] = scipy.linalg.cholesky(a, lower=chunk.op.lower)
                return
            except ImportError:  # pragma: no cover
                pass

        r = xp.linalg.cholesky(a)
        if not chunk.op.lower:
            r = r.T.conj()

        ctx[chunk.key] = r


def _solve_triangular(ctx, chunk):
    (a, b), device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.device, ret_extra=True)

    with device(device_id):
        if xp is np:
            import scipy.linalg

            try:
                ctx[chunk.key] = scipy.linalg.solve_triangular(a, b, lower=chunk.op.lower)
            except np.linalg.LinAlgError:
                if chunk.op.strict is not False:
                    raise
                ctx[chunk.key] = np.linalg.lstsq(a, b, rcond=-1)[0]
        elif xp is cp:
            import cupyx

            ctx[chunk.key] = cupyx.scipy.linalg.solve_triangular(a, b, lower=chunk.op.lower)
        else:
            ctx[chunk.key] = xp.solve_triangular(a, b, lower=chunk.op.lower, sparse=chunk.op.sparse)


def _lu(ctx, chunk):
    (a,), device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.device, ret_extra=True)

    with device(device_id):
        if xp is np:
            import scipy.linalg

            p, l, u = scipy.linalg.lu(a)
        elif is_sparse_module(xp):
            p, l, u = xp.lu(a)
        else:
            raise NotImplementedError
        pc, lc, uc = chunk.op.outputs

        ctx[pc.key] = p
        ctx[lc.key] = l
        ctx[uc.key] = u


def _norm(ctx, chunk):
    (x,), device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.device, ret_extra=True)

    with device(device_id):
        ctx[chunk.key] = xp.linalg.norm(x, ord=chunk.op.ord, axis=chunk.op.axis,
                                        keepdims=chunk.op.keepdims)


def _tensordot(ctx, chunk):
    (a, b), device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.device, ret_extra=True)

    axes = chunk.op.a_axes, chunk.op.b_axes
    with device(device_id):
        if not chunk.op.sparse and is_sparse_module(xp):
            # tell sparse to do calculation on numpy or cupy dot
            ctx[chunk.key] = xp.tensordot(a, b, axes, sparse=False)
        else:
            ctx[chunk.key] = xp.tensordot(a, b, axes)


def _tensordot_estimate_size(ctx, chunk):
    if chunk.is_sparse():
        raise NotImplementedError

    # empirical value in real environments
    calc_usage = chunk.nbytes

    # add input sizes when sparse-to-dense is needed
    for inp in chunk.inputs:
        if inp.is_sparse():
            calc_usage += inp.nbytes

    ctx[chunk.key] = (chunk.nbytes, calc_usage)


def _dot(ctx, chunk):
    (a, b), device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.device, ret_extra=True)

    with device(device_id):
        if not chunk.op.sparse and is_sparse_module(xp):
            # tell sparse to do calculation on numpy or cupy dot
            ctx[chunk.key] = xp.dot(a, b, sparse=False)
        else:
            ctx[chunk.key] = xp.dot(a, b)


def _matmul(ctx, chunk):
    (a, b), device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.device, ret_extra=True)

    with device(device_id):
        if not chunk.op.sparse and is_sparse_module(xp):
            # tell sparse to do calculation on numpy or cupy matmul
            ctx[chunk.key] = xp.matmul(a, b, sparse=False)
        else:
            ctx[chunk.key] = xp.matmul(a, b)


def register_linalg_handler():
    from ..expressions import linalg
    from ...executor import register

    register(linalg.TensorQR, _qr)
    register(linalg.TensorSVD, _svd)
    register(linalg.TensorCholesky, _cholesky)
    register(linalg.TensorSolveTriangular, _solve_triangular)
    register(linalg.TensorLU, _lu)
    register(linalg.TensorNorm, _norm)
    register(linalg.TensorTensorDot, _tensordot, _tensordot_estimate_size)
    register(linalg.TensorDot, _dot)
    register(linalg.TensorMatmul, _matmul)
