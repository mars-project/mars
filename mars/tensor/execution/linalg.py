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

from .array import as_same_device, device, is_sparse_module


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
        if xp is not np:
            raise NotImplementedError

        import scipy.linalg
        ctx[chunk.key] = scipy.linalg.solve_triangular(a, b, lower=chunk.op.lower)


def _lu(ctx, chunk):
    import scipy.linalg

    (a,), device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.device, ret_extra=True)

    with device(device_id):
        p, l, u = scipy.linalg.lu(a)
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
    from ...operands.arithmetic import TensorDot, Dot, Matmul
    from ...operands.linalg import QR, SVD, Cholesky, SolveTriangular, Norm, LU
    from ...executor import register

    register(QR, _qr)
    register(SVD, _svd)
    register(Cholesky, _cholesky)
    register(SolveTriangular, _solve_triangular)
    register(LU, _lu)
    register(Norm, _norm)
    register(TensorDot, _tensordot)
    register(Dot, _dot)
    register(Matmul, _matmul)
