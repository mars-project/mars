#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import numpy as np

from ...utils import check_chunks_unknown_shape
from ...tiles import TilesError
from ..core import TensorOrder
from ..utils import decide_chunk_sizes
from .utils import calc_svd_shapes


class SFQR:
    __slots__ = ()

    @classmethod
    def tile(cls, op):
        """
        Short-and-Fat QR

        Q [R_1 R_2 ...] = [A_1 A_2 ...]
        """
        from .qr import TensorQR
        from .dot import TensorDot
        from ..base import TensorTranspose

        a = op.input
        q, r = op.outputs

        tinyq, tinyr = np.linalg.qr(np.ones((1, 1), dtype=a.dtype))
        q_dtype, r_dtype = tinyq.dtype, tinyr.dtype

        check_nan_shape = False
        rechunk_size = dict()
        if a.chunk_shape[0] != 1:
            check_nan_shape = True
            rechunk_size[0] = a.shape[0]

        if len(a.chunks) > 1:
            check_nan_shape = True
            if a.chunks[0].shape[0] > a.chunks[0].shape[1]:
                rechunk_size[1] = a.shape[0]

        if check_nan_shape:
            check_chunks_unknown_shape([a], TilesError)

        if rechunk_size:
            new_chunks = decide_chunk_sizes(a.shape, rechunk_size, a.dtype.itemsize)
            a = a.rechunk(new_chunks)._inplace_tile()

        # A_1's QR decomposition
        r_chunks = []
        first_chunk = a.chunks[0]
        x, y = first_chunk.shape
        q_shape, r_shape = (first_chunk.shape, (y, y)) if x > y else ((x, x), first_chunk.shape)
        qr_op = TensorQR()
        q_chunk, r_chunk = qr_op.new_chunks([first_chunk], index=(0, 0),
                                            kws=[{'side': 'q', 'dtype': q_dtype,
                                                  'shape': q_shape, 'order': q.order},
                                                 {'side': 'r', 'dtype': r_dtype,
                                                  'shape': r_shape, 'order': r.order}])
        # q is an orthogonal matrix, so q.T and inverse of q is equal
        trans_op = TensorTranspose()
        q_transpose = trans_op.new_chunk([q_chunk], shape=q_chunk.shape)
        r_chunks.append(r_chunk)

        r_rest = [TensorDot().new_chunk([q_transpose, c], shape=(q_transpose.shape[0], c.shape[1]),
                                        index=c.index, order=q.order) for c in a.chunks[1:]]
        r_chunks.extend(r_rest)

        new_op = op.copy()
        q_nsplits = ((q_chunk.shape[0],), (q_chunk.shape[1],))
        r_nsplits = ((r_chunks[0].shape[0],), tuple(c.shape[1] for c in r_chunks))
        kws = [
            {'chunks': [q_chunk], 'nsplits': q_nsplits, 'dtype': q.dtype, 'shape': q.shape},
            {'chunks': r_chunks, 'nsplits': r_nsplits, 'dtype': r.dtype, 'shape': r.shape}
        ]
        return new_op.new_tensors(op.inputs, kws=kws)


class TSQR:
    __slots__ = ()

    @classmethod
    def tile(cls, op):
        from ..merge.concatenate import TensorConcatenate
        from ..indexing.slice import TensorSlice
        from .dot import TensorDot
        from .qr import TensorQR
        from .svd import TensorSVD

        calc_svd = getattr(op, '_is_svd', lambda: None)() or False

        a = op.input

        tinyq, tinyr = np.linalg.qr(np.ones((1, 1), dtype=a.dtype))
        q_dtype, r_dtype = tinyq.dtype, tinyr.dtype

        if a.chunk_shape[1] != 1:
            check_chunks_unknown_shape([a], TilesError)
            new_chunk_size = decide_chunk_sizes(a.shape, {1: a.shape[1]}, a.dtype.itemsize)
            a = a.rechunk(new_chunk_size)._inplace_tile()

        # stage 1, map phase
        stage1_q_chunks, stage1_r_chunks = stage1_chunks = [[], []]  # Q and R chunks
        for c in a.chunks:
            x, y = c.shape
            q_shape, r_shape = (c.shape, (y, y)) if x > y else ((x, x), c.shape)
            qr_op = TensorQR()
            qr_chunks = qr_op.new_chunks([c], index=c.index,
                                         kws=[{'side': 'q', 'dtype': q_dtype, 'shape': q_shape},
                                              {'side': 'r', 'dtype': r_dtype, 'shape': r_shape}])
            stage1_chunks[0].append(qr_chunks[0])
            stage1_chunks[1].append(qr_chunks[1])

        # stage 2, reduce phase
        # concatenate all r chunks into one
        shape = (sum(c.shape[0] for c in stage1_r_chunks), stage1_r_chunks[0].shape[1])
        concat_op = TensorConcatenate(axis=0, dtype=stage1_r_chunks[0].dtype)
        concat_r_chunk = concat_op.new_chunk(stage1_r_chunks, shape=shape,
                                             index=(0, 0), order=TensorOrder.C_ORDER)
        qr_op = TensorQR()
        qr_chunks = qr_op.new_chunks([concat_r_chunk], index=concat_r_chunk.index,
                                     kws=[{'side': 'q', 'dtype': q_dtype, 'order': TensorOrder.C_ORDER,
                                           'shape': (concat_r_chunk.shape[0], min(concat_r_chunk.shape))},
                                          {'side': 'r', 'dtype': r_dtype, 'order': TensorOrder.C_ORDER,
                                           'shape': (min(concat_r_chunk.shape), concat_r_chunk.shape[1])}])
        stage2_q_chunk, stage2_r_chunk = qr_chunks

        # stage 3, map phase
        # split stage2_q_chunk into the same size as stage1_q_chunks
        q_splits = np.cumsum([c.shape[1] for c in stage1_q_chunks]).tolist()
        q_slices = [slice(q_splits[i]) if i == 0 else slice(q_splits[i-1], q_splits[i])
                    for i in range(len(q_splits))]
        stage2_q_chunks = []
        for c, s in zip(stage1_q_chunks, q_slices):
            slice_op = TensorSlice(slices=[s], dtype=c.dtype)
            slice_length = s.stop - (s.start or 0)
            stage2_q_chunks.append(slice_op.new_chunk([stage2_q_chunk], index=c.index,
                                                      order=TensorOrder.C_ORDER,
                                                      shape=(slice_length, stage2_q_chunk.shape[1])))
        stage3_q_chunks = []
        for c1, c2 in zip(stage1_q_chunks, stage2_q_chunks):
            dot_op = TensorDot(dtype=q_dtype)
            shape = (c1.shape[0], c2.shape[1])
            stage3_q_chunks.append(dot_op.new_chunk([c1, c2], shape=shape, index=c1.index,
                                                    order=TensorOrder.C_ORDER))

        if not calc_svd:
            q, r = op.outputs
            new_op = op.copy()
            q_nsplits = (tuple(c.shape[0] for c in stage3_q_chunks), (stage3_q_chunks[0].shape[1],))
            r_nsplits = ((stage2_r_chunk.shape[0],), (stage2_r_chunk.shape[1],))
            kws = [
                # Q
                {'chunks': stage3_q_chunks, 'nsplits': q_nsplits, 'dtype': q.dtype, 'shape': q.shape},
                # R, calculate from stage2
                {'chunks': [stage2_r_chunk], 'nsplits': r_nsplits, 'dtype': r.dtype, 'shape': r.shape}
            ]
            return new_op.new_tensors(op.inputs, kws=kws)
        else:
            U, s, V = op.outputs
            U_dtype, s_dtype, V_dtype = U.dtype, s.dtype, V.dtype
            U_shape, s_shape, V_shape = U.shape, s.shape, V.shape

            svd_op = TensorSVD()
            u_shape, s_shape, v_shape = calc_svd_shapes(stage2_r_chunk)
            stage2_usv_chunks = svd_op.new_chunks([stage2_r_chunk],
                                                  kws=[{'side': 'U', 'dtype': U_dtype,
                                                        'index': stage2_r_chunk.index,
                                                        'shape': u_shape,
                                                        'order': TensorOrder.C_ORDER},
                                                       {'side': 's', 'dtype': s_dtype,
                                                        'index': stage2_r_chunk.index[1:],
                                                        'shape': s_shape,
                                                        'order': TensorOrder.C_ORDER},
                                                       {'side': 'V', 'dtype': V_dtype,
                                                        'index': stage2_r_chunk.index,
                                                        'shape': v_shape,
                                                        'order': TensorOrder.C_ORDER}])
            stage2_u_chunk, stage2_s_chunk, stage2_v_chunk = stage2_usv_chunks

            # stage 4, U = Q @ u
            stage4_u_chunks = []
            if U is not None:  # U is not garbage collected
                for c1 in stage3_q_chunks:
                    dot_op = TensorDot(dtype=U_dtype)
                    shape = (c1.shape[0], stage2_u_chunk.shape[1])
                    stage4_u_chunks.append(dot_op.new_chunk([c1, stage2_u_chunk], shape=shape,
                                                            index=c1.index,
                                                            order=TensorOrder.C_ORDER))

            new_op = op.copy()
            u_nsplits = (tuple(c.shape[0] for c in stage4_u_chunks), (stage4_u_chunks[0].shape[1],))
            s_nsplits = ((stage2_s_chunk.shape[0],),)
            v_nsplits = ((stage2_v_chunk.shape[0],), (stage2_v_chunk.shape[1],))
            kws = [
                {'chunks': stage4_u_chunks, 'nsplits': u_nsplits,
                 'dtype': U_dtype, 'shape': U_shape, 'order': U.order},   # U
                {'chunks': [stage2_s_chunk], 'nsplits': s_nsplits,
                 'dtype': s_dtype, 'shape': s_shape, 'order': s.order},  # s
                {'chunks': [stage2_v_chunk], 'nsplits': v_nsplits,
                 'dtype': V_dtype, 'shape': V_shape, 'order': V.order},  # V
            ]
            return new_op.new_tensors(op.inputs, kws=kws)
