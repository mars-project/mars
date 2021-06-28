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

import numpy as np

try:
    from sklearn.preprocessing import normalize as sklearn_normalize
except ImportError:  # pragma: no cover
    sklearn_normalize = None

from ... import opcodes as OperandDef
from ...core import ExecutableTuple, recursive_tile
from ...serialization.serializables import KeyField, StringField, Int32Field, BoolField
from ...tensor.operands import TensorOperand, TensorOperandMixin
from ...tensor.core import TensorOrder
from ...tensor.array_utils import as_same_device, device, sparse, issparse
from ... import tensor as mt
from ..utils import check_array


class TensorNormalize(TensorOperand, TensorOperandMixin):
    _op_module_ = 'learn'
    _op_type_ = OperandDef.NORMALIZE

    _input = KeyField('input')
    _norm = StringField('norm')
    _axis = Int32Field('axis')
    _return_norm = BoolField('return_norm')
    # for test purpose
    _use_sklearn = BoolField('use_sklearn')

    def __init__(self, norm=None, axis=None, return_norm=None,
                 use_sklearn=None, **kw):
        super().__init__(_norm=norm, _axis=axis, _return_norm=return_norm,
                         _use_sklearn=use_sklearn, **kw)
        if self._use_sklearn is None:
            # force to use sklearn if not specified
            self._use_sklearn = True

    @property
    def input(self):
        return self._input

    @property
    def norm(self):
        return self._norm

    @property
    def axis(self):
        return self._axis

    @property
    def return_norm(self):
        return self._return_norm

    @property
    def use_sklearn(self):
        return self._use_sklearn

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    @property
    def output_limit(self):
        return 2 if self._return_norm else 1

    def __call__(self, x, copy=True):
        x = check_array(x, accept_sparse=True,
                        estimator='the normalize function',
                        dtype=(np.float64, np.float32, np.float16))

        normed = None
        if not self._return_norm:
            res = self.new_tensor([x], shape=x.shape,
                                  order=x.order)
        else:
            kws = [
                {'shape': x.shape,
                 'order': x.order},
                {'shape': (x.shape[0] if self._axis == 1 else x.shape[1],),
                 'order': TensorOrder.C_ORDER}
            ]
            res, normed = self.new_tensors([x], kws=kws, output_limit=2)

        if not copy and self._axis == 1:
            # follow the behaviour of sklearn
            x.data = res.data

        if normed is None:
            return res
        return ExecutableTuple([res, normed])

    @classmethod
    def _tile_one_chunk(cls, op):
        outs = op.outputs
        chunk_op = op.copy().reset_key()
        kws = [
            {'shape': outs[0].shape,
             'order': outs[0].order,
             'index': (0, 0)}]
        if len(outs) == 2:
            kws.append({'shape': outs[1].shape,
                        'order': outs[1].order,
                        'index': (0,)})
        chunks = chunk_op.new_chunks([op.input.chunks[0]], kws=kws,
                                     output_limit=len(outs))

        tensor_kws = [
            {'shape': outs[0].shape,
             'order': outs[0].order,
             'chunks': [chunks[0]],
             'nsplits': tuple((s,) for s in outs[0].shape)
             }
        ]
        if len(outs) == 2:
            tensor_kws.append({'shape': outs[1].shape,
                               'order': outs[1].order,
                               'chunks': [chunks[1]],
                               'nsplits': tuple((s,) for s in outs[1].shape)
                               })

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, kws=tensor_kws,
                                  output_limit=len(outs))

    @classmethod
    def _need_tile_into_chunks(cls, op):
        # if true, try to tile into chunks
        # whose implementation is based on sklearn itself
        x = op.input
        if op.gpu:  # pragma: no cover
            return False
        if x.issparse() and op.return_norm and op.norm in ('l1', 'l2'):
            # sklearn cannot handle
            return False
        if x.chunk_shape[op.axis] > 1:
            return False
        return True

    @classmethod
    def _tile_chunks(cls, op):
        assert op.input.chunk_shape[op.axis] == 1
        x = op.input
        axis = op.axis
        outs = op.outputs

        out_chunks = [], []
        for i, c in enumerate(x.chunks):
            chunk_op = op.copy().reset_key()
            kws = [
                {'shape': c.shape,
                 'order': c.order,
                 'index': c.index}
            ]
            if op.return_norm:
                kws.append({
                    'shape': (c.shape[1 - axis],),
                    'order': TensorOrder.C_ORDER,
                    'index': (i,),
                })
            chunks = chunk_op.new_chunks([c], kws=kws,
                                         output_limit=op.output_limit)
            out_chunks[0].append(chunks[0])
            if len(chunks) == 2:
                out_chunks[1].append(chunks[1])

        tensor_kws = [
            {'shape': outs[0].shape,
             'order': outs[0].order,
             'chunks': out_chunks[0],
             'nsplits': x.nsplits}
        ]
        if len(outs) == 2:
            tensor_kws.append({
                'shape': outs[1].shape,
                'order': outs[1].order,
                'chunks': out_chunks[1],
                'nsplits': (x.nsplits[1 - axis],)
            })
        new_op = op.copy()
        return new_op.new_tensors(op.inputs, kws=tensor_kws,
                                  output_limit=len(outs))

    @classmethod
    def tile(cls, op):
        x = op.input
        norm = op.norm
        axis = op.axis

        if len(x.chunks) == 1:
            return cls._tile_one_chunk(op)

        if cls._need_tile_into_chunks(op):
            return cls._tile_chunks(op)
        else:
            if norm == 'l1':
                norms = mt.abs(x).sum(axis=axis)
            elif norm == 'l2':
                norms = mt.sqrt((x ** 2).sum(axis=axis))
            else:
                assert norm == 'max'
                # sparse.max will still be a sparse,
                # force to convert to dense
                norms = mt.max(x, axis=axis).todense()
            norms = mt.where(mt.equal(norms, 0.0), 1.0, norms)
            if axis == 1:
                x = x / norms[:, mt.newaxis]
            else:
                x = x / norms[mt.newaxis, :]

            ret = [(yield from recursive_tile(x))]
            if op.return_norm:
                ret.append((yield from recursive_tile(norms)))

            new_op = op.copy()
            kws = [out.params for out in op.outputs]
            for i, r in enumerate(ret):
                kws[i]['chunks'] = r.chunks
                kws[i]['nsplits'] = r.nsplits
            return new_op.new_tensors(op.inputs, kws=kws)

    @classmethod
    def execute(cls, ctx, op):
        (x,), device_id, xp = as_same_device(
            [ctx[inp.key] for inp in op.inputs], device=op.device, ret_extra=True)
        axis = op.axis
        return_norm = op.return_norm
        norm = op.norm
        outs = op.outputs

        with device(device_id):
            if device_id < 0 and op.use_sklearn and sklearn_normalize is not None:
                # no GPU
                try:
                    if xp is sparse:
                        if axis == 0:
                            xm = x.raw.tocsc()
                        else:
                            xm = x.raw
                    else:
                        xm = x
                    ret = sklearn_normalize(xm, norm=norm, axis=axis,
                                            return_norm=return_norm)
                    normed = None
                    if return_norm:
                        ret, normed = ret
                    if issparse(ret):
                        ret = sparse.SparseNDArray(ret)
                    ctx[outs[0].key] = ret
                    if normed is not None:
                        ctx[outs[1].key] = normed
                    return
                except NotImplementedError:
                    pass

            # fall back
            if axis == 0:
                x = x.T

            if norm == 'l1':
                norms = xp.abs(x).sum(axis=1)
            elif norm == 'l2':
                norms = xp.sqrt((x ** 2).sum(axis=1))
            else:
                norms = xp.max(x, axis=1)
                if issparse(norms):
                    norms = norms.toarray()
            norms[norms == 0.0] = 1.0
            x = x / norms[:, np.newaxis]

            if axis == 0:
                x = x.T

            ctx[outs[0].key] = x
            if return_norm:
                ctx[outs[1].key] = norms


def normalize(X, norm='l2', axis=1, copy=True, return_norm=False):
    """
    Scale input vectors individually to unit norm (vector length).

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape [n_samples, n_features]
        The data to normalize, element by element.
        scipy.sparse matrices should be in CSR format to avoid an
        un-necessary copy.

    norm : 'l1', 'l2', or 'max', optional ('l2' by default)
        The norm to use to normalize each non zero sample (or each non-zero
        feature if axis is 0).

    axis : 0 or 1, optional (1 by default)
        axis used to normalize the data along. If 1, independently normalize
        each sample, otherwise (if 0) normalize each feature.

    copy : boolean, optional, default True
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a tensor and if axis is 1).

    return_norm : boolean, default False
        whether to return the computed norms

    Returns
    -------
    X : {array-like, sparse matrix}, shape [n_samples, n_features]
        Normalized input X.

    norms : Tensor, shape [n_samples] if axis=1 else [n_features]
        A tensor of norms along given axis for X.
        When X is sparse, a NotImplementedError will be raised
        for norm 'l1' or 'l2'.

    See also
    --------
    Normalizer: Performs normalization using the ``Transformer`` API
        (e.g. as part of a preprocessing :class:`mars.learn.pipeline.Pipeline`).
    """
    if norm not in ('l1', 'l2', 'max'):
        raise ValueError(f"'{norm}' is not a supported norm")
    if axis not in (0, 1):
        raise ValueError(f"'{axis}' is not a supported axis")

    op = TensorNormalize(norm=norm, axis=axis, return_norm=return_norm,
                         dtype=np.dtype(np.float64))
    return op(X, copy=copy)
