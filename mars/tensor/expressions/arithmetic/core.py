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

import itertools

import numpy as np

from ....compat import lrange
from ..utils import unify_chunks, broadcast_shape, check_out_param, filter_inputs
from ...core import Tensor, ExecutableTuple
from ..core import TensorOperandMixin
from ..datasource import tensor as astensor


class TensorElementWise(TensorOperandMixin):
    __slots__ = ()

    @classmethod
    def tile(cls, op):
        inputs = unify_chunks(*[(input, lrange(input.ndim)[::-1]) for input in op.inputs])

        chunk_shapes = [t.chunk_shape for t in inputs]
        out_chunk_shape = broadcast_shape(*chunk_shapes)

        out_chunks = [list() for _ in op.outputs]
        nsplits = [[None] * shape for shape in out_chunk_shape]
        get_index = lambda idx, t: tuple(0 if t.nsplits[i] == (1,) else ix for i, ix in enumerate(idx))
        for out_index in itertools.product(*(map(range, out_chunk_shape))):
            in_chunks = [t.cix[get_index(out_index[-t.ndim:], t)] if t.ndim != 0 else t.chunks[0]
                         for t in inputs]
            chunk_op = op.copy().reset_key()
            chunk_shape = broadcast_shape(*(c.shape for c in in_chunks))
            chunks = chunk_op.new_chunks(in_chunks, chunk_shape, index=out_index,
                                         dtype=[o.dtype for o in op.outputs],
                                         kws=[{'side': str(i)} for i in range(len(op.outputs))])
            for i, out_chunk in enumerate(chunks):
                out_chunks[i].append(out_chunk)
            for i, idx, s in zip(itertools.count(0), out_index, chunks[0].shape):
                nsplits[i][idx] = s

        new_op = op.copy()
        kws = [{'chunks': out_chunk, 'nsplits': nsplits} for out_chunk in out_chunks]
        return new_op.new_tensors(list(inputs), [o.shape for o in op.outputs],
                                  dtype=[o.dtype for o in op.outputs],
                                  kws=kws, output_limit=len(op.outputs))


class TensorElementWiseWithInputs(TensorElementWise):
    def _set_sparse(self, inputs):
        raise NotImplementedError

    def new_tensors(self, inputs, shape, **kw):
        self._set_sparse(inputs)
        return super(TensorElementWiseWithInputs, self).new_tensors(inputs, shape, **kw)

    def new_chunks(self, inputs, shape, **kw):
        self._set_sparse(inputs)
        return super(TensorElementWiseWithInputs, self).new_chunks(inputs, shape, **kw)


class TensorBinOp(TensorElementWiseWithInputs):
    __slots__ = ()

    def check_inputs(self, inputs):
        if len(inputs) < 2 or len(inputs) > 4:
            raise ValueError(
                "Binary operand can only accept 2 to 4 inputs, got {0}".format(len(inputs)))

    @classmethod
    def _is_sparse(cls, x1, x2):
        return False

    def _set_sparse(self, inputs):
        setattr(self, '_sparse', self._is_sparse(inputs[0], inputs[1]))

    def _process_inputs(self, x1, x2, out, where):
        x1, x2 = astensor(x1), astensor(x2)
        self._lhs = x1
        self._rhs = x2

        if out is not None:
            if isinstance(out, Tensor):
                self._out = out
            else:
                raise TypeError('out should be Tensor object, got {0} instead'.format(type(out)))
        if where is True:
            where = None
        if where is not None:
            where = astensor(where)
            self._where = where

        return x1, x2, out, where

    @classmethod
    def constant_cls(cls):
        raise NotImplementedError

    def to_constant(self, x1, x2):
        constant_op_cls = self.constant_cls()
        constant_op = constant_op_cls(getattr(self, '_casting'), getattr(self, '_err'),
                                      getattr(self, '_dtype'), getattr(self, '_sparse'))
        return constant_op(x1, x2)

    def _call(self, x1, x2, out=None, where=None):
        # if x1 or x2 is scalar, and out is none, to constant
        if (np.isscalar(x1) or np.isscalar(x2)) and not out and not where:
            return self.to_constant(x1, x2)

        x1, x2, out, where = self._process_inputs(x1, x2, out, where)
        # check broadcast
        shape = broadcast_shape(x1.shape, x2.shape)

        inputs = filter_inputs([x1, x2, out, where])
        t = self.new_tensor(inputs, shape)

        if out is None:
            return t

        check_out_param(out, t, getattr(self, '_casting'))
        out_shape, out_dtype = out.shape, out.dtype

        # if `out` is specified, use out's dtype and shape
        if t.shape != out_shape:
            t = self.new_tensor(inputs, out_shape)
        setattr(self, '_dtype', out_dtype)

        out.data = t.data
        return out

    def __call__(self, x1, x2, out=None, where=None):
        return self._call(x1, x2, out=out, where=where)

    def rcall(self, x1, x2, out=None, where=None):
        return self._call(x2, x1, out=out, where=where)


class TensorConstant(TensorElementWiseWithInputs):
    __slots__ = ()

    @classmethod
    def _is_sparse(cls, x1, x2):
        return False

    def _set_sparse(self, inputs):
        if len(inputs) == 2:
            setattr(self, '_sparse', self._is_sparse(*inputs))
        elif len(inputs) == 1:
            if np.isscalar(getattr(self, '_lhs')):
                setattr(self, '_sparse', self._is_sparse(getattr(self, '_lhs'), inputs[0]))
            if np.isscalar(getattr(self, '_rhs')):
                setattr(self, '_sparse', self._is_sparse(inputs[0], getattr(self, '_rhs')))

    def _call(self, x1, x2):
        x1_scalar = np.isscalar(x1)
        x2_scalar = np.isscalar(x2)
        if x1_scalar and x2_scalar:
            shape = ()
        elif x1_scalar:
            x2 = astensor(x2)
            shape = x2.shape
        else:
            x1 = astensor(x1)
            shape = x1.shape

        self._lhs = x1
        self._rhs = x2
        return self.new_tensor(filter_inputs([x1, x2]), shape)

    def __call__(self, x1, x2):
        return self._call(x1, x2)

    def rcall(self, x1, x2):
        return self._call(x2, x1)


class TensorUnaryOp(TensorElementWiseWithInputs):
    __slots__ = ()

    def _process_inputs(self, x, out, where):
        x = astensor(x)

        if out is not None:
            if isinstance(out, Tensor):
                self._out = out
            else:
                raise TypeError('out should be Tensor object, got {0} instead'.format(type(out)))
        if where is True:
            where = None
        if where is not None:
            where = astensor(where)
            self._where = where

        return x, out, where

    @classmethod
    def _is_sparse(cls, x):
        return False

    def _set_sparse(self, inputs):
        setattr(self, '_sparse', self._is_sparse(inputs[0]))

    def _call(self, x, out=None, where=None):
        x, out, where = self._process_inputs(x, out, where)
        shape = x.shape

        inputs = filter_inputs([x, out, where])
        t = self.new_tensor(inputs, shape)

        if out is None:
            return t

        check_out_param(out, t, getattr(self, '_casting'))
        out_shape, out_dtype = out.shape, out.dtype

        # if `out` is specified, use out's dtype and shape
        if t.shape != out_shape:
            t = self.new_tensor(inputs, out_shape)
        setattr(self, '_dtype', out_dtype)

        out.data = t.data
        return out

    def __call__(self, x, out=None, where=None):
        return self._call(x, out=out, where=where)


class TensorCompare(TensorBinOp):
    @classmethod
    def _is_sparse(cls, x1, x2):
        if x1.issparse() and x2.issparse():
            return True
        return False


class TensorCompareConstant(TensorConstant):
    @classmethod
    def _is_sparse(cls, x1, x2):
        if hasattr(x1, 'issparse') and x1.issparse():
            return True
        if hasattr(x2, 'issparse') and x2.issparse():
            return True
        return False


class TensorOutBinOp(TensorElementWiseWithInputs):
    __slots__ = ()

    def _process_inputs(self, x, out1, out2, where):
        x = astensor(x)

        if out1 is not None:
            if isinstance(out1, Tensor):
                self._out1 = out1
            else:
                raise TypeError('out1 should be Tensor object, got {0} instead'.format(type(out1)))
        if out2 is not None:
            if isinstance(out2, Tensor):
                self._out2 = out2
            else:
                raise TypeError('out2 should be Tensor object, got {0} instead'.format(type(out2)))
        if where is True:
            where = None
        if where is not None:
            where = astensor(where)
            self._where = where

        return x, out1, out2, where

    @classmethod
    def _is_sparse(cls, x):
        return False

    def _set_sparse(self, inputs):
        setattr(self, '_sparse', self._is_sparse(inputs[0]))

    @property
    def _fun(self):
        raise NotImplementedError

    def _call(self, x, out1=None, out2=None, out=None, where=None):
        dtype = [r.dtype for r in self._fun(np.empty(1, dtype=x.dtype))]

        out = out or (None, None)
        out1 = out1 or out[0]
        out2 = out2 or out[1]
        x, out1, out2, where = self._process_inputs(x, out1, out2, where)
        shape = x.shape

        inputs = filter_inputs([x, out1, out2, where])
        t1, t2 = self.new_tensors(inputs, shape, dtype=dtype)

        if out1 is None and out2 is None:
            return ExecutableTuple([t1, t2])

        if out1 is not None:
            check_out_param(out1, t1, getattr(self, '_casting'))
            out1_shape, out1_dtype = out1.shape, out1.dtype
        else:
            out1_shape, out1_dtype = t1.shape, t1.dtype
        if out2 is not None:
            check_out_param(out2, t2, getattr(self, '_casting'))
            out2_shape, out2_dtype = out2.shape, out2.dtype
        else:
            out2_shape, out2_dtype = t2.shape, t2.dtype
        # if `out` is specified, use out's dtype and shape
        if t1.shape != out1_shape or t2.shape != out2_shape:
            t1, t2 = self.new_tensor(inputs, [out1_shape, out2_shape],
                                     dtype=[out1_dtype, out2_dtype])

        if out1 is not None:
            out1.data = t1.data
        else:
            out1 = t1
        if out2 is not None:
            out2.data = t2.data
        else:
            out2 = t2
        return ExecutableTuple([out1, out2])

    def __call__(self, x, out1=None, out2=None, out=None, where=None):
        return self._call(x, out1=out1, out2=out2, out=out, where=where)

    def _set_inputs(self, inputs):
        super(TensorOutBinOp, self)._set_inputs(inputs)
        inputs_iter = iter(self._inputs)

        self._input = next(inputs_iter)
        if getattr(self, '_out1', None) is not None:
            self._out1 = next(inputs_iter)
        if getattr(self, '_out2', None) is not None:
            self._out2 = next(inputs_iter)
        if getattr(self, '_where', None) is not None:
            self._where = next(inputs_iter)
