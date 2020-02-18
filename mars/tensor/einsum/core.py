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

import itertools
from collections import defaultdict

import numpy as np

from ... import opcodes as OperandDef
from ...serialize import AnyField, StringField
from ..array_utils import as_same_device, device
from ..arithmetic.utils import tree_add
from ..core import TensorOrder
from ..operands import TensorOperand, TensorOperandMixin
from ..utils import decide_unify_split
from .einsumfunc import parse_einsum_input, einsum_path


class TensorEinsum(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.EINSUM

    _subscripts = StringField('subscripts')
    _optimize = AnyField('optimize')
    _order = StringField('order')
    _casting = StringField('casting')

    def __init__(self, subscripts=None, optimize=None, dtype=None, order=None, casting=None, **kw):
        super().__init__(_subscripts=subscripts, _optimize=optimize, _dtype=dtype,
                         _order=order, _casting=casting, **kw)

    @property
    def subscripts(self):
        return self._subscripts

    @property
    def optimize(self):
        return self._optimize

    @property
    def order(self):
        return self._order

    @property
    def casting(self):
        return self._casting

    def __call__(self, input_tensors, shape):
        if self.order in 'KA':
            if any(t.order == TensorOrder.C_ORDER for t in input_tensors):
                order = TensorOrder.C_ORDER
            else:
                order = TensorOrder.F_ORDER
        else:
            if self.order == 'C':
                order = TensorOrder.C_ORDER
            else:
                order = TensorOrder.F_ORDER
        return self.new_tensor(input_tensors, shape=shape, dtype=self.dtype, order=order)

    @classmethod
    def tile(cls, op):
        out_tensor = op.outputs[0]
        input_scripts, output_scripts = op.subscripts.split('->')
        tensor_axes = list(zip(op.inputs, input_scripts.split(',')))

        # rechunk to unify nsplits
        input_nsplits = defaultdict(list)
        for (t, axes) in tensor_axes:
            for splits, ax in zip(t.nsplits, axes):
                input_nsplits[ax].append(splits)
        input_tensors = []
        for (t, axes) in tensor_axes:
            new_nsplits = tuple(decide_unify_split(*input_nsplits[ax]) if t.shape[j] > 1 else (t.shape[j],)
                                for j, ax in enumerate(axes))
            input_tensors.append(t.rechunk(new_nsplits)._inplace_tile())

        tensor_indexes = dict()
        output_axes = defaultdict(list)
        axes_splits = dict()
        tensor_contract_axes = []
        for i, (t, axes) in enumerate(zip(input_tensors, input_scripts.split(','))):
            for j in range(t.ndim):
                if axes[j] in output_scripts:
                    # Record the output tensor's axes and its nsplit.
                    tensor_indexes[axes[j]] = range(len(t.nsplits[j]))
                    output_axes[axes[j]].append((i, j))
            axis_splits = dict((axes[j], t.nsplits[j]) for j in range(t.ndim))
            axes_splits.update(axis_splits)
            tensor_contract_axes.append([ax for ax in axes if ax not in output_scripts])

        out_chunks = []
        output_indexes = [tensor_indexes[ax] for ax in output_scripts]
        for out_idx in itertools.product(*output_indexes):
            all_indexes = [[None] * t.ndim for t in input_tensors]
            tensor_shape = []
            for i, idx in enumerate(out_idx):
                tensor_shape.append(axes_splits[output_scripts[i]][idx])
                for (t_idx, axis) in output_axes[output_scripts[i]]:
                    if input_tensors[t_idx].shape[axis] == 1:
                        all_indexes[t_idx][axis] = 0
                    else:
                        all_indexes[t_idx][axis] = idx
            tensor_shape = tuple(tensor_shape)
            einsum_chunks = []
            contract_axes = [s for s in set(input_scripts.replace(',', '')) if s not in output_scripts]
            for contract_indexes in itertools.product(*[range(len(axes_splits[ax])) for ax in contract_axes]):
                for j, t_contract_axes in enumerate(tensor_contract_axes):
                    for axis in t_contract_axes:
                        axis_index = tensor_axes[j][1].index(axis)
                        all_indexes[j][axis_index] = contract_indexes[contract_axes.index(axis)]
                einsum_op = op.copy().reset_key()
                in_chunks = [t.cix[tuple(indices)] for t, indices in zip(input_tensors, all_indexes)]
                chunk = einsum_op.new_chunk(in_chunks, shape=tensor_shape, order=out_tensor.order)
                einsum_chunks.append(chunk)

            if len(einsum_chunks) == 1:
                c = einsum_chunks[0]
                chunk_op = c.op.copy()
                chunk = chunk_op.new_chunk(c.inputs, shape=c.shape, index=out_idx, order=out_tensor.order)
            else:
                chunk = tree_add(op.dtype, einsum_chunks, out_idx, tensor_shape, sparse=op.sparse)
            out_chunks.append(chunk)

        nsplits = [axes_splits[ax] for ax in output_scripts]
        new_op = op.copy()
        return new_op.new_tensors(input_tensors, out_tensor.shape, chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def execute(cls, ctx, op):
        inputs, device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            if xp is np:
                ctx[op.outputs[0].key] = xp.einsum(op.subscripts, *inputs, optimize=op.optimize, dtype=op.dtype,
                                                   order=op.order, casting=op.casting)
            else:
                # Cupy doesn't support `optimize`, `order` and `casting`.
                ctx[op.outputs[0].key] = xp.einsum(op.subscripts, *inputs, dtype=op.dtype)


def einsum(subscripts, *operands, dtype=None, order='K', casting='safe', optimize=False):
    """
    Evaluates the Einstein summation convention on the operands.

    Using the Einstein summation convention, many common multi-dimensional,
    linear algebraic array operations can be represented in a simple fashion.
    In *implicit* mode `einsum` computes these values.

    In *explicit* mode, `einsum` provides further flexibility to compute
    other array operations that might not be considered classical Einstein
    summation operations, by disabling, or forcing summation over specified
    subscript labels.

    See the notes and examples for clarification.

    :param subscripts: Specifies the subscripts for summation as comma separated list of subscript labels.
    An implicit (classical Einstein summation) calculation is performed unless the explicit indicator ‘->’ is
    included as well as subscript labels of the precise output form.
    :param operands: These are the arrays for the operation.
    :param dtype: If provided, forces the calculation to use the data type specified.
    Note that you may have to also give a more liberal casting parameter to allow the conversions.
    Default is None.
    :param order: Controls the memory layout of the output.
    :param casting: Controls what kind of data casting may occur. Setting this to ‘unsafe’ is not recommended,
     as it can adversely affect accumulations.
    :param optimize: Controls if intermediate optimization should occur.
    :return: The calculation based on the Einstein summation convention.

    The Einstein summation convention can be used to compute
    many multi-dimensional, linear algebraic array operations. `einsum`
    provides a succinct way of representing these.

    A non-exhaustive list of these operations,
    which can be computed by `einsum`, is shown below along with examples:

    * Trace of an array, :py:func:`numpy.trace`.
    * Return a diagonal, :py:func:`numpy.diag`.
    * Array axis summations, :py:func:`numpy.sum`.
    * Transpositions and permutations, :py:func:`numpy.transpose`.
    * Matrix multiplication and dot product, :py:func:`numpy.matmul` :py:func:`numpy.dot`.
    * Vector inner and outer products, :py:func:`numpy.inner` :py:func:`numpy.outer`.
    * Broadcasting, element-wise and scalar multiplication, :py:func:`numpy.multiply`.
    * Tensor contractions, :py:func:`numpy.tensordot`.
    * Chained array operations, in efficient calculation order, :py:func:`numpy.einsum_path`.

    The subscripts string is a comma-separated list of subscript labels,
    where each label refers to a dimension of the corresponding operand.
    Whenever a label is repeated it is summed, so ``mt.einsum('i,i', a, b)``
    is equivalent to :py:func:`mt.inner(a,b) <mars.tensor.inner>`. If a label
    appears only once, it is not summed, so ``mt.einsum('i', a)`` produces a
    view of ``a`` with no changes. A further example ``mt.einsum('ij,jk', a, b)``
    describes traditional matrix multiplication and is equivalent to
    :py:func:`mt.matmul(a,b) <mars.tensor.matmul>`.

    In *implicit mode*, the chosen subscripts are important
    since the axes of the output are reordered alphabetically.  This
    means that ``mt.einsum('ij', a)`` doesn't affect a 2D array, while
    ``mt.einsum('ji', a)`` takes its transpose. Additionally,
    ``mt.einsum('ij,jk', a, b)`` returns a matrix multiplication, while,
    ``mt.einsum('ij,jh', a, b)`` returns the transpose of the
    multiplication since subscript 'h' precedes subscript 'i'.

    In *explicit mode* the output can be directly controlled by
    specifying output subscript labels.  This requires the
    identifier '->' as well as the list of output subscript labels.
    This feature increases the flexibility of the function since
    summing can be disabled or forced when required. The call
    ``mt.einsum('i->', a)`` is like :py:func:`mt.sum(a, axis=-1) <mars.tensor.sum>`,
    and ``mt.einsum('ii->i', a)`` is like :py:func:`mt.diag(a) <mars.tensor.diag>`.
    The difference is that `einsum` does not allow broadcasting by default.
    Additionally ``mt.einsum('ij,jh->ih', a, b)`` directly specifies the
    order of the output subscript labels and therefore returns matrix
    multiplication, unlike the example above in implicit mode.

    To enable and control broadcasting, use an ellipsis.  Default
    NumPy-style broadcasting is done by adding an ellipsis
    to the left of each term, like ``mt.einsum('...ii->...i', a)``.
    To take the trace along the first and last axes,
    you can do ``mt.einsum('i...i', a)``, or to do a matrix-matrix
    product with the left-most indices instead of rightmost, one can do
    ``mt.einsum('ij...,jk...->ik...', a, b)``.

    When there is only one operand, no axes are summed, and no output
    parameter is provided, a view into the operand is returned instead
    of a new array.  Thus, taking the diagonal as ``mt.einsum('ii->i', a)``
    produces a view (changed in version 1.10.0).

    `einsum` also provides an alternative way to provide the subscripts
    and operands as ``einsum(op0, sublist0, op1, sublist1, ..., [sublistout])``.
    If the output shape is not provided in this format `einsum` will be
    calculated in implicit mode, otherwise it will be performed explicitly.
    The examples below have corresponding `einsum` calls with the two
    parameter methods.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> a = mt.arange(25).reshape(5,5)
    >>> b = mt.arange(5)
    >>> c = mt.arange(6).reshape(2,3)
    Trace of a matrix:
    >>> mt.einsum('ii', a).execute()
    60
    >>> mt.einsum(a, [0,0]).execute()
    60
    Extract the diagonal (requires explicit form):
    >>> mt.einsum('ii->i', a).execute()
    array([ 0,  6, 12, 18, 24])
    >>> mt.einsum(a, [0,0], [0]).execute()
    array([ 0,  6, 12, 18, 24])
    >>> mt.diag(a).execute()
    array([ 0,  6, 12, 18, 24])
    Sum over an axis (requires explicit form):
    >>> mt.einsum('ij->i', a).execute()
    array([ 10,  35,  60,  85, 110])
    >>> mt.einsum(a, [0,1], [0]).execute()
    array([ 10,  35,  60,  85, 110])
    >>> mt.sum(a, axis=1).execute()
    array([ 10,  35,  60,  85, 110])
    For higher dimensional arrays summing a single axis can be done with ellipsis:
    >>> mt.einsum('...j->...', a).execute()
    array([ 10,  35,  60,  85, 110])
    >>> mt.einsum(a, [Ellipsis,1], [Ellipsis]).execute()
    array([ 10,  35,  60,  85, 110])
    Compute a matrix transpose, or reorder any number of axes:
    >>> mt.einsum('ji', c).execute()
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> mt.einsum('ij->ji', c).execute()
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> mt.einsum(c, [1,0]).execute()
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> mt.transpose(c).execute()
    array([[0, 3],
           [1, 4],
           [2, 5]])
    Vector inner products:
    >>> mt.einsum('i,i', b, b).execute()
    30
    >>> mt.einsum(b, [0], b, [0]).execute()
    30
    >>> mt.inner(b,b).execute()
    30
    Matrix vector multiplication:
    >>> mt.einsum('ij,j', a, b).execute()
    array([ 30,  80, 130, 180, 230])
    >>> mt.einsum(a, [0,1], b, [1]).execute()
    array([ 30,  80, 130, 180, 230])
    >>> mt.dot(a, b).execute()
    array([ 30,  80, 130, 180, 230])
    >>> mt.einsum('...j,j', a, b).execute()
    array([ 30,  80, 130, 180, 230])
    Broadcasting and scalar multiplication:
    >>> mt.einsum('..., ...', 3, c).execute()
    array([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> mt.einsum(',ij', 3, c).execute()
    array([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> mt.einsum(3, [Ellipsis], c, [Ellipsis]).execute()
    array([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> mt.multiply(3, c).execute()
    array([[ 0,  3,  6],
           [ 9, 12, 15]])
    Vector outer product:
    >>> mt.einsum('i,j', mt.arange(2)+1, b).execute()
    array([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])
    >>> mt.einsum(mt.arange(2)+1, [0], b, [1]).execute()
    array([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])
    >>> mt.outer(mt.arange(2)+1, b).execute()
    array([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])
    Tensor contraction:
    >>> a = mt.arange(60.).reshape(3,4,5)
    >>> b = mt.arange(24.).reshape(4,3,2)
    >>> mt.einsum('ijk,jil->kl', a, b).execute()
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])
    >>> mt.einsum(a, [0,1,2], b, [1,0,3], [2,3]).execute()
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])
    >>> mt.tensordot(a,b, axes=([1,0],[0,1])).execute()
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])
    Writeable returned arrays (since version 1.10.0):
    >>> a = mt.zeros((3, 3))
    >>> mt.einsum('ii->i', a)[:] = 1
    >>> a.execute()
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    Example of ellipsis use:
    >>> a = mt.arange(6).reshape((3,2))
    >>> b = mt.arange(12).reshape((4,3))
    >>> mt.einsum('ki,jk->ij', a, b).execute()
    array([[10, 28, 46, 64],
           [13, 40, 67, 94]])
    >>> mt.einsum('ki,...k->i...', a, b).execute()
    array([[10, 28, 46, 64],
           [13, 40, 67, 94]])
    >>> mt.einsum('k...,jk', a, b).execute()
    array([[10, 28, 46, 64],
           [13, 40, 67, 94]])
    Chained array operations. For more complicated contractions, speed ups
    might be achieved by repeatedly computing a 'greedy' path or pre-computing the
    'optimal' path and repeatedly applying it, using an
    `einsum_path` insertion (since version 1.12.0). Performance improvements can be
    particularly significant with larger arrays:
    >>> a = mt.ones(64).reshape(2,4,8)
    Basic `einsum`: ~1520ms  (benchmarked on 3.1GHz Intel i5.)
    >>> for iteration in range(500):
    ...     _ = mt.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a)
    Sub-optimal `einsum` (due to repeated path calculation time): ~330ms
    >>> for iteration in range(500):
    ...     _ = mt.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a, optimize='optimal')
    Greedy `einsum` (faster optimal path approximation): ~160ms
    >>> for iteration in range(500):
    ...     _ = mt.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a, optimize='greedy')
    Optimal `einsum` (best usage pattern in some use cases): ~110ms
    >>> path = mt.einsum_path('ijk,ilm,njm,nlk,abc->',a,a,a,a,a, optimize='optimal')[0]
    >>> for iteration in range(500):
    ...     _ = mt.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a, optimize=path)

    """
    all_inputs = [subscripts] + list(operands)
    inputs, outputs, operands = parse_einsum_input(all_inputs)
    subscripts = "->".join((inputs, outputs))
    axes_shape = dict()
    for axes, op in zip(inputs.split(','), operands):
        for ax, s in zip(axes, op.shape):
            axes_shape[ax] = s

    if optimize:
        optimize, _ = einsum_path(*all_inputs, optimize=optimize)

    shape = tuple(axes_shape[ax] for ax in outputs)
    op = TensorEinsum(subscripts=subscripts, optimize=optimize, dtype=dtype or operands[0].dtype,
                      order=order, casting=casting)
    return op(operands, shape)
