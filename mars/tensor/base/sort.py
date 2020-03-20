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

from ... import opcodes as OperandDef
from ...serialize import ValueType, Int32Field, StringField, ListField, BoolField
from ...core import ExecutableTuple
from ..operands import TensorOperand, TensorShuffleProxy, TensorOrder
from ..array_utils import as_same_device, device
from ..datasource import tensor as astensor
from ..utils import validate_axis, validate_order
from .psrs import TensorPSRSOperandMixin


class TensorSort(TensorOperand, TensorPSRSOperandMixin):
    _op_type_ = OperandDef.SORT

    _axis = Int32Field('axis')
    _kind = StringField('kind')
    _parallel_kind = StringField('parallel_kind')
    _order = ListField('order', ValueType.string)
    _psrs_kinds = ListField('psrs_kinds', ValueType.string)
    _need_align = BoolField('need_align')
    _return_value = BoolField('return_value')
    _return_indices = BoolField('return_indices')

    def __init__(self, axis=None, kind=None, parallel_kind=None, order=None,
                 psrs_kinds=None, need_align=None, return_value=None,
                 return_indices=None, dtype=None, gpu=None, **kw):
        super().__init__(_axis=axis, _kind=kind, _parallel_kind=parallel_kind,
                         _order=order, _psrs_kinds=psrs_kinds,
                         _need_align=need_align, _return_value=return_value,
                         _return_indices=return_indices,
                         _dtype=dtype, _gpu=gpu, **kw)

    @property
    def axis(self):
        return self._axis

    @property
    def kind(self):
        return self._kind

    @property
    def parallel_kind(self):
        return self._parallel_kind

    @property
    def order(self):
        return self._order

    @property
    def psrs_kinds(self):
        return self._psrs_kinds

    @property
    def need_align(self):
        return self._need_align

    @property
    def return_value(self):
        return self._return_value

    @property
    def return_indices(self):
        return self._return_indices

    @property
    def output_limit(self):
        return int(bool(self._return_value)) + int(bool(self._return_indices))

    def __call__(self, a):
        kws = []
        if self._return_value:
            kws.append({
                'shape': a.shape,
                'order': a.order,
                'dtype': a.dtype,
                'type': 'sorted'
            })
        if self._return_indices:
            kws.append({
                'shape': a.shape,
                'order': TensorOrder.C_ORDER,
                'dtype': np.dtype(np.int64),
                'type': 'argsort'
            })
        ret = self.new_tensors([a], kws=kws)
        if len(kws) == 1:
            return ret[0]
        return ExecutableTuple(ret)

    @classmethod
    def _tile_psrs(cls, op):
        """
        Refer to http://csweb.cs.wfu.edu/bigiron/LittleFE-PSRS/build/html/PSRSalgorithm.html
        to see explanation of parallel sorting by regular sampling
        """
        out_tensor = op.outputs[0]
        in_tensor, axis_chunk_shape, out_idxes, need_align = cls.preprocess(op)
        axis_offsets = [0] + np.cumsum(in_tensor.nsplits[op.axis]).tolist()[:-1]
        return_value, return_indices = op.return_value, op.return_indices

        out_value_chunks, out_indices_chunks = [], []
        for out_idx in out_idxes:
            # stage 1: local sort and regular samples collected
            sorted_chunks, indices_chunks, sampled_chunks = cls.local_sort_and_regular_sample(
                op, in_tensor, axis_chunk_shape, axis_offsets, out_idx)

            # stage 2: gather and merge samples, choose and broadcast p-1 pivots
            concat_pivot_chunk = cls.concat_and_pivot(
                op, axis_chunk_shape, out_idx, sorted_chunks, sampled_chunks)

            # stage 3: Local data is partitioned
            partition_chunks = cls.partition_local_data(
                op, axis_chunk_shape, sorted_chunks, indices_chunks, concat_pivot_chunk)

            proxy_chunk = TensorShuffleProxy(dtype=partition_chunks[0].dtype).new_chunk(
                partition_chunks, shape=())

            # stage 4: all *ith* classes are gathered and merged
            partition_sort_chunks, partition_indices_chunks, sort_info_chunks = \
                cls.partition_merge_data(op, need_align, None, partition_chunks, proxy_chunk)

            if not need_align:
                if return_value:
                    out_value_chunks.extend(partition_sort_chunks)
                if return_indices:
                    out_indices_chunks.extend(partition_indices_chunks)
            else:
                align_reduce_value_chunks, align_reduce_indices_chunks = \
                    cls.align_partitions_data(
                        op, out_idx, in_tensor, partition_sort_chunks,
                        partition_indices_chunks, sort_info_chunks)
                if return_value:
                    out_value_chunks.extend(align_reduce_value_chunks)
                if return_indices:
                    out_indices_chunks.extend(align_reduce_indices_chunks)

        new_op = op.copy()
        nsplits = list(in_tensor.nsplits)
        if not need_align:
            nsplits[op.axis] = (np.nan,) * axis_chunk_shape
        kws = []
        if return_value:
            kws.append({
                'shape': out_tensor.shape,
                'order': out_tensor.order,
                'chunks': out_value_chunks,
                'nsplits': nsplits,
                'dtype': out_tensor.dtype
            })
        if return_indices:
            kws.append({
                'shape': out_tensor.shape,
                'order': TensorOrder.C_ORDER,
                'chunks': out_indices_chunks,
                'nsplits': nsplits,
                'dtype': np.dtype(np.int64)
            })
        return new_op.new_tensors(op.inputs, kws=kws)

    @classmethod
    def tile(cls, op):
        in_tensor = op.inputs[0]
        return_value, return_indices = op.return_value, op.return_indices

        if in_tensor.chunk_shape[op.axis] == 1:
            out_chunks, out_indices_chunks = [], []
            for chunk in in_tensor.chunks:
                chunk_op = op.copy().reset_key()
                kws = []
                if return_value:
                    kws.append({
                        'shape': chunk.shape,
                        'index': chunk.index,
                        'order': chunk.order,
                        'dtype': chunk.dtype,
                        'type': 'sorted'
                    })
                if return_indices:
                    kws.append({
                        'shape': chunk.shape,
                        'index': chunk.index,
                        'order': TensorOrder.C_ORDER,
                        'dtype': np.dtype(np.int64),
                        'type': 'argsort'
                    })
                chunks = chunk_op.new_chunks([chunk], kws=kws)
                if return_value:
                    out_chunks.append(chunks[0])
                if return_indices:
                    out_indices_chunks.append(chunks[-1])

            new_op = op.copy()
            kws = [out.params for out in op.outputs]
            if return_value:
                kws[0]['nsplits'] = in_tensor.nsplits
                kws[0]['chunks'] = out_chunks
            if return_indices:
                kws[-1]['nsplits'] = in_tensor.nsplits
                kws[-1]['chunks'] = out_indices_chunks
            return new_op.new_tensors([in_tensor], kws=kws)
        else:
            # use parallel sorting by regular sampling
            return cls._tile_psrs(op)

    @classmethod
    def execute(cls, ctx, op):
        (a,), device_id, xp = as_same_device([ctx[inp.key] for inp in op.inputs],
                                             device=op.device, ret_extra=True)
        return_value, return_indices = op.return_value, op.return_indices

        with device(device_id):
            kw = {}
            if op.kind is not None:
                kw['kind'] = op.kind
            if op.order is not None:
                kw['order'] = op.order

            if return_indices:
                if not return_value:
                    ctx[op.outputs[0].key] = xp.argsort(a, axis=op.axis, **kw)
                else:
                    indices = ctx[op.outputs[1].key] = xp.argsort(a, axis=op.axis, **kw)
                    ctx[op.outputs[0].key] = xp.take_along_axis(a, indices, op.axis)
            else:
                ctx[op.outputs[0].key] = xp.sort(a, axis=op.axis, **kw)


_AVAILABLE_KINDS = {'QUICKSORT', 'MERGESORT', 'HEAPSORT', 'STABLE'}


def _validate_sort_psrs_kinds(psrs_kinds):
    if psrs_kinds is not None:
        if isinstance(psrs_kinds, (list, tuple)):
            psrs_kinds = list(psrs_kinds)
            if len(psrs_kinds) != 3:
                raise ValueError('psrs_kinds should have 3 elements')
            for i, psrs_kind in enumerate(psrs_kinds):
                if psrs_kind is None:
                    if i < 2:
                        continue
                    else:
                        raise ValueError('3rd element of psrs_kinds '
                                         'should be specified')
                upper_psrs_kind = psrs_kind.upper()
                if upper_psrs_kind not in _AVAILABLE_KINDS:
                    raise ValueError('{} is an unrecognized kind '
                                     'in psrs_kinds'.format(psrs_kind))
        else:
            raise TypeError('psrs_kinds should be list or tuple')
    else:
        psrs_kinds = ['quicksort', 'mergesort', 'mergesort']
    return psrs_kinds


def _validate_sort_arguments(a, axis, kind, parallel_kind, psrs_kinds, order):
    a = astensor(a)
    if axis is None:
        a = a.flatten()
        axis = 0
    else:
        axis = validate_axis(a.ndim, axis)
    if kind is not None:
        raw_kind = kind
        kind = kind.upper()
        if kind not in _AVAILABLE_KINDS:
            # check kind
            raise ValueError('{} is an unrecognized kind of sort'.format(raw_kind))
    if parallel_kind is not None:
        raw_parallel_kind = parallel_kind
        parallel_kind = parallel_kind.upper()
        if parallel_kind not in {'PSRS'}:
            raise ValueError('{} is an unrecognized kind of '
                             'parallel sort'.format(raw_parallel_kind))

    order = validate_order(a.dtype, order)
    psrs_kinds = _validate_sort_psrs_kinds(psrs_kinds)
    return a, axis, kind, parallel_kind, psrs_kinds, order


def sort(a, axis=-1, kind=None, parallel_kind=None, psrs_kinds=None,
         order=None, return_index=False, **kw):
    r"""
    Return a sorted copy of a tensor.

    Parameters
    ----------
    a : array_like
        Tensor to be sorted.
    axis : int or None, optional
        Axis along which to sort. If None, the tensor is flattened before
        sorting. The default is -1, which sorts along the last axis.
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
        Sorting algorithm. The default is 'quicksort'. Note that both 'stable'
        and 'mergesort' use timsort or radix sort under the covers and, in general,
        the actual implementation will vary with data type. The 'mergesort' option
        is retained for backwards compatibility.
        Note that this argument would not take effect if `a` has more than
        1 chunk on the sorting axis.
    parallel_kind: {'PSRS'}, optional
        Parallel sorting algorithm, for the details, refer to:
        http://csweb.cs.wfu.edu/bigiron/LittleFE-PSRS/build/html/PSRSalgorithm.html
    psrs_kinds: list with 3 elements, optional
        Sorting algorithms during PSRS algorithm.
    order : str or list of str, optional
        When `a` is a tensor with fields defined, this argument specifies
        which fields to compare first, second, etc.  A single field can
        be specified as a string, and not all fields need be specified,
        but unspecified fields will still be used, in the order in which
        they come up in the dtype, to break ties.
    return_index: bool
        Return indices as well if True.

    Returns
    -------
    sorted_tensor : Tensor
        Tensor of the same type and shape as `a`.

    See Also
    --------
    Tensor.sort : Method to sort a tensor in-place.
    argsort : Indirect sort.
    lexsort : Indirect stable sort on multiple keys.
    searchsorted : Find elements in a sorted tensor.
    partition : Partial sort.

    Notes
    -----
    The various sorting algorithms are characterized by their average speed,
    worst case performance, work space size, and whether they are stable. A
    stable sort keeps items with the same key in the same relative
    order. The four algorithms implemented in NumPy have the following
    properties:

    =========== ======= ============= ============ ========
       kind      speed   worst case    work space   stable
    =========== ======= ============= ============ ========
    'quicksort'    1     O(n^2)            0          no
    'heapsort'     3     O(n*log(n))       0          no
    'mergesort'    2     O(n*log(n))      ~n/2        yes
    'timsort'      2     O(n*log(n))      ~n/2        yes
    =========== ======= ============= ============ ========

    .. note:: The datatype determines which of 'mergesort' or 'timsort'
       is actually used, even if 'mergesort' is specified. User selection
       at a finer scale is not currently available.

    All the sort algorithms make temporary copies of the data when
    sorting along any but the last axis.  Consequently, sorting along
    the last axis is faster and uses less space than sorting along
    any other axis.

    The sort order for complex numbers is lexicographic. If both the real
    and imaginary parts are non-nan then the order is determined by the
    real parts except when they are equal, in which case the order is
    determined by the imaginary parts.

    quicksort has been changed to an introsort which will switch
    heapsort when it does not make enough progress. This makes its
    worst case O(n*log(n)).

    'stable' automatically choses the best stable sorting algorithm
    for the data type being sorted. It, along with 'mergesort' is
    currently mapped to timsort or radix sort depending on the
    data type. API forward compatibility currently limits the
    ability to select the implementation and it is hardwired for the different
    data types.

    Timsort is added for better performance on already or nearly
    sorted data. On random data timsort is almost identical to
    mergesort. It is now used for stable sort while quicksort is still the
    default sort if none is chosen. For details of timsort, refer to
    `CPython listsort.txt <https://github.com/python/cpython/blob/3.7/Objects/listsort.txt>`_.
    'mergesort' and 'stable' are mapped to radix sort for integer data types. Radix sort is an
    O(n) sort instead of O(n log n).

    Examples
    --------
    >>> import mars.tensor as mt
    >>> a = mt.array([[1,4],[3,1]])
    >>> mt.sort(a).execute()                # sort along the last axis
    array([[1, 4],
           [1, 3]])
    >>> mt.sort(a, axis=None).execute()     # sort the flattened tensor
    array([1, 1, 3, 4])
    >>> mt.sort(a, axis=0).execute()        # sort along the first axis
    array([[1, 1],
           [3, 4]])

    Use the `order` keyword to specify a field to use when sorting a
    structured array:

    >>> dtype = [('name', 'S10'), ('height', float), ('age', int)]
    >>> values = [('Arthur', 1.8, 41), ('Lancelot', 1.9, 38),
    ...           ('Galahad', 1.7, 38)]
    >>> a = mt.array(values, dtype=dtype)       # create a structured tensor
    >>> mt.sort(a, order='height').execute()                # doctest: +SKIP
    array([('Galahad', 1.7, 38), ('Arthur', 1.8, 41),
           ('Lancelot', 1.8999999999999999, 38)],
          dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])

    Sort by age, then height if ages are equal:

    >>> mt.sort(a, order=['age', 'height']).execute()       # doctest: +SKIP
    array([('Galahad', 1.7, 38), ('Lancelot', 1.8999999999999999, 38),
           ('Arthur', 1.8, 41)],
          dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])
    """
    need_align = kw.pop('need_align', None)
    if len(kw) > 0:
        raise TypeError('sort() got an unexpected keyword '
                        'argument \'{}\''.format(next(iter(kw))))
    a, axis, kind, parallel_kind, psrs_kinds, order = \
        _validate_sort_arguments(a, axis, kind, parallel_kind, psrs_kinds, order)
    op = TensorSort(axis=axis, kind=kind, parallel_kind=parallel_kind,
                    order=order, psrs_kinds=psrs_kinds, need_align=need_align,
                    return_value=True, return_indices=return_index,
                    dtype=a.dtype, gpu=a.op.gpu)
    return op(a)
