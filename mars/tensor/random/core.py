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
from collections.abc import Iterable
from contextlib import contextmanager

import numpy as np

from ...config import options
from ...serialize import ValueType, TupleField, Int32Field
from ...utils import tokenize
from ..core import TENSOR_TYPE, CHUNK_TYPE
from ..utils import decide_chunk_sizes, gen_random_seeds, broadcast_shape
from ..array_utils import array_module, device
from ..operands import TensorOperand, TensorMapReduceOperand, TensorOperandMixin
from ..datasource import tensor as astensor
from ..base import broadcast_to


class RandomState(object):
    def __init__(self, seed=None):
        self._random_state = np.random.RandomState(seed=seed)

    def seed(self, seed=None):
        """
        Seed the generator.

        This method is called when `RandomState` is initialized. It can be
        called again to re-seed the generator. For details, see `RandomState`.

        Parameters
        ----------
        seed : int or 1-d array_like, optional
            Seed for `RandomState`.
            Must be convertible to 32 bit unsigned integers.

        See Also
        --------
        RandomState
        """
        self._random_state.seed(seed=seed)

    def to_numpy(self):
        return self._random_state

    @classmethod
    def from_numpy(cls, np_random_state):
        state = RandomState()
        state._random_state = np_random_state
        return state

    @classmethod
    def _handle_size(cls, size):
        if size is None:
            return size
        try:
            return tuple(int(s) for s in size)
        except TypeError:
            return size,


_random_state = RandomState()


def handle_array(arg):
    if not isinstance(arg, TENSOR_TYPE):
        if not isinstance(arg, Iterable):
            return arg

        arg = np.asarray(arg)
        return arg[(0,) * max(1, arg.ndim)]
    elif hasattr(arg, 'op') and hasattr(arg.op, 'data'):
        return arg.op.data[(0,) * max(1, arg.ndim)]

    return np.empty((0,), dtype=arg.dtype)


class TensorRandomOperandMixin(TensorOperandMixin):
    __slots__ = ()

    @classmethod
    def tile(cls, op):
        tensor = op.outputs[0]
        chunk_size = tensor.extra_params.raw_chunk_size or options.chunk_size
        nsplits = decide_chunk_sizes(tensor.shape, chunk_size, tensor.dtype.itemsize)
        fields = getattr(op, '_input_fields_', [])
        to_one_chunk_fields = set(getattr(op, '_into_one_chunk_fields_', list()))

        new_inputs = []
        changed = False
        for field in fields:
            t = getattr(op, field)
            if not isinstance(t, TENSOR_TYPE):
                continue

            if field not in to_one_chunk_fields:
                t_nsplits = nsplits
            else:
                t_nsplits = t.shape  # into 1 chunk
            rechunked = t.rechunk(t_nsplits)
            if rechunked is not t:
                rechunked._inplace_tile()
                changed = True
                new_inputs.append(rechunked)
            else:
                new_inputs.append(t)
        if changed:
            op.inputs = new_inputs

        idxes = list(itertools.product(*[range(len(s)) for s in nsplits]))
        seeds = gen_random_seeds(len(idxes), op.state)

        out_chunks = []
        for seed, idx, shape in zip(seeds, idxes, itertools.product(*nsplits)):
            inputs = []
            for inp in op.inputs:
                if len(inp.chunks) == 1:
                    inputs.append(inp.chunks[0])
                else:
                    inputs.append(inp.cix[idx])
            try:
                s = len(tuple(op.size))
                size = shape[:s]
            except TypeError:
                if op.size is None:
                    size = None
                else:
                    size = shape[:1]
            except AttributeError:
                size = shape

            chunk_op = op.copy().reset_key()
            chunk_op._seed = int(seed)
            chunk_op._state = None
            chunk_op._size = size
            out_chunk = chunk_op.new_chunk(inputs, shape=shape, index=idx,
                                           order=tensor.order)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, tensor.shape, order=tensor.order,
                                  chunks=out_chunks, nsplits=nsplits, **tensor.extra_params)

    @classmethod
    def execute(cls, ctx, op):
        xp = array_module(op.gpu)
        if xp is np:
            device_id = -1
        else:
            device_id = op.device or 0
        get_val = lambda x: ctx[x.key] if isinstance(x, CHUNK_TYPE) else x

        with device(device_id):
            rs = xp.random.RandomState(op.seed)

            method_name = getattr(cls, '_func_name')
            try:
                if method_name in ('rand', 'randn'):
                    try:
                        res = getattr(rs, method_name)(*op.size, dtype=op.dtype)
                    except TypeError:
                        res = getattr(rs, method_name)(*op.size)
                elif method_name == 'randint':
                    try:
                        res = rs.randint(get_val(op.low), get_val(op.high), size=op.size,
                                         dtype=op.dtype)
                    except TypeError:
                        res = rs.randint(get_val(op.low), get_val(op.high), size=op.size)
                else:
                    try:
                        res = getattr(rs, method_name)(*(get_val(getattr(op, arg)) for arg in op.args),
                                                       dtype=op.dtype)
                    except TypeError:
                        res = getattr(rs, method_name)(*(get_val(getattr(op, arg)) for arg in op.args))
                if hasattr(res, 'dtype') and res.dtype != op.dtype:
                    res = res.astype(op.dtype, copy=False)
                if xp is not np:
                    ctx[op.outputs[0].key] = xp.asarray(res)
                else:
                    ctx[op.outputs[0].key] = res
            except AttributeError:
                if xp is not np:
                    # cupy cannot generate data, fallback to numpy
                    rs = np.random.RandomState(op.seed)
                    if method_name in ('rand', 'randn'):
                        res = getattr(rs, method_name)(*op.size)
                    else:
                        res = getattr(rs, method_name)(*(get_val(getattr(op, arg)) for arg in op.args))
                    if res.dtype != op.dtype:
                        res = res.astype(op.dtype, copy=False)
                    ctx[op.outputs[0].key] = xp.asarray(res)
                else:
                    raise

    def _calc_shape(self, shapes):
        shapes = list(shapes)
        if getattr(self, '_size', None) is not None:
            shapes.append(getattr(self, '_size'))
        return broadcast_shape(*shapes)

    @classmethod
    def _handle_arg(cls, arg, chunk_size):
        if isinstance(arg, (list, np.ndarray)):
            arg = astensor(arg, chunk_size=chunk_size)

        return arg

    @contextmanager
    def _get_inputs_shape_by_given_fields(self, inputs, shape, raw_chunk_size=None, tensor=True):
        fields = getattr(self, '_input_fields_', [])
        to_one_chunk_fields = set(getattr(self, '_into_one_chunk_fields_', list()))

        field_to_obj = dict()
        to_broadcast_shapes = []
        if fields:
            if getattr(self, fields[0], None) is None:
                # create from beginning
                for field, val in zip(fields, inputs):
                    if field not in to_one_chunk_fields:
                        if isinstance(val, list):
                            val = np.asarray(val)
                        if tensor:
                            val = self._handle_arg(val, raw_chunk_size)
                    if isinstance(val, TENSOR_TYPE + CHUNK_TYPE):
                        field_to_obj[field] = val
                        if field not in to_one_chunk_fields:
                            to_broadcast_shapes.append(val.shape)
                    setattr(self, field, val)
            else:
                inputs_iter = iter(inputs)
                for field in fields:
                    if isinstance(getattr(self, field), TENSOR_TYPE + CHUNK_TYPE):
                        field_to_obj[field] = next(inputs_iter)

        if tensor:
            if shape is None:
                shape = self._calc_shape(to_broadcast_shapes)

            for field, inp in field_to_obj.items():
                if field not in to_one_chunk_fields:
                    field_to_obj[field] = broadcast_to(inp, shape)

        yield [field_to_obj[f] for f in fields if f in field_to_obj], shape

        inputs_iter = iter(getattr(self, '_inputs'))
        for field in fields:
            if field in field_to_obj:
                setattr(self, field, next(inputs_iter))

    @classmethod
    def _get_shape(cls, kws, kw):
        if kw.get('shape') is not None:
            return kw.get('shape')
        elif kws is not None and len(kws) > 0:
            return kws[0].get('shape')

    def _new_tileables(self, inputs, kws=None, **kw):
        raw_chunk_size = kw.get('chunk_size', None)
        shape = self._get_shape(kws, kw)
        with self._get_inputs_shape_by_given_fields(inputs, shape, raw_chunk_size, True) as (inputs, shape):
            kw['shape'] = shape
            return super()._new_tileables(inputs, kws=kws, **kw)

    def _new_chunks(self, inputs, kws=None, **kw):
        shape = self._get_shape(kws, kw)
        with self._get_inputs_shape_by_given_fields(inputs, shape, None, False) as (inputs, shape):
            kw['shape'] = shape
            return super()._new_chunks(inputs, kws=kws, **kw)


def _on_serialize_random_state(rs):
    return rs.get_state() if rs is not None else None


def _on_deserialize_random_state(tup):
    rs = np.random.RandomState()
    rs.set_state(tup)
    return rs


class TensorSeedOperandMixin(object):
    @property
    def state(self):
        return getattr(self, '_state', None)

    @property
    def seed(self):
        return getattr(self, '_seed', None)

    @property
    def args(self):
        return [slot for slot in self.__slots__
                if slot not in set(TensorRandomOperand.__slots__)]

    def _update_key(self):
        self._key = tokenize(type(self).__name__,
                             *tuple(getattr(self, k, None) for k in self._keys_))
        return self


class TensorRandomOperand(TensorSeedOperandMixin, TensorOperand):
    _state = TupleField('state', on_serialize=_on_serialize_random_state,
                        on_deserialize=_on_deserialize_random_state)
    _seed = Int32Field('seed')


class TensorRandomMapReduceOperand(TensorSeedOperandMixin, TensorMapReduceOperand):
    _state = TupleField('state', on_serialize=_on_serialize_random_state,
                        on_deserialize=_on_deserialize_random_state)
    _seed = Int32Field('seed')


class TensorDistribution(TensorRandomOperand):
    _size = TupleField('size', ValueType.int64)

    @property
    def size(self):
        return self._size

    @classmethod
    def execute(cls, ctx, op):
        xp = array_module(op.gpu)
        if xp is np:
            device_id = -1
        else:
            device_id = op.device or 0

        with device(device_id):
            rs = xp.random.RandomState(op.seed)

            args = []
            for k in op.args:
                val = getattr(op, k, None)
                if isinstance(val, CHUNK_TYPE):
                    args.append(ctx[val.key])
                else:
                    args.append(val)

            method_name = getattr(cls, '_func_name')
            try:
                res = getattr(rs, method_name)(*args)
                if xp is not np:
                    ctx[op.outputs[0].key] = xp.asarray(res)
                else:
                    ctx[op.outputs[0].key] = res
            except AttributeError:
                if xp is not np:
                    # cupy cannot generate, fall back to numpy
                    rs = np.random.RandomState(op.seed)
                    res = getattr(rs, method_name)(*args)
                    ctx[op.outputs[0].key] = xp.asarray(res)
                else:
                    raise


class TensorSimpleRandomData(TensorRandomOperand):
    _size = TupleField('size', ValueType.int64)

    @property
    def size(self):
        return self._size
