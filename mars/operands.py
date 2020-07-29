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


import weakref
from copy import deepcopy
from enum import Enum
from typing import TypeVar, Union, List

import numpy as np

from . import opcodes as OperandDef
from .context import RunningMode, get_context
from .core import Entity, Chunk, Tileable, AttributeAsDictKey, ExecutableTuple, \
    FuseChunkData, FuseChunk, ObjectChunkData, ObjectChunk, ObjectData, Object
from .serialize import SerializableMetaclass, ValueType, ProviderType, IdentityField, \
    ListField, DictField, Int32Field, BoolField, StringField
from .tiles import NotSupportTile
from .utils import AttributeDict, to_str, calc_data_size, is_eager_mode, calc_object_overhead


operand_type_to_oprand_cls = {}
OP_TYPE_KEY = '_op_type_'
OP_MODULE_KEY = '_op_module_'
T = TypeVar('T')


class OperandMetaclass(SerializableMetaclass):
    def __new__(mcs, name, bases, kv):
        cls = super().__new__(mcs, name, bases, kv)

        for base in bases:
            if OP_TYPE_KEY not in kv and hasattr(base, OP_TYPE_KEY):
                kv[OP_TYPE_KEY] = getattr(base, OP_TYPE_KEY)
            if OP_MODULE_KEY not in kv and hasattr(base, OP_MODULE_KEY):
                kv[OP_MODULE_KEY] = getattr(base, OP_MODULE_KEY)

        if kv.get(OP_TYPE_KEY) is not None and kv.get(OP_MODULE_KEY) is not None:
            # common operand can be inherited for different modules, like tensor or dataframe, so forth
            operand_type_to_oprand_cls[kv[OP_MODULE_KEY], kv[OP_TYPE_KEY]] = cls

        return cls


class Operand(AttributeAsDictKey, metaclass=OperandMetaclass):
    """
    Operand base class. All operands should have a type, which can be Add, Subtract etc.
    `sparse` indicates that if the operand is applied on a sparse tensor/chunk.
    `gpu` indicates that if the operand should be executed on the GPU.
    `device`, 0 means the CPU, otherwise means the GPU device.
    Operand can have inputs and outputs
    which should be the :class:`mars.tensor.core.TensorData`, :class:`mars.tensor.core.ChunkData` etc.
    """
    __slots__ = '__weakref__',
    attr_tag = 'attr'
    _init_update_key_ = False

    _op_id = IdentityField('type')

    _sparse = BoolField('sparse')
    _gpu = BoolField('gpu')
    _device = Int32Field('device')
    # worker to execute, only work for chunk op,
    # if specified, the op should be executed on the specified worker
    # only work for those operand that has no input
    _expect_worker = StringField('expect_worker')
    # will this operand create a view of input data or not
    _create_view = BoolField('create_view')

    _inputs = ListField('inputs', ValueType.key)
    _prepare_inputs = ListField('prepare_inputs', ValueType.bool)
    _outputs = ListField('outputs', ValueType.key, weak_ref=True)

    _stage = Int32Field('stage', on_serialize=lambda s: s.value if s is not None else s,
                        on_deserialize=lambda n: OperandStage(n) if n is not None else n)

    _extra_params = DictField('extra_params', key_type=ValueType.string, on_deserialize=AttributeDict)

    def __new__(cls, *args, **kwargs):
        if '_op_id' in kwargs and kwargs['_op_id']:
            op_id = kwargs['_op_id']
            module, tp = op_id.rsplit('.', 1)
            cls = operand_type_to_oprand_cls[module, int(tp)]
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        extras = AttributeDict((k, kwargs.pop(k)) for k in set(kwargs) - set(self.__slots__))
        kwargs['_extra_params'] = kwargs.pop('_extra_params', extras)
        super().__init__(*args, **kwargs)
        if hasattr(self, OP_MODULE_KEY) and hasattr(self, OP_TYPE_KEY):
            self._op_id = '{0}.{1}'.format(getattr(self, OP_MODULE_KEY), getattr(self, OP_TYPE_KEY))

    def __repr__(self):
        if self.stage is None:
            return '{0} <key={1}>'.format(type(self).__name__, self.key)
        else:
            return '{0} <key={1}, stage={2}>'.format(type(self).__name__, self.key, self.stage.name)

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from .serialize.protos.operand_pb2 import OperandDef
            return OperandDef
        return super().cls(provider)

    @property
    def inputs(self) -> List[Union[Chunk, Tileable]]:
        return getattr(self, '_inputs', None)

    @inputs.setter
    def inputs(self, vals):
        self._set_inputs(vals)

    @property
    def outputs(self) -> List[Union[Chunk, Tileable]]:
        outputs = getattr(self, '_outputs', None)
        if outputs:
            return [ref() for ref in outputs]

    @outputs.setter
    def outputs(self, outputs):
        self._attach_outputs(*outputs)

    @property
    def output_limit(self):
        return 1

    @property
    def retryable(self) -> bool:
        return True

    def get_dependent_data_keys(self):
        return [dep.key for dep in self.inputs or ()]

    @property
    def gpu(self):
        return getattr(self, '_gpu', False)

    @property
    def device(self):
        return getattr(self, '_device', None)

    @property
    def create_view(self):
        return getattr(self, '_create_view', False)

    @property
    def expect_worker(self):
        return getattr(self, '_expect_worker', None)

    @property
    def prepare_inputs(self):
        val = getattr(self, '_prepare_inputs', None)
        if not val:
            return [True] * len(self.inputs or ())
        return val

    @property
    def stage(self) -> Union[None, "OperandStage"]:
        return getattr(self, '_stage', None)

    @property
    def extra_params(self):
        return self._extra_params

    @extra_params.setter
    def extra_params(self, extra_params):
        self._extra_params = extra_params

    @property
    def sparse(self) -> bool:
        return getattr(self, '_sparse', False)

    def is_sparse(self) -> bool:
        return getattr(self, '_sparse', False) or False

    issparse = is_sparse

    def is_gpu(self) -> bool:
        return getattr(self, '_gpu', False) or False

    @classmethod
    def _get_entity_data(cls, entity):
        if isinstance(entity, Entity):
            return entity.data
        return entity

    @classmethod
    def _get_inputs_data(cls, inputs):
        return [cls._get_entity_data(inp) for inp in inputs]

    def _set_inputs(self, inputs):
        if inputs is not None:
            inputs = self._get_inputs_data(inputs)
        if hasattr(self, 'check_inputs'):
            self.check_inputs(inputs)
        setattr(self, '_inputs', inputs)

    def _attach_outputs(self, *outputs):
        self._outputs = tuple(weakref.ref(self._get_entity_data(o)) if o is not None else o
                              for o in outputs)

        if len(self._outputs) > self.output_limit:
            raise ValueError("Outputs' size exceeds limitation")

    def copy(self: T) -> T:
        new_op = super().copy()
        new_op.outputs = []
        new_op.extra_params = deepcopy(self.extra_params)
        return new_op

    def on_output_modify(self, new_output):
        # when `create_view` is True, if the output is modified,
        # the modification should be set back to the input.
        # This function is for this sort of usage.
        # Remember, if `create_view` is False, this function should take no effect.
        raise NotImplementedError

    def on_input_modify(self, new_input):
        # when `create_view` is True, if the input is modified,
        # this function could be used to respond the modification.
        # Remember, if `create_view` is False, this function should take no effect.
        raise NotImplementedError


class TileableOperandMixin(object):
    __slots__ = ()

    def check_inputs(self, inputs):
        pass

    @classmethod
    def _check_if_gpu(cls, inputs):
        if inputs is not None and \
                len([inp for inp in inputs
                     if inp is not None and getattr(inp, 'op', None) is not None]) > 0:
            if all(inp.op.gpu is True for inp in inputs):
                return True
            elif all(inp.op.gpu is False for inp in inputs):
                return False

    def _create_chunk(self, output_idx, index, **kw):
        raise NotImplementedError

    def _new_chunks(self, inputs, kws=None, **kw):
        output_limit = kw.pop('output_limit', None)
        if output_limit is None:
            output_limit = getattr(self, 'output_limit')

        self.check_inputs(inputs)
        getattr(self, '_set_inputs')(inputs)
        if getattr(self, '_gpu', None) is None:
            self._gpu = self._check_if_gpu(self._inputs)
        if getattr(self, '_key', None) is None:
            getattr(self, '_update_key')()

        chunks = []
        for j in range(output_limit):
            create_chunk_kw = kw.copy()
            if kws:
                create_chunk_kw.update(kws[j])
            index = create_chunk_kw.pop('index', None)
            chunk = self._create_chunk(j, index, **create_chunk_kw)
            chunks.append(chunk)

        setattr(self, 'outputs', chunks)
        if len(chunks) > 1:
            # for each output chunk, hold the reference to the other outputs
            # so that either no one or everyone are gc collected
            for j, t in enumerate(chunks):
                t.data._siblings = [c.data for c in chunks[:j] + chunks[j + 1:]]
        return chunks

    def new_chunks(self, inputs, kws=None, **kwargs):
        """
        Create chunks.
        A chunk is a node in a fine grained graph, all the chunk objects are created by
        calling this function, it happens mostly in tiles.
        The generated chunks will be set as this operand's outputs and each chunk will
        hold this operand as it's op.
        :param inputs: input chunks
        :param kws: kwargs for each output
        :param kwargs: common kwargs for all outputs

        .. note::
            It's a final method, do not override.
            Override the method `_new_chunks` if needed.
        """
        return self._new_chunks(inputs, kws=kws, **kwargs)

    def new_chunk(self, inputs, kws=None, **kw):
        if getattr(self, 'output_limit') != 1:
            raise TypeError('cannot new chunk with more than 1 outputs')

        return self.new_chunks(inputs, kws=kws, **kw)[0]

    def _create_tileable(self, output_idx, **kw):
        raise NotImplementedError

    def _new_tileables(self, inputs, kws=None, **kw):
        output_limit = kw.pop('output_limit', None)
        if output_limit is None:
            output_limit = getattr(self, 'output_limit')

        self.check_inputs(inputs)
        getattr(self, '_set_inputs')(inputs)
        if getattr(self, '_gpu', None) is None:
            self._gpu = self._check_if_gpu(self._inputs)
        if getattr(self, '_key', None) is None:
            getattr(self, '_update_key')()  # update key when inputs are set

        tileables = []
        for j in range(output_limit):
            create_tensor_kw = kw.copy()
            if kws:
                create_tensor_kw.update(kws[j])
            tileable = self._create_tileable(j, **create_tensor_kw)
            tileables.append(tileable)

        setattr(self, 'outputs', tileables)
        if len(tileables) > 1:
            # for each output tileable, hold the reference to the other outputs
            # so that either no one or everyone are gc collected
            for j, t in enumerate(tileables):
                t.data._siblings = [tileable.data for tileable in tileables[:j] + tileables[j + 1:]]
        return tileables

    def new_tileables(self, inputs, kws=None, **kw):
        """
        Create tileable objects(Tensors or DataFrames).
        This is a base function for create tileable objects like tensors or dataframes,
        it will be called inside the `new_tensors` and `new_dataframes`.
        If eager mode is on, it will trigger the execution after tileable objects are created.
        :param inputs: input tileables
        :param kws: kwargs for each output
        :param kw: common kwargs for all outputs

        .. note::
            It's a final method, do not override.
            Override the method `_new_tileables` if needed.
        """

        tileables = self._new_tileables(inputs, kws=kws, **kw)
        if is_eager_mode():
            ExecutableTuple(tileables).execute(fetch=False)
        return tileables

    def new_tileable(self, inputs, kws=None, **kw):
        if getattr(self, 'output_limit') != 1:
            raise TypeError('cannot new chunk with more than 1 outputs')

        return self.new_tileables(inputs, kws=kws, **kw)[0]

    @classmethod
    def tile(cls, op):
        raise NotImplementedError

    @classmethod
    def execute(cls, ctx, op):
        raise NotImplementedError

    @classmethod
    def estimate_size(cls, ctx, op):
        exec_size = 0
        outputs = op.outputs
        if all(not c.is_sparse() and hasattr(c, 'nbytes') and not np.isnan(c.nbytes) for c in outputs):
            for out in outputs:
                ctx[out.key] = (out.nbytes, out.nbytes)

        all_overhead = 0
        for inp in op.inputs or ():
            try:
                if isinstance(inp.op, FetchShuffle):
                    keys_and_shapes = inp.extra_params.get('_shapes', dict()).items()
                else:
                    keys_and_shapes = [(inp.key, getattr(inp, 'shape', None))]

                # execution size of a specific data chunk may be
                # larger than stored type due to objects
                for key, shape in keys_and_shapes:
                    overhead = calc_object_overhead(inp, shape)
                    all_overhead += overhead
                    exec_size += ctx[key][0] + overhead
            except KeyError:
                if not op.sparse:
                    inp_size = calc_data_size(inp)
                    if not np.isnan(inp_size):
                        exec_size += inp_size
        exec_size = int(exec_size)

        total_out_size = 0
        chunk_sizes = dict()
        for out in outputs:
            try:
                if not out.is_sparse():
                    chunk_size = calc_data_size(out) + all_overhead // len(outputs)
                else:
                    chunk_size = exec_size
                if np.isnan(chunk_size):
                    raise TypeError
                chunk_sizes[out.key] = chunk_size
                total_out_size += chunk_size
            except (AttributeError, TypeError, ValueError):
                pass

        exec_size = max(exec_size, total_out_size)
        for out in outputs:
            if out.key in ctx:
                continue
            if out.key in chunk_sizes:
                store_size = chunk_sizes[out.key]
            else:
                store_size = max(exec_size // len(outputs),
                                 total_out_size // max(len(chunk_sizes), 1))
            try:
                if out.is_sparse():
                    max_sparse_size = out.nbytes + np.dtype(np.int64).itemsize * np.prod(out.shape) * out.ndim
                else:
                    max_sparse_size = np.nan
            except TypeError:  # pragma: no cover
                max_sparse_size = np.nan
            if not np.isnan(max_sparse_size):
                store_size = min(store_size, max_sparse_size)
            ctx[out.key] = (store_size, exec_size // len(outputs))

    @classmethod
    def concat_tileable_chunks(cls, tileable):
        raise NotImplementedError

    @classmethod
    def create_tileable_from_chunks(cls, chunks, inputs=None, **kw):
        raise NotImplementedError

    def get_fetch_op_cls(self, obj):
        raise NotImplementedError

    def get_fuse_op_cls(self, obj):
        raise NotImplementedError


class HasInput(Operand):
    __slots__ = ()

    @property
    def input(self):
        return self._input

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]


class VirtualOperand(Operand):
    def get_dependent_data_keys(self):
        return []


class MapReduceOperand(Operand):
    _shuffle_key = StringField('shuffle_key', on_serialize=to_str)

    @property
    def shuffle_key(self):
        return getattr(self, '_shuffle_key', None)

    def get_dependent_data_keys(self):
        if self.stage == OperandStage.reduce:
            inputs = self.inputs or ()
            deps = []
            for inp in inputs:
                if isinstance(inp.op, (ShuffleProxy, FetchShuffle)):
                    deps.extend([(chunk.key, self._shuffle_key) for chunk in inp.inputs or ()])
                else:
                    deps.append(inp.key)
            return deps
        return super().get_dependent_data_keys()


class ShuffleProxy(VirtualOperand):
    _op_type_ = OperandDef.SHUFFLE_PROXY
    _broadcaster = True


class Fetch(Operand):
    _op_type_ = OperandDef.FETCH

    _to_fetch_key = StringField('to_fetch_key', on_serialize=to_str)

    @property
    def to_fetch_key(self):
        return self._to_fetch_key


class FetchMixin(TileableOperandMixin):
    def check_inputs(self, inputs):
        # no inputs
        if inputs and len(inputs) > 0:
            raise ValueError("%s has no inputs" % type(self).__name__)

    @classmethod
    def tile(cls, op):
        raise NotImplementedError('Fetch tile cannot be handled by operand itself')

    @classmethod
    def execute(cls, ctx, op):
        # fetch op need to do nothing
        pass


class Fuse(Operand):
    _op_type_ = OperandDef.FUSE

    _operands = ListField('operands', ValueType.key)

    @property
    def operands(self):
        return self._operands


class FuseChunkMixin(object):
    __slots__ = ()

    def _create_chunk(self, output_idx, index, **kw):
        data = FuseChunkData(_index=index, _op=self, **kw)
        return FuseChunk(data)

    @classmethod
    def tile(cls, op):
        raise NotSupportTile('FuseChunk is a chunk operand which does not support tile')

    def __call__(self, fuse_chunks):
        head_chunk = fuse_chunks[0]
        tail_chunk = fuse_chunks[-1]
        setattr(self, '_operands', [c.op for c in fuse_chunks])
        return self.new_chunk(head_chunk.inputs, shape=tail_chunk.shape, order=tail_chunk.order,
                              _composed=fuse_chunks, _key=tail_chunk.key)


class FetchShuffle(Operand):
    _op_type_ = OperandDef.FETCH_SHUFFLE

    _to_fetch_keys = ListField('to_fetch_keys', ValueType.string,
                               on_serialize=lambda v: [to_str(i) for i in v])
    _to_fetch_idxes = ListField('to_fetch_idxes', ValueType.tuple(ValueType.uint64))

    @property
    def to_fetch_keys(self):
        return self._to_fetch_keys

    @property
    def to_fetch_idxes(self):
        return self._to_fetch_idxes


class OperandStage(Enum):
    map = 0
    reduce = 1
    combine = 2
    agg = 3


class ObjectOperand(Operand):
    pass


class ObjectOperandMixin(TileableOperandMixin):
    def _create_chunk(self, output_idx, index, **kw):
        if 'i' not in kw:
            kw['i'] = output_idx
        data = ObjectChunkData(op=self, index=index, **kw)
        return ObjectChunk(data)

    def _create_tileable(self, output_idx, **kw):
        if 'i' not in kw:
            kw['i'] = output_idx
        data = ObjectData(op=self, **kw)
        return Object(data)

    def get_fetch_op_cls(self, obj):
        return ObjectFetch

    def get_fuse_op_cls(self, obj):
        return ObjectFuseChunk


class ObjectFetchMixin(ObjectOperandMixin, FetchMixin):
    __slots__ = ()


class ObjectFuseChunkMixin(FuseChunkMixin, ObjectOperandMixin):
    __slots__ = ()


class ObjectFuseChunk(ObjectFuseChunkMixin, Fuse):
    pass


class ObjectFetch(Fetch, ObjectFetchMixin):
    def __init__(self, to_fetch_key=None, **kw):
        super().__init__(_to_fetch_key=to_fetch_key, **kw)

    def _new_chunks(self, inputs, kws=None, **kw):
        if '_key' in kw and self._to_fetch_key is None:
            self._to_fetch_key = kw['_key']
        return super()._new_chunks(inputs, kws=kws, **kw)

    def _new_tileables(self, inputs, kws=None, **kw):
        if '_key' in kw and self._to_fetch_key is None:
            self._to_fetch_key = kw['_key']
        return super()._new_tileables(inputs, kws=kws, **kw)


class MergeDictOperand(ObjectOperand, ObjectOperandMixin):
    _merge = BoolField('merge')

    def __init__(self, merge=None, **kw):
        super().__init__(_merge=merge, _object_type=Object, **kw)

    @property
    def merge(self):
        return self._merge

    @classmethod
    def concat_tileable_chunks(cls, tileable):
        assert not tileable.is_coarse()

        op = cls(merge=True)
        chunk = cls(merge=True).new_chunk(tileable.chunks)
        return op.new_tileable([tileable], chunks=[chunk], nsplits=((1,),))

    @classmethod
    def execute(cls, ctx, op):
        assert op.merge
        inputs = [ctx[inp.key] for inp in op.inputs]
        ctx[op.outputs[0].key] = next(inp for inp in inputs if inp)


class SuccessorsExclusive(ObjectOperandMixin, VirtualOperand):
    _op_module_ = 'core'
    _op_type_ = OperandDef.SUCCESSORS_EXCLUSIVE

    def _new_chunks(self, inputs, kws=None, **kw):
        ctx = get_context()
        if ctx.running_mode == RunningMode.local:
            # set inputs to None if local
            inputs = None
        return super()._new_chunks(inputs, kws=kws, **kw)

    @classmethod
    def execute(cls, ctx, op):
        # only for local
        if ctx.running_mode == RunningMode.local:
            ctx[op.outputs[0].key] = ctx.create_lock()
        else:  # pragma: no cover
            raise RuntimeError('Cannot execute SuccessorsExclusive '
                               'which is a virtual operand '
                               'for the distributed runtime')
