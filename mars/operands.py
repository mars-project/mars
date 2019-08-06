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


from copy import deepcopy
import weakref

from .compat import six
from .serialize import SerializableMetaclass, ValueType, ProviderType, \
    IdentityField, ListField, DictField, Int32Field, BoolField, StringField
from .core import Entity, AttributeAsDictKey
from .utils import AttributeDict, to_str
from . import opcodes as OperandDef


operand_type_to_oprand_cls = {}
OP_TYPE_KEY = '_op_type_'
OP_MODULE_KEY = '_op_module_'


class OperandMetaclass(SerializableMetaclass):
    def __new__(mcs, name, bases, kv):
        cls = super(OperandMetaclass, mcs).__new__(mcs, name, bases, kv)

        for base in bases:
            if OP_TYPE_KEY not in kv and hasattr(base, OP_TYPE_KEY):
                kv[OP_TYPE_KEY] = getattr(base, OP_TYPE_KEY)
            if OP_MODULE_KEY not in kv and hasattr(base, OP_MODULE_KEY):
                kv[OP_MODULE_KEY] = getattr(base, OP_MODULE_KEY)

        if kv.get(OP_TYPE_KEY) is not None and kv.get(OP_MODULE_KEY) is not None:
            # common operand can be inherited for different modules, like tensor or dataframe, so forth
            operand_type_to_oprand_cls[kv[OP_MODULE_KEY], kv[OP_TYPE_KEY]] = cls

        return cls


class Operand(six.with_metaclass(OperandMetaclass, AttributeAsDictKey)):
    """
    Operand base class. All operands should have a type, which can be Add, Subtract etc.
    `sparse` indicates that if the operand is applied on a sparse tensor/chunk.
    `gpu` indicates that if the operand should be executed on the GPU.
    `device`, 0 means the CPU, otherwise means the GPU device.
    Operand can have inputs and outputs
    which should be the :class:`mars.tensor.core.TensorData`, :class:`mars.tensor.core.ChunkData` etc.
    """

    attr_tag = 'attr'
    _init_update_key_ = False

    _op_id = IdentityField('type')

    _sparse = BoolField('sparse')
    _gpu = BoolField('gpu')
    _device = Int32Field('device')
    # will this operand create a view of input data or not
    _create_view = BoolField('create_view')

    _inputs = ListField('inputs', ValueType.key)
    _outputs = ListField('outputs', ValueType.key, weak_ref=True)

    _extra_params = DictField('extra_params', key_type=ValueType.string, on_deserialize=AttributeDict)

    def __new__(cls, *args, **kwargs):
        if '_op_id' in kwargs and kwargs['_op_id']:
            op_id = kwargs['_op_id']
            module, tp = op_id.split('.', 1)
            cls = operand_type_to_oprand_cls[module, int(tp)]
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        extras = AttributeDict((k, kwargs.pop(k)) for k in set(kwargs) - set(self.__slots__))
        kwargs['_extra_params'] = kwargs.pop('_extra_params', extras)
        super(Operand, self).__init__(*args, **kwargs)
        if hasattr(self, OP_MODULE_KEY) and hasattr(self, OP_TYPE_KEY):
            self._op_id = '{0}.{1}'.format(getattr(self, OP_MODULE_KEY), getattr(self, OP_TYPE_KEY))

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from .serialize.protos.operand_pb2 import OperandDef
            return OperandDef
        return super(Operand, cls).cls(provider)

    @property
    def inputs(self):
        return getattr(self, '_inputs', None)

    @inputs.setter
    def inputs(self, vals):
        self._set_inputs(vals)

    @property
    def outputs(self):
        outputs = getattr(self, '_outputs', None)
        if outputs:
            return [ref() for ref in outputs]

    @outputs.setter
    def outputs(self, outputs):
        self._attach_outputs(*outputs)

    @property
    def output_limit(self):
        return 1

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
    def extra_params(self):
        return self._extra_params

    @extra_params.setter
    def extra_params(self, extra_params):
        self._extra_params = extra_params

    @property
    def sparse(self):
        return getattr(self, '_sparse', False)

    def is_sparse(self):
        return getattr(self, '_sparse', False) or False

    issparse = is_sparse

    def is_gpu(self):
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

    def copy(self):
        new_op = super(Operand, self).copy()
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


class HasInput(Operand):
    __slots__ = ()

    @property
    def input(self):
        return self._input

    def _set_inputs(self, inputs):
        super(HasInput, self)._set_inputs(inputs)
        self._input = self._inputs[0]


class VirtualOperand(Operand):
    def get_dependent_data_keys(self):
        return []


class ShuffleProxy(VirtualOperand):
    _op_type_ = OperandDef.SHUFFLE_PROXY
    _broadcaster = True


class ShuffleMap(Operand):
    pass


class ShuffleReduce(Operand):
    _shuffle_key = StringField('shuffle_key', on_serialize=to_str)

    @property
    def shuffle_key(self):
        return getattr(self, '_shuffle_key', None)

    def get_dependent_data_keys(self):
        inputs = self.inputs or ()
        return [(chunk.key, self._shuffle_key)
                for proxy in inputs for chunk in proxy.inputs or ()]


class Fetch(Operand):
    _op_type_ = OperandDef.FETCH

    _to_fetch_key = StringField('to_fetch_key', on_serialize=to_str)

    @property
    def to_fetch_key(self):
        return self._to_fetch_key


class Fuse(Operand):
    _op_type_ = OperandDef.FUSE

    _operands = ListField('operands', ValueType.key)

    @property
    def operands(self):
        return self._operands


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
