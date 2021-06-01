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
from functools import partial
from typing import List, Tuple, Dict, Type, Union

from ...serialization.serializables import SerializableMeta, FieldTypes, \
    BoolField, Int32Field, Float32Field, StringField, \
    ListField, DictField, ReferenceField
from ...serialization.core import Placeholder
from ...serialization.serializables.core import SerializableSerializer
from ...typing import OperandType
from ...utils import AttributeDict
from ..base import Base
from ..entity.core import Entity, EntityData
from ..entity.chunks import Chunk
from ..entity.tileables import Tileable
from ..entity.output_types import OutputType
from ..mode import enter_mode


class OperandMetaclass(SerializableMeta):
    def __new__(mcs,
                name: str,
                bases: Tuple[Type],
                properties: Dict):
        if '__call__' in properties:
            # if __call__ is specified for an operand,
            # make sure that entering user space
            properties['__call__'] = enter_mode(kernel=False)(properties['__call__'])

        return super().__new__(mcs, name, bases, properties)


class OperandStage(Enum):
    map = 0
    reduce = 1
    combine = 2
    agg = 3


class Operand(Base, metaclass=OperandMetaclass):
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
    _output_type_ = None

    sparse = BoolField('sparse', default=False)
    gpu = BoolField('gpu', default=None)
    device = Int32Field('device', default=None)
    # worker to execute, only work for chunk op,
    # if specified, the op should be executed on the specified worker
    # only work for those operand that has no input
    expect_worker = StringField('expect_worker', default=None)
    # will this operand create a view of input data or not
    create_view = BoolField('create_view', default=False)
    # will this operand be assigned a worker or not
    reassign_worker = BoolField('reassign_worker', default=False)
    stage = ReferenceField('stage', OperandStage, default=None)
    memory_scale = Float32Field('memory_scale', default=None)
    tileable_op_key = StringField('tileable_op_key', default=None)
    extra_params = DictField('extra_params', key_type=FieldTypes.string)

    _inputs = ListField('inputs', FieldTypes.reference(EntityData))
    _pure_depends = ListField('pure_depends', FieldTypes.bool)
    _outputs = ListField('outputs')
    _output_types = ListField('output_type', FieldTypes.reference(OutputType))

    def __init__(self: OperandType, *args, **kwargs):
        extras = AttributeDict((k, kwargs.pop(k)) for k in set(kwargs) - set(self._FIELDS))
        kwargs['extra_params'] = kwargs.pop('extra_params', extras)
        super().__init__(*args, **kwargs)

    def __repr__(self):
        if self.stage is None:
            return f'{type(self).__name__} <key={self.key}>'
        else:
            return f'{type(self).__name__} <key={self.key}, stage={self.stage.name}>'

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

    @property
    def inputs(self) -> List[Union[Chunk, Tileable]]:
        inputs = getattr(self, '_inputs', None)
        if not inputs:
            return list()
        return inputs

    @inputs.setter
    def inputs(self, vals):
        self._set_inputs(vals)

    @property
    def output_limit(self):
        return 1

    @property
    def pure_depends(self):
        val = getattr(self, '_pure_depends', None)
        if not val:
            return [False] * len(self.inputs or ())
        return val

    @property
    def output_types(self):
        return getattr(self, '_output_types', None)

    @output_types.setter
    def output_types(self, value):
        self._output_types = value

    def _attach_outputs(self, *outputs):
        self._outputs = [weakref.ref(self._get_entity_data(o)) if o is not None else o
                         for o in outputs]

        if len(self._outputs) > self.output_limit:
            raise ValueError("Outputs' size exceeds limitation")

    @property
    def outputs(self) -> List[Union[Chunk, Tileable]]:
        outputs = getattr(self, '_outputs', None)
        if outputs:
            return [ref() for ref in outputs]

    @outputs.setter
    def outputs(self, outputs):
        self._attach_outputs(*outputs)

    def is_sparse(self) -> bool:
        return self.sparse

    issparse = is_sparse

    def is_gpu(self) -> bool:
        return self.gpu

    @property
    def retryable(self) -> bool:
        return True

    def get_dependent_data_keys(self):
        return [dep.key for dep in self.inputs or ()]

    def _get_output_type(self, output_idx):
        if self.output_types:
            try:
                return self.output_types[output_idx]
            except IndexError:
                return self.output_types[0]
        else:
            return self._output_type_

    def copy(self: OperandType) -> OperandType:
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


class OperandSerializer(SerializableSerializer):
    serializer_name = 'operand'

    @classmethod
    def _get_tag_to_values(cls, obj: Operand):
        tag_to_values = super()._get_tag_to_values(obj)
        # outputs are weak-refs which are not pickle-able
        tag_to_values['outputs'] = \
            [out_ref() for out_ref in tag_to_values['outputs']]
        return tag_to_values

    def deserialize(self,
                    header: Dict,
                    buffers: List,
                    context: Dict) -> Operand:
        # convert outputs back to weak-refs
        operand: Operand = (yield from super().deserialize(header, buffers, context))
        for i, out in enumerate(operand._outputs):
            def cb(o, index):
                outputs = operand._outputs
                outputs[index] = weakref.ref(o)

                if len(outputs) > 1 and \
                        all(not isinstance(o, Placeholder) for o in outputs):
                    # all replaced
                    # add siblings for multiple outputs
                    outputs = operand.outputs
                    for j in range(len(outputs)):
                        outputs[j]._siblings = outputs[:j] + outputs[j + 1:]

            if isinstance(out, Placeholder):
                out.callbacks.append(partial(cb, index=i))
            else:
                cb(out, i)
        return operand


OperandSerializer.register(Operand)


class VirtualOperand(Operand):
    def get_dependent_data_keys(self):
        return []


class HasInput(Operand):
    __slots__ = ()

    @property
    def input(self):
        return self._input

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]
