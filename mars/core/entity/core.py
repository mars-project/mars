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

from ...serialization.serializables import Serializable, FieldTypes, \
    DictField, ReferenceField
from ...utils import AttributeDict
from ..base import Base


class EntityData(Base):
    __slots__ = '__weakref__', '_siblings'
    type_name = None

    # required fields
    _op = ReferenceField('op', 'mars.core.operand.base.Operand')
    # optional fields
    _extra_params = DictField('extra_params',
                              key_type=FieldTypes.string)

    def __init__(self, *args, **kwargs):
        extras = AttributeDict((k, kwargs.pop(k)) for k in set(kwargs) - set(self._FIELDS))
        kwargs['_extra_params'] = kwargs.pop('_extra_params', extras)
        super().__init__(*args, **kwargs)

    @property
    def op(self):
        return self._op

    @property
    def inputs(self):
        return self.op.inputs

    @inputs.setter
    def inputs(self, new_inputs):
        self.op.inputs = new_inputs

    def is_sparse(self):
        return self.op.is_sparse()

    issparse = is_sparse

    @property
    def extra_params(self):
        return self._extra_params

    def build_graph(self, **kw):
        from ..graph.builder.utils import build_graph

        return build_graph([self], **kw)

    def visualize(self, graph_attrs=None, node_attrs=None, **kw):
        from graphviz import Source

        g = self.build_graph(**kw)
        dot = g.to_dot(graph_attrs=graph_attrs, node_attrs=node_attrs,
                       result_chunk_keys={c.key for c in self.chunks})

        return Source(dot)


class Entity(Serializable):
    _allow_data_type_ = ()
    type_name = None

    _data = ReferenceField('data', EntityData)

    def __init__(self, data=None, **kw):
        super().__init__(_data=data, **kw)

    def __dir__(self):
        obj_dir = object.__dir__(self)
        if self._data is not None:
            obj_dir = sorted(set(dir(self._data) + obj_dir))
        return obj_dir

    def __str__(self):
        return self._data.__str__()

    def __repr__(self):
        return self._data.__repr__()

    def _check_data(self, data):
        if data is not None and not isinstance(data, self._allow_data_type_):
            raise TypeError(f'Expect {self._allow_data_type_}, got {type(data)}')

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        self._check_data(new_data)
        self._data = new_data

    def __copy__(self):
        return self.copy()

    def copy(self):
        return self.copy_to(type(self)(None))

    def copy_to(self, target):
        target.data = self._data
        return target

    def copy_from(self, obj):
        self.data = obj.data

    def tiles(self):
        from .tileables import handler

        new_entity = self.copy()
        new_entity.data = handler.tiles(self.data)
        return new_entity

    def __getattr__(self, attr):
        return getattr(self._data, attr)

    def __setattr__(self, key, value):
        try:
            object.__setattr__(self, key, value)
        except AttributeError:
            return setattr(self._data, key, value)


ENTITY_TYPE = (Entity, EntityData)
