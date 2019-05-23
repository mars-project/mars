# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mars/serialize/protos/graph.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mars.serialize.protos import operand_pb2 as mars_dot_serialize_dot_protos_dot_operand__pb2
from mars.serialize.protos import tensor_pb2 as mars_dot_serialize_dot_protos_dot_tensor__pb2
from mars.serialize.protos import dataframe_pb2 as mars_dot_serialize_dot_protos_dot_dataframe__pb2
from mars.serialize.protos import fusechunk_pb2 as mars_dot_serialize_dot_protos_dot_fusechunk__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='mars/serialize/protos/graph.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n!mars/serialize/protos/graph.proto\x1a#mars/serialize/protos/operand.proto\x1a\"mars/serialize/protos/tensor.proto\x1a%mars/serialize/protos/dataframe.proto\x1a%mars/serialize/protos/fusechunk.proto\"\xe3\x03\n\x08GraphDef\x12\x1e\n\x05level\x18\x02 \x01(\x0e\x32\x0f.GraphDef.Level\x12\x1f\n\x04node\x18\x01 \x03(\x0b\x32\x11.GraphDef.NodeDef\x1a\xf5\x02\n\x07NodeDef\x12\x19\n\x02op\x18\x01 \x01(\x0b\x32\x0b.OperandDefH\x00\x12\'\n\x0ctensor_chunk\x18\x02 \x01(\x0b\x32\x0f.TensorChunkDefH\x00\x12\x1c\n\x06tensor\x18\x03 \x01(\x0b\x32\n.TensorDefH\x00\x12-\n\x0f\x64\x61taframe_chunk\x18\x04 \x01(\x0b\x32\x12.DataFrameChunkDefH\x00\x12\"\n\tdataframe\x18\x05 \x01(\x0b\x32\r.DataFrameDefH\x00\x12%\n\x0bindex_chunk\x18\x06 \x01(\x0b\x32\x0e.IndexChunkDefH\x00\x12\x1a\n\x05index\x18\x07 \x01(\x0b\x32\t.IndexDefH\x00\x12\'\n\x0cseries_chunk\x18\x08 \x01(\x0b\x32\x0f.SeriesChunkDefH\x00\x12\x1c\n\x06series\x18\t \x01(\x0b\x32\n.SeriesDefH\x00\x12#\n\nfuse_chunk\x18\n \x01(\x0b\x32\r.FuseChunkDefH\x00\x42\x06\n\x04node\"\x1e\n\x05Level\x12\t\n\x05\x43HUNK\x10\x00\x12\n\n\x06\x45NTITY\x10\x01\x62\x06proto3')
  ,
  dependencies=[mars_dot_serialize_dot_protos_dot_operand__pb2.DESCRIPTOR,mars_dot_serialize_dot_protos_dot_tensor__pb2.DESCRIPTOR,mars_dot_serialize_dot_protos_dot_dataframe__pb2.DESCRIPTOR,mars_dot_serialize_dot_protos_dot_fusechunk__pb2.DESCRIPTOR,])



_GRAPHDEF_LEVEL = _descriptor.EnumDescriptor(
  name='Level',
  full_name='GraphDef.Level',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='CHUNK', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ENTITY', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=642,
  serialized_end=672,
)
_sym_db.RegisterEnumDescriptor(_GRAPHDEF_LEVEL)


_GRAPHDEF_NODEDEF = _descriptor.Descriptor(
  name='NodeDef',
  full_name='GraphDef.NodeDef',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='op', full_name='GraphDef.NodeDef.op', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tensor_chunk', full_name='GraphDef.NodeDef.tensor_chunk', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tensor', full_name='GraphDef.NodeDef.tensor', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dataframe_chunk', full_name='GraphDef.NodeDef.dataframe_chunk', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dataframe', full_name='GraphDef.NodeDef.dataframe', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='index_chunk', full_name='GraphDef.NodeDef.index_chunk', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='index', full_name='GraphDef.NodeDef.index', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='series_chunk', full_name='GraphDef.NodeDef.series_chunk', index=7,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='series', full_name='GraphDef.NodeDef.series', index=8,
      number=9, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='fuse_chunk', full_name='GraphDef.NodeDef.fuse_chunk', index=9,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='node', full_name='GraphDef.NodeDef.node',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=267,
  serialized_end=640,
)

_GRAPHDEF = _descriptor.Descriptor(
  name='GraphDef',
  full_name='GraphDef',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='level', full_name='GraphDef.level', index=0,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='node', full_name='GraphDef.node', index=1,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_GRAPHDEF_NODEDEF, ],
  enum_types=[
    _GRAPHDEF_LEVEL,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=189,
  serialized_end=672,
)

_GRAPHDEF_NODEDEF.fields_by_name['op'].message_type = mars_dot_serialize_dot_protos_dot_operand__pb2._OPERANDDEF
_GRAPHDEF_NODEDEF.fields_by_name['tensor_chunk'].message_type = mars_dot_serialize_dot_protos_dot_tensor__pb2._TENSORCHUNKDEF
_GRAPHDEF_NODEDEF.fields_by_name['tensor'].message_type = mars_dot_serialize_dot_protos_dot_tensor__pb2._TENSORDEF
_GRAPHDEF_NODEDEF.fields_by_name['dataframe_chunk'].message_type = mars_dot_serialize_dot_protos_dot_dataframe__pb2._DATAFRAMECHUNKDEF
_GRAPHDEF_NODEDEF.fields_by_name['dataframe'].message_type = mars_dot_serialize_dot_protos_dot_dataframe__pb2._DATAFRAMEDEF
_GRAPHDEF_NODEDEF.fields_by_name['index_chunk'].message_type = mars_dot_serialize_dot_protos_dot_dataframe__pb2._INDEXCHUNKDEF
_GRAPHDEF_NODEDEF.fields_by_name['index'].message_type = mars_dot_serialize_dot_protos_dot_dataframe__pb2._INDEXDEF
_GRAPHDEF_NODEDEF.fields_by_name['series_chunk'].message_type = mars_dot_serialize_dot_protos_dot_dataframe__pb2._SERIESCHUNKDEF
_GRAPHDEF_NODEDEF.fields_by_name['series'].message_type = mars_dot_serialize_dot_protos_dot_dataframe__pb2._SERIESDEF
_GRAPHDEF_NODEDEF.fields_by_name['fuse_chunk'].message_type = mars_dot_serialize_dot_protos_dot_fusechunk__pb2._FUSECHUNKDEF
_GRAPHDEF_NODEDEF.containing_type = _GRAPHDEF
_GRAPHDEF_NODEDEF.oneofs_by_name['node'].fields.append(
  _GRAPHDEF_NODEDEF.fields_by_name['op'])
_GRAPHDEF_NODEDEF.fields_by_name['op'].containing_oneof = _GRAPHDEF_NODEDEF.oneofs_by_name['node']
_GRAPHDEF_NODEDEF.oneofs_by_name['node'].fields.append(
  _GRAPHDEF_NODEDEF.fields_by_name['tensor_chunk'])
_GRAPHDEF_NODEDEF.fields_by_name['tensor_chunk'].containing_oneof = _GRAPHDEF_NODEDEF.oneofs_by_name['node']
_GRAPHDEF_NODEDEF.oneofs_by_name['node'].fields.append(
  _GRAPHDEF_NODEDEF.fields_by_name['tensor'])
_GRAPHDEF_NODEDEF.fields_by_name['tensor'].containing_oneof = _GRAPHDEF_NODEDEF.oneofs_by_name['node']
_GRAPHDEF_NODEDEF.oneofs_by_name['node'].fields.append(
  _GRAPHDEF_NODEDEF.fields_by_name['dataframe_chunk'])
_GRAPHDEF_NODEDEF.fields_by_name['dataframe_chunk'].containing_oneof = _GRAPHDEF_NODEDEF.oneofs_by_name['node']
_GRAPHDEF_NODEDEF.oneofs_by_name['node'].fields.append(
  _GRAPHDEF_NODEDEF.fields_by_name['dataframe'])
_GRAPHDEF_NODEDEF.fields_by_name['dataframe'].containing_oneof = _GRAPHDEF_NODEDEF.oneofs_by_name['node']
_GRAPHDEF_NODEDEF.oneofs_by_name['node'].fields.append(
  _GRAPHDEF_NODEDEF.fields_by_name['index_chunk'])
_GRAPHDEF_NODEDEF.fields_by_name['index_chunk'].containing_oneof = _GRAPHDEF_NODEDEF.oneofs_by_name['node']
_GRAPHDEF_NODEDEF.oneofs_by_name['node'].fields.append(
  _GRAPHDEF_NODEDEF.fields_by_name['index'])
_GRAPHDEF_NODEDEF.fields_by_name['index'].containing_oneof = _GRAPHDEF_NODEDEF.oneofs_by_name['node']
_GRAPHDEF_NODEDEF.oneofs_by_name['node'].fields.append(
  _GRAPHDEF_NODEDEF.fields_by_name['series_chunk'])
_GRAPHDEF_NODEDEF.fields_by_name['series_chunk'].containing_oneof = _GRAPHDEF_NODEDEF.oneofs_by_name['node']
_GRAPHDEF_NODEDEF.oneofs_by_name['node'].fields.append(
  _GRAPHDEF_NODEDEF.fields_by_name['series'])
_GRAPHDEF_NODEDEF.fields_by_name['series'].containing_oneof = _GRAPHDEF_NODEDEF.oneofs_by_name['node']
_GRAPHDEF_NODEDEF.oneofs_by_name['node'].fields.append(
  _GRAPHDEF_NODEDEF.fields_by_name['fuse_chunk'])
_GRAPHDEF_NODEDEF.fields_by_name['fuse_chunk'].containing_oneof = _GRAPHDEF_NODEDEF.oneofs_by_name['node']
_GRAPHDEF.fields_by_name['level'].enum_type = _GRAPHDEF_LEVEL
_GRAPHDEF.fields_by_name['node'].message_type = _GRAPHDEF_NODEDEF
_GRAPHDEF_LEVEL.containing_type = _GRAPHDEF
DESCRIPTOR.message_types_by_name['GraphDef'] = _GRAPHDEF
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

GraphDef = _reflection.GeneratedProtocolMessageType('GraphDef', (_message.Message,), dict(

  NodeDef = _reflection.GeneratedProtocolMessageType('NodeDef', (_message.Message,), dict(
    DESCRIPTOR = _GRAPHDEF_NODEDEF,
    __module__ = 'mars.serialize.protos.graph_pb2'
    # @@protoc_insertion_point(class_scope:GraphDef.NodeDef)
    ))
  ,
  DESCRIPTOR = _GRAPHDEF,
  __module__ = 'mars.serialize.protos.graph_pb2'
  # @@protoc_insertion_point(class_scope:GraphDef)
  ))
_sym_db.RegisterMessage(GraphDef)
_sym_db.RegisterMessage(GraphDef.NodeDef)


# @@protoc_insertion_point(module_scope)
