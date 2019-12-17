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


from mars.serialize.protos import anyref_pb2 as mars_dot_serialize_dot_protos_dot_anyref__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='mars/serialize/protos/graph.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n!mars/serialize/protos/graph.proto\x1a\"mars/serialize/protos/anyref.proto\"g\n\x08GraphDef\x12\x1e\n\x05level\x18\x02 \x01(\x0e\x32\x0f.GraphDef.Level\x12\x1b\n\x04node\x18\x01 \x03(\x0b\x32\r.AnyReference\"\x1e\n\x05Level\x12\t\n\x05\x43HUNK\x10\x00\x12\n\n\x06\x45NTITY\x10\x01\x62\x06proto3')
  ,
  dependencies=[mars_dot_serialize_dot_protos_dot_anyref__pb2.DESCRIPTOR,])



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
  serialized_start=146,
  serialized_end=176,
)
_sym_db.RegisterEnumDescriptor(_GRAPHDEF_LEVEL)


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
  nested_types=[],
  enum_types=[
    _GRAPHDEF_LEVEL,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=73,
  serialized_end=176,
)

_GRAPHDEF.fields_by_name['level'].enum_type = _GRAPHDEF_LEVEL
_GRAPHDEF.fields_by_name['node'].message_type = mars_dot_serialize_dot_protos_dot_anyref__pb2._ANYREFERENCE
_GRAPHDEF_LEVEL.containing_type = _GRAPHDEF
DESCRIPTOR.message_types_by_name['GraphDef'] = _GRAPHDEF
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

GraphDef = _reflection.GeneratedProtocolMessageType('GraphDef', (_message.Message,), dict(
  DESCRIPTOR = _GRAPHDEF,
  __module__ = 'mars.serialize.protos.graph_pb2'
  # @@protoc_insertion_point(class_scope:GraphDef)
  ))
_sym_db.RegisterMessage(GraphDef)


# @@protoc_insertion_point(module_scope)
