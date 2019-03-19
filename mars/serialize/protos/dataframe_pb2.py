# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mars/serialize/protos/dataframe.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mars.serialize.protos import value_pb2 as mars_dot_serialize_dot_protos_dot_value__pb2
from mars.serialize.protos import indexvalue_pb2 as mars_dot_serialize_dot_protos_dot_indexvalue__pb2
from mars.serialize.protos import chunk_pb2 as mars_dot_serialize_dot_protos_dot_chunk__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='mars/serialize/protos/dataframe.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n%mars/serialize/protos/dataframe.proto\x1a!mars/serialize/protos/value.proto\x1a&mars/serialize/protos/indexvalue.proto\x1a!mars/serialize/protos/chunk.proto\"\xcb\x01\n\x08IndexDef\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05shape\x18\x02 \x03(\x03\x12\x15\n\x05\x64type\x18\x03 \x01(\x0b\x32\x06.Value\x12\x12\n\x02op\x18\x04 \x01(\x0b\x32\x06.Value\x12\x17\n\x07nsplits\x18\x05 \x01(\x0b\x32\x06.Value\x12\x19\n\x06\x63hunks\x18\x06 \x03(\x0b\x32\t.ChunkDef\x12\x16\n\x06params\x18\x07 \x01(\x0b\x32\x06.Value\x12\n\n\x02id\x18\x08 \x01(\t\x12 \n\x0bindex_value\x18\t \x01(\x0b\x32\x0b.IndexValue\"\xe2\x01\n\tSeriesDef\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05shape\x18\x02 \x03(\x03\x12\x15\n\x05\x64type\x18\x03 \x01(\x0b\x32\x06.Value\x12\x12\n\x02op\x18\x04 \x01(\x0b\x32\x06.Value\x12\x17\n\x07nsplits\x18\x05 \x01(\x0b\x32\x06.Value\x12\x19\n\x06\x63hunks\x18\x06 \x03(\x0b\x32\t.ChunkDef\x12\x16\n\x06params\x18\x07 \x01(\x0b\x32\x06.Value\x12\n\n\x02id\x18\x08 \x01(\t\x12\x14\n\x04name\x18\t \x01(\x0b\x32\x06.Value\x12 \n\x0bindex_value\x18\n \x01(\x0b\x32\x0b.IndexValue\"\xf4\x01\n\x0c\x44\x61taFrameDef\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05shape\x18\x02 \x03(\x03\x12\x16\n\x06\x64types\x18\x03 \x01(\x0b\x32\x06.Value\x12\x12\n\x02op\x18\x04 \x01(\x0b\x32\x06.Value\x12\x17\n\x07nsplits\x18\x05 \x01(\x0b\x32\x06.Value\x12\x19\n\x06\x63hunks\x18\x06 \x03(\x0b\x32\t.ChunkDef\x12\x16\n\x06params\x18\x07 \x01(\x0b\x32\x06.Value\x12\n\n\x02id\x18\x08 \x01(\t\x12 \n\x0bindex_value\x18\t \x01(\x0b\x32\x0b.IndexValue\x12\"\n\rcolumns_value\x18\n \x01(\x0b\x32\x0b.IndexValueb\x06proto3')
  ,
  dependencies=[mars_dot_serialize_dot_protos_dot_value__pb2.DESCRIPTOR,mars_dot_serialize_dot_protos_dot_indexvalue__pb2.DESCRIPTOR,mars_dot_serialize_dot_protos_dot_chunk__pb2.DESCRIPTOR,])




_INDEXDEF = _descriptor.Descriptor(
  name='IndexDef',
  full_name='IndexDef',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='IndexDef.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shape', full_name='IndexDef.shape', index=1,
      number=2, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dtype', full_name='IndexDef.dtype', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='op', full_name='IndexDef.op', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nsplits', full_name='IndexDef.nsplits', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='chunks', full_name='IndexDef.chunks', index=5,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='params', full_name='IndexDef.params', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='id', full_name='IndexDef.id', index=7,
      number=8, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='index_value', full_name='IndexDef.index_value', index=8,
      number=9, type=11, cpp_type=10, label=1,
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
  ],
  serialized_start=152,
  serialized_end=355,
)


_SERIESDEF = _descriptor.Descriptor(
  name='SeriesDef',
  full_name='SeriesDef',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='SeriesDef.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shape', full_name='SeriesDef.shape', index=1,
      number=2, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dtype', full_name='SeriesDef.dtype', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='op', full_name='SeriesDef.op', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nsplits', full_name='SeriesDef.nsplits', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='chunks', full_name='SeriesDef.chunks', index=5,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='params', full_name='SeriesDef.params', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='id', full_name='SeriesDef.id', index=7,
      number=8, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='SeriesDef.name', index=8,
      number=9, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='index_value', full_name='SeriesDef.index_value', index=9,
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
  ],
  serialized_start=358,
  serialized_end=584,
)


_DATAFRAMEDEF = _descriptor.Descriptor(
  name='DataFrameDef',
  full_name='DataFrameDef',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='DataFrameDef.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shape', full_name='DataFrameDef.shape', index=1,
      number=2, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dtypes', full_name='DataFrameDef.dtypes', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='op', full_name='DataFrameDef.op', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nsplits', full_name='DataFrameDef.nsplits', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='chunks', full_name='DataFrameDef.chunks', index=5,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='params', full_name='DataFrameDef.params', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='id', full_name='DataFrameDef.id', index=7,
      number=8, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='index_value', full_name='DataFrameDef.index_value', index=8,
      number=9, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='columns_value', full_name='DataFrameDef.columns_value', index=9,
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
  ],
  serialized_start=587,
  serialized_end=831,
)

_INDEXDEF.fields_by_name['dtype'].message_type = mars_dot_serialize_dot_protos_dot_value__pb2._VALUE
_INDEXDEF.fields_by_name['op'].message_type = mars_dot_serialize_dot_protos_dot_value__pb2._VALUE
_INDEXDEF.fields_by_name['nsplits'].message_type = mars_dot_serialize_dot_protos_dot_value__pb2._VALUE
_INDEXDEF.fields_by_name['chunks'].message_type = mars_dot_serialize_dot_protos_dot_chunk__pb2._CHUNKDEF
_INDEXDEF.fields_by_name['params'].message_type = mars_dot_serialize_dot_protos_dot_value__pb2._VALUE
_INDEXDEF.fields_by_name['index_value'].message_type = mars_dot_serialize_dot_protos_dot_indexvalue__pb2._INDEXVALUE
_SERIESDEF.fields_by_name['dtype'].message_type = mars_dot_serialize_dot_protos_dot_value__pb2._VALUE
_SERIESDEF.fields_by_name['op'].message_type = mars_dot_serialize_dot_protos_dot_value__pb2._VALUE
_SERIESDEF.fields_by_name['nsplits'].message_type = mars_dot_serialize_dot_protos_dot_value__pb2._VALUE
_SERIESDEF.fields_by_name['chunks'].message_type = mars_dot_serialize_dot_protos_dot_chunk__pb2._CHUNKDEF
_SERIESDEF.fields_by_name['params'].message_type = mars_dot_serialize_dot_protos_dot_value__pb2._VALUE
_SERIESDEF.fields_by_name['name'].message_type = mars_dot_serialize_dot_protos_dot_value__pb2._VALUE
_SERIESDEF.fields_by_name['index_value'].message_type = mars_dot_serialize_dot_protos_dot_indexvalue__pb2._INDEXVALUE
_DATAFRAMEDEF.fields_by_name['dtypes'].message_type = mars_dot_serialize_dot_protos_dot_value__pb2._VALUE
_DATAFRAMEDEF.fields_by_name['op'].message_type = mars_dot_serialize_dot_protos_dot_value__pb2._VALUE
_DATAFRAMEDEF.fields_by_name['nsplits'].message_type = mars_dot_serialize_dot_protos_dot_value__pb2._VALUE
_DATAFRAMEDEF.fields_by_name['chunks'].message_type = mars_dot_serialize_dot_protos_dot_chunk__pb2._CHUNKDEF
_DATAFRAMEDEF.fields_by_name['params'].message_type = mars_dot_serialize_dot_protos_dot_value__pb2._VALUE
_DATAFRAMEDEF.fields_by_name['index_value'].message_type = mars_dot_serialize_dot_protos_dot_indexvalue__pb2._INDEXVALUE
_DATAFRAMEDEF.fields_by_name['columns_value'].message_type = mars_dot_serialize_dot_protos_dot_indexvalue__pb2._INDEXVALUE
DESCRIPTOR.message_types_by_name['IndexDef'] = _INDEXDEF
DESCRIPTOR.message_types_by_name['SeriesDef'] = _SERIESDEF
DESCRIPTOR.message_types_by_name['DataFrameDef'] = _DATAFRAMEDEF
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

IndexDef = _reflection.GeneratedProtocolMessageType('IndexDef', (_message.Message,), dict(
  DESCRIPTOR = _INDEXDEF,
  __module__ = 'mars.serialize.protos.dataframe_pb2'
  # @@protoc_insertion_point(class_scope:IndexDef)
  ))
_sym_db.RegisterMessage(IndexDef)

SeriesDef = _reflection.GeneratedProtocolMessageType('SeriesDef', (_message.Message,), dict(
  DESCRIPTOR = _SERIESDEF,
  __module__ = 'mars.serialize.protos.dataframe_pb2'
  # @@protoc_insertion_point(class_scope:SeriesDef)
  ))
_sym_db.RegisterMessage(SeriesDef)

DataFrameDef = _reflection.GeneratedProtocolMessageType('DataFrameDef', (_message.Message,), dict(
  DESCRIPTOR = _DATAFRAMEDEF,
  __module__ = 'mars.serialize.protos.dataframe_pb2'
  # @@protoc_insertion_point(class_scope:DataFrameDef)
  ))
_sym_db.RegisterMessage(DataFrameDef)


# @@protoc_insertion_point(module_scope)
