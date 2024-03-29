# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: scene.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import common_pb2 as common__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='scene.proto',
  package='sgame_state',
  syntax='proto2',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0bscene.proto\x12\x0bsgame_state\x1a\x0c\x63ommon.proto\"G\n\x04\x43\x61ke\x12\x10\n\x08\x63onfigId\x18\x01 \x02(\x05\x12-\n\x08\x63ollider\x18\x02 \x02(\x0b\x32\x1b.sgame_state.SphereCollider\"\xc0\x01\n\x06\x42ullet\x12\x12\n\nruntime_id\x18\x01 \x02(\x05\x12%\n\x04\x63\x61mp\x18\x02 \x02(\x0e\x32\x17.sgame_state.PLAYERCAMP\x12\x14\n\x0csource_actor\x18\x03 \x02(\x05\x12-\n\tslot_type\x18\x04 \x02(\x0e\x32\x1a.sgame_state.SkillSlotType\x12\x10\n\x08skill_id\x18\x05 \x02(\x05\x12$\n\x08location\x18\x06 \x02(\x0b\x32\x12.sgame_state.VInt3'
  ,
  dependencies=[common__pb2.DESCRIPTOR,])




_CAKE = _descriptor.Descriptor(
  name='Cake',
  full_name='sgame_state.Cake',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='configId', full_name='sgame_state.Cake.configId', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='collider', full_name='sgame_state.Cake.collider', index=1,
      number=2, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=42,
  serialized_end=113,
)


_BULLET = _descriptor.Descriptor(
  name='Bullet',
  full_name='sgame_state.Bullet',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='runtime_id', full_name='sgame_state.Bullet.runtime_id', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='camp', full_name='sgame_state.Bullet.camp', index=1,
      number=2, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='source_actor', full_name='sgame_state.Bullet.source_actor', index=2,
      number=3, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='slot_type', full_name='sgame_state.Bullet.slot_type', index=3,
      number=4, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='skill_id', full_name='sgame_state.Bullet.skill_id', index=4,
      number=5, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='location', full_name='sgame_state.Bullet.location', index=5,
      number=6, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=116,
  serialized_end=308,
)

_CAKE.fields_by_name['collider'].message_type = common__pb2._SPHERECOLLIDER
_BULLET.fields_by_name['camp'].enum_type = common__pb2._PLAYERCAMP
_BULLET.fields_by_name['slot_type'].enum_type = common__pb2._SKILLSLOTTYPE
_BULLET.fields_by_name['location'].message_type = common__pb2._VINT3
DESCRIPTOR.message_types_by_name['Cake'] = _CAKE
DESCRIPTOR.message_types_by_name['Bullet'] = _BULLET
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Cake = _reflection.GeneratedProtocolMessageType('Cake', (_message.Message,), {
  'DESCRIPTOR' : _CAKE,
  '__module__' : 'scene_pb2'
  # @@protoc_insertion_point(class_scope:sgame_state.Cake)
  })
_sym_db.RegisterMessage(Cake)

Bullet = _reflection.GeneratedProtocolMessageType('Bullet', (_message.Message,), {
  'DESCRIPTOR' : _BULLET,
  '__module__' : 'scene_pb2'
  # @@protoc_insertion_point(class_scope:sgame_state.Bullet)
  })
_sym_db.RegisterMessage(Bullet)


# @@protoc_insertion_point(module_scope)
