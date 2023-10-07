# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: hero.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import common_pb2 as common__pb2
from . import command_pb2 as command__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='hero.proto',
  package='sgame_state',
  syntax='proto2',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\nhero.proto\x12\x0bsgame_state\x1a\x0c\x63ommon.proto\x1a\rcommand.proto\"\x89\x02\n\x0eSkillSlotState\x12\x10\n\x08\x63onfigId\x18\x01 \x02(\x05\x12-\n\tslot_type\x18\x02 \x02(\x0e\x32\x1a.sgame_state.SkillSlotType\x12\r\n\x05level\x18\x03 \x02(\x05\x12\x0e\n\x06usable\x18\x04 \x02(\x08\x12\x10\n\x08\x63ooldown\x18\x05 \x02(\x05\x12\x14\n\x0c\x63ooldown_max\x18\x06 \x02(\x05\x12\x11\n\tusedTimes\x18\x07 \x01(\x05\x12\x14\n\x0chitHeroTimes\x18\x08 \x01(\x05\x12\x17\n\x0fsuccUsedInFrame\x18\t \x01(\x05\x12\x14\n\x0cnextConfigID\x18\n \x01(\x05\x12\x17\n\x0f\x63omboEffectTime\x18\x0b \x01(\x05\">\n\nSkillState\x12\x30\n\x0bslot_states\x18\x01 \x03(\x0b\x32\x1b.sgame_state.SkillSlotState\"X\n\x0e\x42uffSkillState\x12\x10\n\x08\x63onfigId\x18\x01 \x02(\x05\x12\x11\n\tstartTime\x18\x02 \x02(\x04\x12\r\n\x05times\x18\x03 \x02(\x05\x12\x12\n\neffectType\x18\x04 \x01(\x05\"H\n\rBuffMarkState\x12\x16\n\x0eorigin_actorId\x18\x01 \x02(\x05\x12\x10\n\x08\x63onfigId\x18\x02 \x02(\x05\x12\r\n\x05layer\x18\x03 \x02(\x05\"m\n\tBuffState\x12\x30\n\x0b\x62uff_skills\x18\x01 \x03(\x0b\x32\x1b.sgame_state.BuffSkillState\x12.\n\nbuff_marks\x18\x02 \x03(\x0b\x32\x1a.sgame_state.BuffMarkState\"9\n\x0cPassiveSkill\x12\x17\n\x0fpassive_skillid\x18\x01 \x01(\x05\x12\x10\n\x08\x63ooldown\x18\x02 \x01(\x05\"7\n\x0b\x41\x63tiveSkill\x12\x16\n\x0e\x61\x63tive_skillid\x18\x01 \x01(\x05\x12\x10\n\x08\x63ooldown\x18\x02 \x01(\x05\"\xa1\x01\n\tEquipSlot\x12\x10\n\x08\x63onfigId\x18\x01 \x02(\x05\x12\x10\n\x08\x62uyPrice\x18\x02 \x02(\x05\x12\x0e\n\x06\x61mount\x18\x03 \x02(\x05\x12.\n\x0c\x61\x63tive_skill\x18\x04 \x03(\x0b\x32\x18.sgame_state.ActiveSkill\x12\x30\n\rpassive_skill\x18\x05 \x03(\x0b\x32\x19.sgame_state.PassiveSkill\"4\n\nEquipState\x12&\n\x06\x65quips\x18\x01 \x03(\x0b\x32\x16.sgame_state.EquipSlot\"\x9a\x01\n\x13ReturnCityAbortInfo\x12.\n\tabortType\x18\x01 \x01(\x0e\x32\x1b.sgame_state.SkillAbortType\x12\x10\n\x08isActive\x18\x02 \x01(\x08\x12\x32\n\x0e\x61ttackSlotType\x18\x03 \x01(\x0e\x32\x1a.sgame_state.SkillSlotType\x12\r\n\x05objID\x18\x04 \x01(\r\"R\n\x0bProtectInfo\x12-\n\x0bprotectType\x18\x01 \x01(\x0e\x32\x18.sgame_state.ProtectType\x12\x14\n\x0cprotectValue\x18\x02 \x01(\r\"\xc8\x05\n\tHeroState\x12\x11\n\tplayer_id\x18\x01 \x02(\r\x12,\n\x0b\x61\x63tor_state\x18\x02 \x02(\x0b\x32\x17.sgame_state.ActorState\x12,\n\x0bskill_state\x18\x03 \x02(\x0b\x32\x17.sgame_state.SkillState\x12,\n\x0b\x65quip_state\x18\x04 \x02(\x0b\x32\x17.sgame_state.EquipState\x12*\n\nbuff_state\x18\x05 \x02(\x0b\x32\x16.sgame_state.BuffState\x12\r\n\x05level\x18\x06 \x02(\x05\x12\x0b\n\x03\x65xp\x18\x07 \x02(\x05\x12\r\n\x05money\x18\x08 \x02(\x05\x12\x13\n\x0brevive_time\x18\t \x02(\x05\x12\x0f\n\x07killCnt\x18\n \x02(\x05\x12\x0f\n\x07\x64\x65\x61\x64\x43nt\x18\x0b \x02(\x05\x12\x11\n\tassistCnt\x18\x0c \x02(\x05\x12\x10\n\x08moneyCnt\x18\r \x02(\x05\x12\x11\n\ttotalHurt\x18\x0e \x02(\x05\x12\x17\n\x0ftotalHurtToHero\x18\x0f \x02(\x05\x12\x19\n\x11totalBeHurtByHero\x18\x10 \x02(\x05\x12\x30\n\rpassive_skill\x18\x11 \x03(\x0b\x32\x19.sgame_state.PassiveSkill\x12%\n\x08real_cmd\x18\x12 \x03(\x0b\x32\x13.sgame_state.CmdPkg\x12\x30\n\rtakeHurtInfos\x18\x13 \x03(\x0b\x32\x19.sgame_state.TakeHurtInfo\x12\x18\n\x10\x63\x61nAbortCurSkill\x18\x14 \x03(\x08\x12=\n\x13returnCityAbortInfo\x18\x15 \x03(\x0b\x32 .sgame_state.ReturnCityAbortInfo\x12\x11\n\tisInGrass\x18\x16 \x01(\x08\x12-\n\x0bprotectInfo\x18\x17 \x03(\x0b\x32\x18.sgame_state.ProtectInfo'
  ,
  dependencies=[common__pb2.DESCRIPTOR,command__pb2.DESCRIPTOR,])




_SKILLSLOTSTATE = _descriptor.Descriptor(
  name='SkillSlotState',
  full_name='sgame_state.SkillSlotState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='configId', full_name='sgame_state.SkillSlotState.configId', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='slot_type', full_name='sgame_state.SkillSlotState.slot_type', index=1,
      number=2, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='level', full_name='sgame_state.SkillSlotState.level', index=2,
      number=3, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='usable', full_name='sgame_state.SkillSlotState.usable', index=3,
      number=4, type=8, cpp_type=7, label=2,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='cooldown', full_name='sgame_state.SkillSlotState.cooldown', index=4,
      number=5, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='cooldown_max', full_name='sgame_state.SkillSlotState.cooldown_max', index=5,
      number=6, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='usedTimes', full_name='sgame_state.SkillSlotState.usedTimes', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='hitHeroTimes', full_name='sgame_state.SkillSlotState.hitHeroTimes', index=7,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='succUsedInFrame', full_name='sgame_state.SkillSlotState.succUsedInFrame', index=8,
      number=9, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='nextConfigID', full_name='sgame_state.SkillSlotState.nextConfigID', index=9,
      number=10, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='comboEffectTime', full_name='sgame_state.SkillSlotState.comboEffectTime', index=10,
      number=11, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=57,
  serialized_end=322,
)


_SKILLSTATE = _descriptor.Descriptor(
  name='SkillState',
  full_name='sgame_state.SkillState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='slot_states', full_name='sgame_state.SkillState.slot_states', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=324,
  serialized_end=386,
)


_BUFFSKILLSTATE = _descriptor.Descriptor(
  name='BuffSkillState',
  full_name='sgame_state.BuffSkillState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='configId', full_name='sgame_state.BuffSkillState.configId', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='startTime', full_name='sgame_state.BuffSkillState.startTime', index=1,
      number=2, type=4, cpp_type=4, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='times', full_name='sgame_state.BuffSkillState.times', index=2,
      number=3, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='effectType', full_name='sgame_state.BuffSkillState.effectType', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=388,
  serialized_end=476,
)


_BUFFMARKSTATE = _descriptor.Descriptor(
  name='BuffMarkState',
  full_name='sgame_state.BuffMarkState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='origin_actorId', full_name='sgame_state.BuffMarkState.origin_actorId', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='configId', full_name='sgame_state.BuffMarkState.configId', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='layer', full_name='sgame_state.BuffMarkState.layer', index=2,
      number=3, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
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
  serialized_start=478,
  serialized_end=550,
)


_BUFFSTATE = _descriptor.Descriptor(
  name='BuffState',
  full_name='sgame_state.BuffState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='buff_skills', full_name='sgame_state.BuffState.buff_skills', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='buff_marks', full_name='sgame_state.BuffState.buff_marks', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=552,
  serialized_end=661,
)


_PASSIVESKILL = _descriptor.Descriptor(
  name='PassiveSkill',
  full_name='sgame_state.PassiveSkill',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='passive_skillid', full_name='sgame_state.PassiveSkill.passive_skillid', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='cooldown', full_name='sgame_state.PassiveSkill.cooldown', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=663,
  serialized_end=720,
)


_ACTIVESKILL = _descriptor.Descriptor(
  name='ActiveSkill',
  full_name='sgame_state.ActiveSkill',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='active_skillid', full_name='sgame_state.ActiveSkill.active_skillid', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='cooldown', full_name='sgame_state.ActiveSkill.cooldown', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=722,
  serialized_end=777,
)


_EQUIPSLOT = _descriptor.Descriptor(
  name='EquipSlot',
  full_name='sgame_state.EquipSlot',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='configId', full_name='sgame_state.EquipSlot.configId', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='buyPrice', full_name='sgame_state.EquipSlot.buyPrice', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='amount', full_name='sgame_state.EquipSlot.amount', index=2,
      number=3, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='active_skill', full_name='sgame_state.EquipSlot.active_skill', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='passive_skill', full_name='sgame_state.EquipSlot.passive_skill', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=780,
  serialized_end=941,
)


_EQUIPSTATE = _descriptor.Descriptor(
  name='EquipState',
  full_name='sgame_state.EquipState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='equips', full_name='sgame_state.EquipState.equips', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=943,
  serialized_end=995,
)


_RETURNCITYABORTINFO = _descriptor.Descriptor(
  name='ReturnCityAbortInfo',
  full_name='sgame_state.ReturnCityAbortInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='abortType', full_name='sgame_state.ReturnCityAbortInfo.abortType', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='isActive', full_name='sgame_state.ReturnCityAbortInfo.isActive', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='attackSlotType', full_name='sgame_state.ReturnCityAbortInfo.attackSlotType', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='objID', full_name='sgame_state.ReturnCityAbortInfo.objID', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=998,
  serialized_end=1152,
)


_PROTECTINFO = _descriptor.Descriptor(
  name='ProtectInfo',
  full_name='sgame_state.ProtectInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='protectType', full_name='sgame_state.ProtectInfo.protectType', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='protectValue', full_name='sgame_state.ProtectInfo.protectValue', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=1154,
  serialized_end=1236,
)


_HEROSTATE = _descriptor.Descriptor(
  name='HeroState',
  full_name='sgame_state.HeroState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='player_id', full_name='sgame_state.HeroState.player_id', index=0,
      number=1, type=13, cpp_type=3, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='actor_state', full_name='sgame_state.HeroState.actor_state', index=1,
      number=2, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='skill_state', full_name='sgame_state.HeroState.skill_state', index=2,
      number=3, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='equip_state', full_name='sgame_state.HeroState.equip_state', index=3,
      number=4, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='buff_state', full_name='sgame_state.HeroState.buff_state', index=4,
      number=5, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='level', full_name='sgame_state.HeroState.level', index=5,
      number=6, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='exp', full_name='sgame_state.HeroState.exp', index=6,
      number=7, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='money', full_name='sgame_state.HeroState.money', index=7,
      number=8, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='revive_time', full_name='sgame_state.HeroState.revive_time', index=8,
      number=9, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='killCnt', full_name='sgame_state.HeroState.killCnt', index=9,
      number=10, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='deadCnt', full_name='sgame_state.HeroState.deadCnt', index=10,
      number=11, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='assistCnt', full_name='sgame_state.HeroState.assistCnt', index=11,
      number=12, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='moneyCnt', full_name='sgame_state.HeroState.moneyCnt', index=12,
      number=13, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='totalHurt', full_name='sgame_state.HeroState.totalHurt', index=13,
      number=14, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='totalHurtToHero', full_name='sgame_state.HeroState.totalHurtToHero', index=14,
      number=15, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='totalBeHurtByHero', full_name='sgame_state.HeroState.totalBeHurtByHero', index=15,
      number=16, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='passive_skill', full_name='sgame_state.HeroState.passive_skill', index=16,
      number=17, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='real_cmd', full_name='sgame_state.HeroState.real_cmd', index=17,
      number=18, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='takeHurtInfos', full_name='sgame_state.HeroState.takeHurtInfos', index=18,
      number=19, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='canAbortCurSkill', full_name='sgame_state.HeroState.canAbortCurSkill', index=19,
      number=20, type=8, cpp_type=7, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='returnCityAbortInfo', full_name='sgame_state.HeroState.returnCityAbortInfo', index=20,
      number=21, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='isInGrass', full_name='sgame_state.HeroState.isInGrass', index=21,
      number=22, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='protectInfo', full_name='sgame_state.HeroState.protectInfo', index=22,
      number=23, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=1239,
  serialized_end=1951,
)

_SKILLSLOTSTATE.fields_by_name['slot_type'].enum_type = common__pb2._SKILLSLOTTYPE
_SKILLSTATE.fields_by_name['slot_states'].message_type = _SKILLSLOTSTATE
_BUFFSTATE.fields_by_name['buff_skills'].message_type = _BUFFSKILLSTATE
_BUFFSTATE.fields_by_name['buff_marks'].message_type = _BUFFMARKSTATE
_EQUIPSLOT.fields_by_name['active_skill'].message_type = _ACTIVESKILL
_EQUIPSLOT.fields_by_name['passive_skill'].message_type = _PASSIVESKILL
_EQUIPSTATE.fields_by_name['equips'].message_type = _EQUIPSLOT
_RETURNCITYABORTINFO.fields_by_name['abortType'].enum_type = common__pb2._SKILLABORTTYPE
_RETURNCITYABORTINFO.fields_by_name['attackSlotType'].enum_type = common__pb2._SKILLSLOTTYPE
_PROTECTINFO.fields_by_name['protectType'].enum_type = common__pb2._PROTECTTYPE
_HEROSTATE.fields_by_name['actor_state'].message_type = common__pb2._ACTORSTATE
_HEROSTATE.fields_by_name['skill_state'].message_type = _SKILLSTATE
_HEROSTATE.fields_by_name['equip_state'].message_type = _EQUIPSTATE
_HEROSTATE.fields_by_name['buff_state'].message_type = _BUFFSTATE
_HEROSTATE.fields_by_name['passive_skill'].message_type = _PASSIVESKILL
_HEROSTATE.fields_by_name['real_cmd'].message_type = command__pb2._CMDPKG
_HEROSTATE.fields_by_name['takeHurtInfos'].message_type = common__pb2._TAKEHURTINFO
_HEROSTATE.fields_by_name['returnCityAbortInfo'].message_type = _RETURNCITYABORTINFO
_HEROSTATE.fields_by_name['protectInfo'].message_type = _PROTECTINFO
DESCRIPTOR.message_types_by_name['SkillSlotState'] = _SKILLSLOTSTATE
DESCRIPTOR.message_types_by_name['SkillState'] = _SKILLSTATE
DESCRIPTOR.message_types_by_name['BuffSkillState'] = _BUFFSKILLSTATE
DESCRIPTOR.message_types_by_name['BuffMarkState'] = _BUFFMARKSTATE
DESCRIPTOR.message_types_by_name['BuffState'] = _BUFFSTATE
DESCRIPTOR.message_types_by_name['PassiveSkill'] = _PASSIVESKILL
DESCRIPTOR.message_types_by_name['ActiveSkill'] = _ACTIVESKILL
DESCRIPTOR.message_types_by_name['EquipSlot'] = _EQUIPSLOT
DESCRIPTOR.message_types_by_name['EquipState'] = _EQUIPSTATE
DESCRIPTOR.message_types_by_name['ReturnCityAbortInfo'] = _RETURNCITYABORTINFO
DESCRIPTOR.message_types_by_name['ProtectInfo'] = _PROTECTINFO
DESCRIPTOR.message_types_by_name['HeroState'] = _HEROSTATE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SkillSlotState = _reflection.GeneratedProtocolMessageType('SkillSlotState', (_message.Message,), {
  'DESCRIPTOR' : _SKILLSLOTSTATE,
  '__module__' : 'hero_pb2'
  # @@protoc_insertion_point(class_scope:sgame_state.SkillSlotState)
  })
_sym_db.RegisterMessage(SkillSlotState)

SkillState = _reflection.GeneratedProtocolMessageType('SkillState', (_message.Message,), {
  'DESCRIPTOR' : _SKILLSTATE,
  '__module__' : 'hero_pb2'
  # @@protoc_insertion_point(class_scope:sgame_state.SkillState)
  })
_sym_db.RegisterMessage(SkillState)

BuffSkillState = _reflection.GeneratedProtocolMessageType('BuffSkillState', (_message.Message,), {
  'DESCRIPTOR' : _BUFFSKILLSTATE,
  '__module__' : 'hero_pb2'
  # @@protoc_insertion_point(class_scope:sgame_state.BuffSkillState)
  })
_sym_db.RegisterMessage(BuffSkillState)

BuffMarkState = _reflection.GeneratedProtocolMessageType('BuffMarkState', (_message.Message,), {
  'DESCRIPTOR' : _BUFFMARKSTATE,
  '__module__' : 'hero_pb2'
  # @@protoc_insertion_point(class_scope:sgame_state.BuffMarkState)
  })
_sym_db.RegisterMessage(BuffMarkState)

BuffState = _reflection.GeneratedProtocolMessageType('BuffState', (_message.Message,), {
  'DESCRIPTOR' : _BUFFSTATE,
  '__module__' : 'hero_pb2'
  # @@protoc_insertion_point(class_scope:sgame_state.BuffState)
  })
_sym_db.RegisterMessage(BuffState)

PassiveSkill = _reflection.GeneratedProtocolMessageType('PassiveSkill', (_message.Message,), {
  'DESCRIPTOR' : _PASSIVESKILL,
  '__module__' : 'hero_pb2'
  # @@protoc_insertion_point(class_scope:sgame_state.PassiveSkill)
  })
_sym_db.RegisterMessage(PassiveSkill)

ActiveSkill = _reflection.GeneratedProtocolMessageType('ActiveSkill', (_message.Message,), {
  'DESCRIPTOR' : _ACTIVESKILL,
  '__module__' : 'hero_pb2'
  # @@protoc_insertion_point(class_scope:sgame_state.ActiveSkill)
  })
_sym_db.RegisterMessage(ActiveSkill)

EquipSlot = _reflection.GeneratedProtocolMessageType('EquipSlot', (_message.Message,), {
  'DESCRIPTOR' : _EQUIPSLOT,
  '__module__' : 'hero_pb2'
  # @@protoc_insertion_point(class_scope:sgame_state.EquipSlot)
  })
_sym_db.RegisterMessage(EquipSlot)

EquipState = _reflection.GeneratedProtocolMessageType('EquipState', (_message.Message,), {
  'DESCRIPTOR' : _EQUIPSTATE,
  '__module__' : 'hero_pb2'
  # @@protoc_insertion_point(class_scope:sgame_state.EquipState)
  })
_sym_db.RegisterMessage(EquipState)

ReturnCityAbortInfo = _reflection.GeneratedProtocolMessageType('ReturnCityAbortInfo', (_message.Message,), {
  'DESCRIPTOR' : _RETURNCITYABORTINFO,
  '__module__' : 'hero_pb2'
  # @@protoc_insertion_point(class_scope:sgame_state.ReturnCityAbortInfo)
  })
_sym_db.RegisterMessage(ReturnCityAbortInfo)

ProtectInfo = _reflection.GeneratedProtocolMessageType('ProtectInfo', (_message.Message,), {
  'DESCRIPTOR' : _PROTECTINFO,
  '__module__' : 'hero_pb2'
  # @@protoc_insertion_point(class_scope:sgame_state.ProtectInfo)
  })
_sym_db.RegisterMessage(ProtectInfo)

HeroState = _reflection.GeneratedProtocolMessageType('HeroState', (_message.Message,), {
  'DESCRIPTOR' : _HEROSTATE,
  '__module__' : 'hero_pb2'
  # @@protoc_insertion_point(class_scope:sgame_state.HeroState)
  })
_sym_db.RegisterMessage(HeroState)


# @@protoc_insertion_point(module_scope)