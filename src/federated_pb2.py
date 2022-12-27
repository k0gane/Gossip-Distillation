# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: federated.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='federated.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x0f\x66\x65\x64\x65rated.proto\"\x16\n\x05Model\x12\r\n\x05model\x18\x01 \x03(\t\"\x1b\n\tAnyString\x12\x0e\n\x06\x61nystr\x18\x01 \x01(\t\"+\n\rTrained_Model\x12\x1a\n\nmodel_list\x18\x01 \x03(\x0b\x32\x06.Model\"\x18\n\x06Result\x12\x0e\n\x06result\x18\x01 \x01(\t\"\x17\n\x06Timing\x12\r\n\x05hello\x18\x01 \x01(\t\"\x14\n\x04\x46lag\x12\x0c\n\x04\x66lag\x18\x01 \x01(\x08\x32\xa9\x02\n\x0bModelSender\x12*\n\nInitDevice\x12\n.AnyString\x1a\n.AnyString\"\x00(\x01\x30\x01\x12)\n\x0eIsEnoughClient\x12\n.AnyString\x1a\x05.Flag\"\x00(\x01\x30\x01\x12 \n\tSendModel\x12\x05.Flag\x1a\x06.Model\"\x00(\x01\x30\x01\x12$\n\x0bReturnModel\x12\x06.Model\x1a\x07.Result\"\x00(\x01\x30\x01\x12%\n\rIsEnoughModel\x12\x07.Timing\x1a\x05.Flag\"\x00(\x01\x30\x01\x12&\n\x0e\x46\x65\x64\x65ratedModel\x12\x05.Flag\x1a\x07.Result\"\x00(\x01\x30\x01\x12,\n\x0c\x46inishAction\x12\n.AnyString\x1a\n.AnyString\"\x00(\x01\x30\x01\x62\x06proto3')
)




_MODEL = _descriptor.Descriptor(
  name='Model',
  full_name='Model',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='model', full_name='Model.model', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=19,
  serialized_end=41,
)


_ANYSTRING = _descriptor.Descriptor(
  name='AnyString',
  full_name='AnyString',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='anystr', full_name='AnyString.anystr', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
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
  serialized_start=43,
  serialized_end=70,
)


_TRAINED_MODEL = _descriptor.Descriptor(
  name='Trained_Model',
  full_name='Trained_Model',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='model_list', full_name='Trained_Model.model_list', index=0,
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
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=72,
  serialized_end=115,
)


_RESULT = _descriptor.Descriptor(
  name='Result',
  full_name='Result',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='result', full_name='Result.result', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
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
  serialized_start=117,
  serialized_end=141,
)


_TIMING = _descriptor.Descriptor(
  name='Timing',
  full_name='Timing',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='hello', full_name='Timing.hello', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
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
  serialized_start=143,
  serialized_end=166,
)


_FLAG = _descriptor.Descriptor(
  name='Flag',
  full_name='Flag',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='flag', full_name='Flag.flag', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
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
  serialized_start=168,
  serialized_end=188,
)

_TRAINED_MODEL.fields_by_name['model_list'].message_type = _MODEL
DESCRIPTOR.message_types_by_name['Model'] = _MODEL
DESCRIPTOR.message_types_by_name['AnyString'] = _ANYSTRING
DESCRIPTOR.message_types_by_name['Trained_Model'] = _TRAINED_MODEL
DESCRIPTOR.message_types_by_name['Result'] = _RESULT
DESCRIPTOR.message_types_by_name['Timing'] = _TIMING
DESCRIPTOR.message_types_by_name['Flag'] = _FLAG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Model = _reflection.GeneratedProtocolMessageType('Model', (_message.Message,), {
  'DESCRIPTOR' : _MODEL,
  '__module__' : 'federated_pb2'
  # @@protoc_insertion_point(class_scope:Model)
  })
_sym_db.RegisterMessage(Model)

AnyString = _reflection.GeneratedProtocolMessageType('AnyString', (_message.Message,), {
  'DESCRIPTOR' : _ANYSTRING,
  '__module__' : 'federated_pb2'
  # @@protoc_insertion_point(class_scope:AnyString)
  })
_sym_db.RegisterMessage(AnyString)

Trained_Model = _reflection.GeneratedProtocolMessageType('Trained_Model', (_message.Message,), {
  'DESCRIPTOR' : _TRAINED_MODEL,
  '__module__' : 'federated_pb2'
  # @@protoc_insertion_point(class_scope:Trained_Model)
  })
_sym_db.RegisterMessage(Trained_Model)

Result = _reflection.GeneratedProtocolMessageType('Result', (_message.Message,), {
  'DESCRIPTOR' : _RESULT,
  '__module__' : 'federated_pb2'
  # @@protoc_insertion_point(class_scope:Result)
  })
_sym_db.RegisterMessage(Result)

Timing = _reflection.GeneratedProtocolMessageType('Timing', (_message.Message,), {
  'DESCRIPTOR' : _TIMING,
  '__module__' : 'federated_pb2'
  # @@protoc_insertion_point(class_scope:Timing)
  })
_sym_db.RegisterMessage(Timing)

Flag = _reflection.GeneratedProtocolMessageType('Flag', (_message.Message,), {
  'DESCRIPTOR' : _FLAG,
  '__module__' : 'federated_pb2'
  # @@protoc_insertion_point(class_scope:Flag)
  })
_sym_db.RegisterMessage(Flag)



_MODELSENDER = _descriptor.ServiceDescriptor(
  name='ModelSender',
  full_name='ModelSender',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=191,
  serialized_end=488,
  methods=[
  _descriptor.MethodDescriptor(
    name='InitDevice',
    full_name='ModelSender.InitDevice',
    index=0,
    containing_service=None,
    input_type=_ANYSTRING,
    output_type=_ANYSTRING,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='IsEnoughClient',
    full_name='ModelSender.IsEnoughClient',
    index=1,
    containing_service=None,
    input_type=_ANYSTRING,
    output_type=_FLAG,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='SendModel',
    full_name='ModelSender.SendModel',
    index=2,
    containing_service=None,
    input_type=_FLAG,
    output_type=_MODEL,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='ReturnModel',
    full_name='ModelSender.ReturnModel',
    index=3,
    containing_service=None,
    input_type=_MODEL,
    output_type=_RESULT,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='IsEnoughModel',
    full_name='ModelSender.IsEnoughModel',
    index=4,
    containing_service=None,
    input_type=_TIMING,
    output_type=_FLAG,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='FederatedModel',
    full_name='ModelSender.FederatedModel',
    index=5,
    containing_service=None,
    input_type=_FLAG,
    output_type=_RESULT,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='FinishAction',
    full_name='ModelSender.FinishAction',
    index=6,
    containing_service=None,
    input_type=_ANYSTRING,
    output_type=_ANYSTRING,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_MODELSENDER)

DESCRIPTOR.services_by_name['ModelSender'] = _MODELSENDER

# @@protoc_insertion_point(module_scope)