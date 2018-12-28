#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import os
import sys

from mars.serialize.protos.operand_pb2 import OperandDef

dirpath = os.path.dirname
parent_path = dirpath(dirpath(dirpath(__file__)))

sys.path.insert(0, dirpath(parent_path))


with open(os.path.join(parent_path, 'opcodes.py'), 'w') as f:
    f.write('# Generated automatically.  DO NOT EDIT!\n')
    for val, desc in OperandDef.OperandType.DESCRIPTOR.values_by_number.items():
        f.write('{0} = {1!r}\n'.format(desc.name, val))
