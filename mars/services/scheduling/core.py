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

from ...serialization.serializables import Serializable, Float64Field, \
    Int32Field, Int64Field, StringField


class WorkerSlotInfo(Serializable):
    slot_id: int = Int32Field('slot_id')
    session_id: str = StringField('session_id')
    subtask_id: str = StringField('subtask_id')
    processor_usage: float = Float64Field('processor_usage')


class QuotaInfo(Serializable):
    quota_size: int = Int64Field('quota_size')
    allocated_size: int = Int64Field('allocated_size')
    hold_size: int = Int64Field('hold_size')
