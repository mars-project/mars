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
import enum

from ... import opcodes
from ...serialization.serializables import (
    FieldTypes,
    StringField,
    ListField,
    Int32Field,
    ReferenceField,
)
from .base import Operand
from .core import TileableOperandMixin


class Fetch(Operand):
    _op_type_ = opcodes.FETCH

    source_key = StringField("source_key", default=None)


class FetchMixin(TileableOperandMixin):
    def check_inputs(self, inputs):
        # no inputs
        if inputs and len(inputs) > 0:
            raise ValueError(f"{type(self).__name__} has no inputs")

    @classmethod
    def tile(cls, op):
        raise NotImplementedError("Fetch tile cannot be handled by operand itself")

    @classmethod
    def execute(cls, ctx, op):
        """
        Fetch operand needs nothing to do.
        """


class FetchShuffle(Operand):
    _op_type_ = opcodes.FETCH_SHUFFLE

    source_keys = ListField("source_keys", FieldTypes.string)
    n_mappers = Int32Field("n_mappers")
    n_reducers = Int32Field("n_reducers")
    shuffle_fetch_type = ReferenceField("shuffle_fetch_type")


class ShuffleFetchType(enum.Enum):
    FETCH_BY_KEY = 0
    FETCH_BY_INDEX = 1
