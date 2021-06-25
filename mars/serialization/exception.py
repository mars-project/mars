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

import pickle  # nosec  # pylint: disable=import_pickle
from typing import Dict, List, Union

from .core import Serializer, buffered, pickle_buffers, unpickle_buffers


class UnpickleableError(Exception):
    def __init__(self, raw_error: Union[str, Exception]):
        if isinstance(raw_error, str):
            super().__init__(raw_error)
        else:
            super().__init__(f'Error cannot be pickled, '
                             f'error type: {type(raw_error)}, '
                             f'raw error:\n{raw_error}')


class ExceptionSerializer(Serializer):
    serializer_name = 'pickle'

    @buffered
    def serialize(self, obj: Exception, context: Dict):
        try:
            buffers = pickle_buffers(obj)
        except (TypeError, pickle.PicklingError):
            buffers = pickle_buffers(UnpickleableError(obj))
        return {}, buffers

    def deserialize(self, header: Dict, buffers: List, context: Dict):
        return unpickle_buffers(buffers)


ExceptionSerializer.register(Exception)
