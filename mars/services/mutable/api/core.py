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

from abc import ABC, abstractmethod


class AbstractMutableAPI(ABC):

    @abstractmethod
    async def create_mutable_tensor(self,
                                    session_id: str,
                                    shape: tuple,
                                    dtype: str,
                                    chunk_size,
                                    name: str = None,
                                    default_value=0):
        '''
        create mutable tensor
        '''

    @abstractmethod
    async def get_mutable_tensor(self, session_id: str, name: str):
        '''
        get mutable tensor
        '''
