# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

from typing import List

import pytest

from mars.services.utils import SessionCache


@pytest.mark.asyncio
async def test_session_cache():
    invokes = 0

    async def mock_factory(session_id, v) -> List:
        nonlocal invokes
        invokes += 1
        return [session_id, v]

    sess_cache = SessionCache(mock_factory, 2)
    assert await sess_cache.create(v=1, session_id='0') == ['0', 1]
    assert invokes == 1
    assert await sess_cache.create(v=2, session_id='1') == ['1', 2]
    assert invokes == 2
    assert await sess_cache.create(v=3, session_id='2') == ['2', 3]
    assert invokes == 3

    assert await sess_cache.create(v=3, session_id='2') == ['2', 3]
    assert invokes == 3

    assert await sess_cache.create(v=1, session_id='0') == ['0', 1]
    assert invokes == 4
