# Copyright 2022 XProbe Inc.
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

from ... import remote as mr
from ..context import get_context


def test_context(setup):
    def func():
        ctx = get_context()
        assert ctx is not None

    # no error should happen
    mr.spawn(func).execute()

    # context should be reset after execution
    # for test backend(test://xxx),
    # the worker pool and client are in the same process
    # if context is not reset, get_context() will still get one
    assert get_context() is None
