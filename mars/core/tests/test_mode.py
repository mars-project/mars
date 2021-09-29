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

import pytest

from .. import enter_mode, is_eager_mode, is_kernel_mode, is_build_mode


def test_enter_mode():
    from ...config import option_context, options

    @enter_mode(kernel=True)
    def wrapped():
        return is_eager_mode()

    assert not options.eager_mode
    assert not wrapped()

    with option_context({"eager_mode": True}):
        assert options.eager_mode
        assert not wrapped()

    @enter_mode(kernel=True)
    def wrapped2():
        wrapped()
        with option_context({"eager_mode": True}):
            assert options.eager_mode
            assert not is_eager_mode()
            with enter_mode(kernel=False):
                assert not is_kernel_mode()
            assert is_kernel_mode()

    wrapped2()

    assert not is_kernel_mode()
    assert not is_build_mode()

    @enter_mode(kernel=False)
    def wrapped3():
        wrapped()
        with option_context({"eager_mode": True}):
            assert options.eager_mode
            assert not is_kernel_mode()
            with enter_mode(kernel=True, build=True):
                assert is_kernel_mode()
                assert is_build_mode()
            assert not is_kernel_mode()
            assert not is_build_mode()
            with pytest.raises(ValueError):
                with enter_mode(kernel=True, build=True):
                    raise ValueError("meant to raise error")
            assert not is_kernel_mode()
            assert not is_build_mode()

            @enter_mode(kernel=True)
            def wrapped4():
                raise ValueError("meant to raise error")

            with pytest.raises(ValueError):
                wrapped4()
            assert not is_kernel_mode()
            assert not is_build_mode()

    wrapped3()
