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


class A:
    def __init__(self):
        self.value = ["A"]

    def test_method(self):
        return ["A"]

    @classmethod
    def test_classmethod(cls):
        return ["A"]


class B(A):
    def __init__(self):
        super().__init__()
        self.value += ["B"]

    def test_method(self):
        return super().test_method() + ["B"]

    def test_method2(self):
        return super().test_method() + ["BB"]

    @classmethod
    def test_classmethod(cls):
        return super().test_classmethod() + ["B"]

    @classmethod
    def test_classmethod2(cls):
        return super().test_classmethod() + ["BB"]


class C(B):
    def __init__(self):
        super().__init__()
        self.value += ["C"]

    def test_method(self):
        return super().test_method() + ["C"]

    @classmethod
    def test_classmethod(cls):
        return super().test_classmethod() + ["C"]


class Dummy:
    pass


def test_patch_super():
    from ...tests.core import patch_cls, patch_super as super

    @patch_cls(B)
    class D(B):
        def __init__(self):
            super().__init__()
            self.value += ["D"]

        def test_method(self):
            return super().test_method() + super().test_method2() + ["D"]

        @classmethod
        def test_classmethod(cls):
            return super().test_classmethod() + super().test_classmethod2() + ["D"]

    b = B()
    assert B.test_classmethod() == ["A", "B", "A", "BB", "D"]
    assert b.test_method() == ["A", "B", "A", "BB", "D"]
    assert b.value == ["A", "B", "D"]

    c = C()
    assert C.test_classmethod() == ["A", "B", "A", "BB", "D", "C"]
    assert c.test_method() == ["A", "B", "A", "BB", "D", "C"]
    assert c.value == ["A", "B", "D", "C"]

    @patch_cls(Dummy)
    class E:
        def __init__(self):
            super().__init__()

        def test_method(self):
            return super().test_method() + ["D"]

        @classmethod
        def test_classmethod(cls):
            return super().test_classmethod() + ["D"]

    dummy = Dummy()
    with pytest.raises(AttributeError):
        dummy.test_method()
    with pytest.raises(AttributeError):
        Dummy.test_classmethod()
