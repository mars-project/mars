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

import cloudpickle as pickle

from ...message import ErrorMessage


def test_as_instanceof_cause():
    fake_address = "Fake address"
    fake_pid = 123
    value = 3

    class CustomException(Exception):
        def __init__(self, i):
            self.i = i

        def __str__(self):
            return "Custom Exception."

    try:
        raise CustomException(value)
    except Exception as e:
        em = ErrorMessage(
            b"Fake message id", fake_address, fake_pid, type(e), e, e.__traceback__
        )
        assert "Fake message id" in repr(em)
        try:
            cause = em.as_instanceof_cause()
            # Test serialization.
            cause1 = pickle.loads(pickle.dumps(cause))
            assert type(cause) is type(cause1)
            raise cause
        except Exception as e1:
            e1 = pickle.loads(pickle.dumps(e1))
            # Check cause exception.
            assert isinstance(e1, CustomException)
            assert e1.i == value
            assert e1.address == fake_address
            assert e1.pid == fake_pid
            assert fake_address in str(e1)
            assert "Custom Exception" in str(e1)
            assert str(fake_pid) in str(e1)
            em1 = ErrorMessage(
                b"Fake message id",
                fake_address,
                fake_pid,
                type(e1),
                e1,
                e1.__traceback__,
            )
            try:
                raise em1.as_instanceof_cause()
            except Exception as e2:
                e2 = pickle.loads(pickle.dumps(e2))
                # Check recursive cause exception.
                assert isinstance(e2, CustomException)
                assert e2.i == value
                assert e2.address == fake_address
                assert e2.pid == fake_pid
                assert str(e2).count("Custom Exception") == 1
                assert str(e2).count(fake_address) == 1
                assert str(e2).count(str(fake_pid)) == 1
