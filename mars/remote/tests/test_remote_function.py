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

import numpy as np

from mars import tensor as mt
from mars.session import new_session, Session
from mars.remote import spawn
from mars.tests.core import TestBase, ExecutorForTest


class Test(TestBase):
    def setUp(self) -> None:
        super().setUp()
        self.executor = ExecutorForTest('numpy')
        self.ctx, self.executor = self._create_test_context(self.executor)
        self.ctx.__enter__()

    def tearDown(self) -> None:
        self.ctx.__exit__(None, None, None)

    def testRemoteFunction(self):
        def f1(x):
            return x + 1

        def f2(x, y, z=None):
            return x * y * (z[0] + z[1])

        rs = np.random.RandomState(0)
        raw1 = rs.rand(10, 10)
        raw2 = rs.rand(10, 10)

        r1 = spawn(f1, raw1)
        r2 = spawn(f1, raw2)
        r3 = spawn(f2, (r1, r2), {'z': [r1, r2]})

        result = self.executor.execute_tileables([r3])[0]
        expected = (raw1 + 1) * (raw2 + 1) * (raw1 + 1 + raw2 + 1)
        np.testing.assert_almost_equal(result, expected)

        with self.assertRaises(TypeError):
            spawn(f2, (r1, r2), kwargs=())

        session = new_session()

        def f():
            assert Session.default.session_id == session.session_id
            return mt.ones((2, 3)).sum().to_numpy()

        self.assertEqual(spawn(f).execute(session=session).fetch(session=session), 6)
