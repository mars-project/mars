#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

import unittest

from mars.tensor.execution.core import Executor
from mars.tensor import random
from mars.session import LocalSession, Session


class Test(unittest.TestCase):
    def setUp(self):
        self.executor = Executor('numpy')
        local_session = LocalSession()
        local_session._executor = self.executor
        self.session = Session()
        self.session._sess = local_session

    def testDecref(self):
        a = random.rand(10, 20, chunk_size=5)
        b = a + 1

        b.execute(session=self.session)

        self.assertEqual(len(self.executor.chunk_result), 8)

        del b
        # decref called
        self.assertEqual(len(self.executor.chunk_result), 0)
