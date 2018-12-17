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

from mars.deploy.local.distributor import gen_distributor
from mars.scheduler.session import SessionActor
from mars.scheduler.graph import GraphActor
from mars.worker.dispatcher import DispatchActor


class Test(unittest.TestCase):
    def testDistribute(self):
        distributor = gen_distributor(2, 4)

        idx = distributor.distribute(SessionActor.default_name())
        self.assertEqual(idx, 1)

        idx = distributor.distribute(GraphActor.gen_name('fake_session_id', 'fake_graph_key'))
        self.assertEqual(idx, 0)

        idx = distributor.distribute(DispatchActor.default_name())
        self.assertEqual(idx, 2)

        idx = distributor.distribute('w:1:mars-sender')
        self.assertEqual(idx, 3)
