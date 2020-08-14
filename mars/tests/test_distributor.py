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

import unittest

from mars.distributor import MarsDistributor
from mars.lib.mmh3 import hash as mmh_hash


class Test(unittest.TestCase):
    def testDistributor(self):
        distributor = MarsDistributor(1)
        self.assertEqual(distributor.distribute('Actor'), 0)

        distributor = MarsDistributor(3)
        self.assertEqual(distributor.distribute('w:1:Actor'), 1)
        self.assertEqual(distributor.distribute('w:-1:Actor'), 2)

        uid = 'w:h:Actor'
        self.assertEqual(distributor.distribute(uid),
                         mmh_hash(uid) % distributor.n_process)

        uid = 'w:h1:Actor'
        self.assertEqual(distributor.distribute(uid),
                         1 + mmh_hash(uid) % (distributor.n_process - 1))

        with self.assertRaises(ValueError):
            distributor.distribute('Actor')
        with self.assertRaises(ValueError):
            distributor.distribute('w:x:Actor')

        distributor = MarsDistributor(3, 'w:2:')
        self.assertEqual(distributor.distribute('Actor'), 2)
        self.assertEqual(distributor.distribute('w:x:Actor'), 2)

        distributor = MarsDistributor(3)
        ref_uid = 'w:h1:Actor2'
        new_uid = distributor.make_same_process('w::Actor', ref_uid)
        expect_worker_id = 1 + mmh_hash(ref_uid) % (distributor.n_process - 1)
        self.assertEqual(new_uid, f'w:{expect_worker_id}:Actor')

        with self.assertRaises(ValueError):
            distributor.make_same_process('Actor', ref_uid)

        distributor = MarsDistributor(3, 'w:2:')
        self.assertEqual('w:2:' + repr('Actor'),
                         distributor.make_same_process('Actor', 'w:2:Actor'))
