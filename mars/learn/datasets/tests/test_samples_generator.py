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

from mars.learn.datasets.samples_generator import make_low_rank_matrix
from mars.tensor.linalg import svd


class Test(unittest.TestCase):
    def testMakeLowRankMatrix(self):
        X = make_low_rank_matrix(n_samples=50, n_features=25, effective_rank=5,
                                 tail_strength=0.01, random_state=0)

        self.assertEqual(X.shape, (50, 25), "X shape mismatch")

        _, s, _ = svd(X)
        self.assertLess((s.sum() - 5).execute(n_parallel=1), 0.1, "X rank is not approximately 5")
