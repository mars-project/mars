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

import numpy as np
try:
    import scipy.sparse as sps
    import sklearn
    from sklearn.utils.multiclass import \
        is_multilabel as sklearn_is_multilabel, \
        type_of_target as sklearn_type_of_target
except ImportError:  # pragma: no cover
    sklearn = None

import mars.tensor as mt
from mars.learn.utils.multiclass import is_multilabel, type_of_target


@unittest.skipIf(sklearn is None, 'scikit-learn is not installed')
class Test(unittest.TestCase):
    def testIsMultilabel(self):
        raws = [
            [[1, 2]],
            [0, 1, 0, 1],
            [[1], [0, 2], []],
            np.array([[1, 0], [0, 0]]),
            np.array([[1], [0], [0]]),
            np.array([[1, 0, 0]]),
            np.array([[1., 0.], [0., 0.]]),
            sps.csr_matrix([[1, 0], [0, 1]]),
        ]

        for raw in raws:
            self.assertEqual(is_multilabel(raw).execute(),
                             sklearn_is_multilabel(raw),
                             'raw: {}'.format(raw))

        t = mt.tensor(raws[3], chunk_size=1)
        self.assertEqual(is_multilabel(t).execute(),
                         sklearn_is_multilabel(raws[3]))

    def testTypeOfTarget(self):
        raws = [
            np.array([[0, 1], [0, 0]]),  # multilabel
            np.random.randint(2, size=(5, 3, 3)),  # ndim > 2, unknown
            np.array([[]]),  # ndim == 2, shape[1] == 0, unknown
            np.array([[1, 2], [1, 2]]),
            np.array([1, 2, 3]),
            np.array([.1, .2, 3]),
            np.array([[.1, .2, 3]]),
            np.array([[1., .2]]),
            np.array([[1., 2., 3]]),
            np.array([[1, 2]]),
            np.array([1, 2]),
            np.array([["a"], ["b"]], dtype=object),
            [[1, 2]],
            [],  # empty list
        ]

        for raw in raws:
            self.assertEqual(type_of_target(raw).execute(),
                             sklearn_type_of_target(raw))

        t = mt.tensor(raws[0], chunk_size=1)
        self.assertEqual(type_of_target(t).execute(),
                         sklearn_type_of_target(raws[0]))

        with self.assertRaises(ValueError):
            type_of_target('sth')
