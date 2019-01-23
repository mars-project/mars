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

import copy
import unittest

import numpy as np
try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from mars import utils

try:
    unicode  # noqa F821
except NameError:
    unicode = str


class Test(unittest.TestCase):
    def testStringConversion(self):
        s = None
        self.assertIsNone(utils.to_binary(s))
        self.assertIsNone(utils.to_str(s))
        self.assertIsNone(utils.to_text(s))

        s = 'abcdefg'
        self.assertIsInstance(utils.to_binary(s), bytes)
        self.assertEqual(utils.to_binary(s), b'abcdefg')
        self.assertIsInstance(utils.to_str(s), str)
        self.assertEqual(utils.to_str(s), 'abcdefg')
        self.assertIsInstance(utils.to_text(s), unicode)
        self.assertEqual(utils.to_text(s), u'abcdefg')

        ustr = type('ustr', (str,), {})
        self.assertIsInstance(utils.to_str(ustr(s)), str)
        self.assertEqual(utils.to_str(ustr(s)), 'abcdefg')

        s = b'abcdefg'
        self.assertIsInstance(utils.to_binary(s), bytes)
        self.assertEqual(utils.to_binary(s), b'abcdefg')
        self.assertIsInstance(utils.to_str(s), str)
        self.assertEqual(utils.to_str(s), 'abcdefg')
        self.assertIsInstance(utils.to_text(s), unicode)
        self.assertEqual(utils.to_text(s), u'abcdefg')

        ubytes = type('ubytes', (bytes,), {})
        self.assertIsInstance(utils.to_binary(ubytes(s)), bytes)
        self.assertEqual(utils.to_binary(ubytes(s)), b'abcdefg')

        s = u'abcdefg'
        self.assertIsInstance(utils.to_binary(s), bytes)
        self.assertEqual(utils.to_binary(s), b'abcdefg')
        self.assertIsInstance(utils.to_str(s), str)
        self.assertEqual(utils.to_str(s), 'abcdefg')
        self.assertIsInstance(utils.to_text(s), unicode)
        self.assertEqual(utils.to_text(s), u'abcdefg')

        uunicode = type('uunicode', (unicode,), {})
        self.assertIsInstance(utils.to_text(uunicode(s)), unicode)
        self.assertEqual(utils.to_text(uunicode(s)), u'abcdefg')

        with self.assertRaises(TypeError):
            utils.to_binary(utils)
        with self.assertRaises(TypeError):
            utils.to_str(utils)
        with self.assertRaises(TypeError):
            utils.to_text(utils)

    def testTokenize(self):
        v = (1, 2.3, '456', u'789', b'101112', None, np.ndarray,
             [912, 'uvw'], np.arange(0, 10), np.int64)
        self.assertEqual(utils.tokenize(v), utils.tokenize(copy.deepcopy(v)))

        v = {'a', 'xyz', 'uvw'}
        self.assertEqual(utils.tokenize(v), utils.tokenize(copy.deepcopy(v)))

        v = dict(x='abcd', y=98765)
        self.assertEqual(utils.tokenize(v), utils.tokenize(copy.deepcopy(v)))

        # pandas relative
        df = pd.DataFrame([[utils.to_binary('测试'), utils.to_text('数据')]],
                          index=['a'], columns=['中文', 'data'])
        v = [df, df.index, df.columns, df['data']]
        self.assertEqual(utils.tokenize(v), utils.tokenize(copy.deepcopy(v)))
