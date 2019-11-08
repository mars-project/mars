#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import threading

from mars.tests.core import TestBase
from mars.config import options, option_context, is_integer, is_string


class Test(TestBase):
    def testConfigContext(self):
        with self.assertRaises(AttributeError):
            _ = options.a.b.c

        options.register_option('c.d.e', 'a', is_string)

        try:
            with option_context() as ctx:
                ctx.register_option('a.b.c', 1, validator=is_integer)
                self.assertEqual(ctx.a.b.c, 1)

                ctx.a.b.c = 2
                self.assertEqual(ctx.a.b.c, 2)

                with self.assertRaises(ValueError):
                    ctx.a.b.c = 'a'

                self.assertEqual(ctx.c.d.e, 'a')

                ctx.c.d.e = 'b'

            self.assertEqual(options.c.d.e, 'a')

            options.c.d.e = 'c'

            self.assertEqual(options.c.d.e, 'c')

            with self.assertRaises(AttributeError):
                _ = options.a.b.c  # noqa: F841
        finally:
            options.unregister_option('c.d.e')

    def testMultiThreadConfig(self):
        options.register_option('a.b.c', 1)
        assert_equal = self.assertEqual

        class T(threading.Thread):
            def __init__(self, is_first, condition):
                super(T, self).__init__()
                self.is_first = is_first
                self.condition = condition

            def run(self):
                self.condition.acquire()
                if self.is_first:
                    options.a.b.c = 2
                    self.condition.notify()
                else:
                    self.condition.wait()
                    assert_equal(options.a.b.c, 1)
                self.condition.release()

        try:
            cond = threading.Condition()
            a = T(True, cond)
            b = T(False, cond)
            b.start()
            a.start()
            a.join()
            b.join()
        finally:
            options.unregister_option('a.b.c')
