import unittest

import numpy as np

from mars.executor import Executor
import mars.tensor as mt


class Test(unittest.TestCase):
    def setUp(self):
        self.executor = Executor('jax')

    def testUnaryExecution(self):
        executor_numpy = Executor('numpy')
        a = mt.ones((2, 2))
        a = a * (-1)
        c = mt.abs(a)
        d = mt.abs(c)
        e = mt.abs(d)
        f = mt.abs(e)
        result = self.executor.execute_tensor(f, concat=True)
        expected = executor_numpy.execute_tensor(f, concat=True)
        np.testing.assert_equal(result, expected)
