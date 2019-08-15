import unittest

import numpy as np

from mars.executor import Executor
import mars.tensor as mt
from mars.tensor.fuse.jax import JAX_INSTALLED


@unittest.skipIf(JAX_INSTALLED is False, 'jax not installed')
class Test(unittest.TestCase):
    # test multiple engines execution
    def setUp(self):
        self.executor = Executor(['jax', 'numexpr'])

    def testUnaryExecution(self):
        executor_numpy = Executor('numpy')
        a = mt.ones((2, 2))
        # a = a * (-1)
        c = mt.abs(a)
        d = mt.abs(c)
        e = mt.abs(d)
        f = mt.abs(e)
        result = self.executor.execute_tensor(f, concat=True)
        expected = executor_numpy.execute_tensor(f, concat=True)
        np.testing.assert_equal(result, expected)
