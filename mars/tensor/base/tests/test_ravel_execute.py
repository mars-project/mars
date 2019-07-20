import unittest
import numpy as np

from mars.executor import Executor
from mars.tensor.datasource import ones
import mars.tensor as mt


class Test(unittest.TestCase):
    def setUp(self):
        self.executor = Executor('numpy')

    def testRavelExecution(self):
        arr = ones((10, 5), chunk_size=2)
        flat_arr = mt.ravel(arr)

        res = self.executor.execute_tensor(flat_arr, concat=True)[0]
        self.assertEqual(len(res), 50)
        np.testing.assert_equal(res, np.ones(50))
