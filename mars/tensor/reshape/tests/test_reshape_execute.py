import unittest
import numpy as np

from mars.executor import Executor
from mars.tensor.datasource import ones


class Test(unittest.TestCase):
    def setUp(self):
        self.executor = Executor('numpy')

    def testReshapeExecution(self):
        x = ones((1, 2, 3), chunk_size=[4, 3, 5])
        y = x.reshape(3, 2)
        res = self.executor.execute_tensor(y)[0]
        self.assertEqual(y.shape, (3, 2))
        np.testing.assert_equal(res, np.ones((3, 2)))
