import unittest
import numpy as np

from mars.executor import Executor
from mars.tensor.datasource import ones


class Test(unittest.TestCase):
    def setUp(self):
        self.executor = Executor('numpy')

    def testReshapeExecution(self):
        x = ones((10, 20, 30), chunk_size=[4, 3, 5])
        y = x.reshape(300, 20)
        res = self.executor.execute_tensor(y)
        self.assertEqual(y.shape, (300, 20))
        self.assertEqual(res, np.ones(300, 20))
