import unittest
import numpy as np

from mars.executor import Executor
from mars.tensor.datasource.arange import arange


class Test(unittest.TestCase):
    def setUp(self):
        self.executor = Executor('numpy')

    def testRavelExecution(self):
        array = np.arange(3)
        array2 = np.arange(3.0)
        array3 = np.arange(3, 7)
        array4 = np.arange(3, 7, 2)

        arr = arange(3)
        arr2 = arange(3.0)
        arr3 = arange(3, 7)
        arr4 = arange(3, 7, 2)
        res = self.executor.execute_tensor(arr, concat=True)[0]
        np.testing.assert_equal(res, array)
        res2 = self.executor.execute_tensor(arr2, concat=True)[0]
        np.testing.assert_equal(res2, array2)
        res3 = self.executor.execute_tensor(arr3, concat=True)[0]
        np.testing.assert_equal(res3, array3)
        res4 = self.executor.execute_tensor(arr4, concat=True)[0]
        np.testing.assert_equal(res4, array4)
