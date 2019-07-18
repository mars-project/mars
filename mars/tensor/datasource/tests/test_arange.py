import unittest

import numpy as np
from mars.tensor.datasource.arange import arange


class Test(unittest.TestCase):
    def testArange(self):
        array = np.arange(3)
        array2 = np.arange(3.0)
        array3 = np.arange(3, 7)
        array4 = np.arange(3, 7, 2)

        arr = arange(3)
        arr2 = arange(3.0)
        arr3 = arange(3, 7)
        arr4 = arange(3, 7, 2)
        self.assertEqual(arr.shape, array.shape)
        self.assertEqual(arr2.shape, array2.shape)
        self.assertEqual(arr2.dtype, 'float64')
        self.assertEqual(arr3.shape, array3.shape)
        self.assertEqual(arr4.shape, array4.shape)
