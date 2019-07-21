import unittest

from mars.tensor.datasource import ones
import mars.tensor as mt


class Test(unittest.TestCase):
    def testRavel(self):
        arr = ones((10, 5), chunk_size=2)
        flat_arr = mt.ravel(arr)
        self.assertEqual(flat_arr.shape, (50,))
