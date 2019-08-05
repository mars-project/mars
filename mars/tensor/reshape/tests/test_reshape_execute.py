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

        y2 = x.reshape(3, 2, order='F')
        res = self.executor.execute_tensor(y2)[0]
        self.assertTrue(res.flags['F_CONTIGUOUS'])
        self.assertFalse(res.flags['C_CONTIGUOUS'])

    def testShuffleReshapeExecution(self):
        a = ones((31, 27), chunk_size=10)
        b = a.reshape(27, 31)
        b.op.extra_params['_reshape_with_shuffle'] = True

        res = self.executor.execute_tensor(b, concat=True)[0]
        np.testing.assert_array_equal(res, np.ones((27, 31)))

        b2 = a.reshape(27, 31, order='F')
        b.op.extra_params['_reshape_with_shuffle'] = True
        res = self.executor.execute_tensor(b2)[0]
        self.assertTrue(res.flags['F_CONTIGUOUS'])
        self.assertFalse(res.flags['C_CONTIGUOUS'])
