import unittest

import numpy as np
import pandas as pd

import mars.dataframe as md
from mars.learn.neighbors._proxima2 import build_proxima2_index


class Test(unittest.TestCase):
    def testProxima2(self):
        df = pd.DataFrame(np.random.rand(100, 10))
        mdf = md.DataFrame(df, chunk_size=20)
        tensor = mdf.to_tensor()
        # pk = list()
        pk = np.array(list(range(100))).reshape(100, 1)
        build_proxima2_index(tensor, pk)
