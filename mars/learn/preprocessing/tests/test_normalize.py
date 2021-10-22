# Copyright 1999-2021 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import scipy.sparse as sps
import pytest
from sklearn.preprocessing import normalize as sk_normalize

from .... import tensor as mt
from .. import normalize


def test_normalize_op():
    with pytest.raises(ValueError):
        normalize(mt.random.random(10, 3), norm="unknown")

    with pytest.raises(ValueError):
        normalize(mt.random.random(10, 3), axis=-1)

    with pytest.raises(ValueError):
        normalize(mt.random.rand(10, 3, 3))


def test_normalize_execution(setup):
    raw_dense = np.random.rand(10, 10)
    raw_sparse = sps.random(10, 10, density=0.4, format="csr")

    for chunk_size in [10, 6, (10, 6), (6, 10)]:
        for raw, x in [
            (raw_dense, mt.tensor(raw_dense, chunk_size=chunk_size)),
            (raw_sparse, mt.tensor(raw_sparse, chunk_size=chunk_size)),
        ]:
            for norm in ["l1", "l2", "max"]:
                for axis in (0, 1):
                    for use_sklearn in [True, False]:
                        n = normalize(x, norm=norm, axis=axis, return_norm=False)
                        n.op._use_sklearn = use_sklearn

                        result = n.execute().fetch()
                        expected = sk_normalize(
                            raw, norm=norm, axis=axis, return_norm=False
                        )

                        if sps.issparse(expected):
                            expected = expected.A
                        np.testing.assert_almost_equal(np.asarray(result), expected)

    raw_dense = np.random.rand(10, 10)
    raw_sparse = sps.random(10, 10, density=0.4, format="csr")

    # test copy and return_normalize
    for axis in (0, 1):
        for chunk_size in (10, 6, (6, 10)):
            for raw in (raw_dense, raw_sparse):
                x = mt.tensor(raw, chunk_size=chunk_size)
                n = normalize(x, axis=axis, copy=False, return_norm=True)

                results = n.execute().fetch()
                raw_copy = raw.copy()
                try:
                    expects = sk_normalize(
                        raw_copy, axis=axis, copy=False, return_norm=True
                    )
                except NotImplementedError:
                    continue

                if sps.issparse(expects[0]):
                    expected = expects[0].A
                else:
                    expected = expects[0]
                np.testing.assert_almost_equal(np.asarray(results[0]), expected)
                np.testing.assert_almost_equal(results[1], expects[1])
