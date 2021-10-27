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
import scipy.sparse as sp
import pytest

from .... import tensor as mt
from ..sparsefuncs import count_nonzero


def test_count_nonzero(setup):
    X = np.array(
        [[0, 3, 0], [2, -1, 0], [0, 0, 0], [9, 8, 7], [4, 0, 5]], dtype=np.float64
    )
    X_nonzero = X != 0

    X_csr = sp.csr_matrix(X)
    X_csr_t = mt.tensor(X_csr, chunk_size=3)

    sample_weight = [0.5, 0.2, 0.3, 0.1, 0.1]
    X_nonzero_weighted = X_nonzero * np.array(sample_weight)[:, None]

    for axis in [0, 1, -1, -2, None]:
        np.testing.assert_array_almost_equal(
            count_nonzero(X_csr_t, axis=axis).execute().fetch(),
            X_nonzero.sum(axis=axis),
        )
        np.testing.assert_array_almost_equal(
            count_nonzero(X_csr_t, axis=axis, sample_weight=sample_weight)
            .execute()
            .fetch(),
            X_nonzero_weighted.sum(axis=axis),
        )

    with pytest.raises(ValueError):
        count_nonzero(X_csr_t, axis=2).execute()

    assert count_nonzero(X_csr_t, axis=0).dtype == count_nonzero(X_csr_t, axis=1).dtype
    assert (
        count_nonzero(X_csr_t, axis=0, sample_weight=sample_weight).dtype
        == count_nonzero(X_csr_t, axis=1, sample_weight=sample_weight).dtype
    )

    # Check dtypes with large sparse matrices too
    # XXX: test fails on 32bit (Windows/Linux)
    try:
        X_csr.indices = X_csr.indices.astype(np.int64)
        X_csr.indptr = X_csr.indptr.astype(np.int64)
        X_csr_t = mt.tensor(X_csr, chunk_size=3)

        assert (
            count_nonzero(X_csr_t, axis=0).dtype == count_nonzero(X_csr_t, axis=1).dtype
        )
        assert (
            count_nonzero(X_csr_t, axis=0, sample_weight=sample_weight).dtype
            == count_nonzero(X_csr_t, axis=1, sample_weight=sample_weight).dtype
        )
    except TypeError as e:
        assert "according to the rule 'safe'" in e.args[0] and np.intp().nbytes < 8, e
