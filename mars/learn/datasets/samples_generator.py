# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

from ... import tensor as mt
from ...tensor.expressions.utils import check_random_state
from ...tensor.expressions import linalg


# -------------------------------------------------------------------
# Original implementation is in `sklearn.datasets.samples_generator`.
# -------------------------------------------------------------------


def make_low_rank_matrix(n_samples=100, n_features=100, effective_rank=10,
                         tail_strength=0.5, random_state=None, chunk_size=None):
    """Generate a mostly low rank matrix with bell-shaped singular values

    Most of the variance can be explained by a bell-shaped curve of width
    effective_rank: the low rank part of the singular values profile is::

        (1 - tail_strength) * exp(-1.0 * (i / effective_rank) ** 2)

    The remaining singular values' tail is fat, decreasing as::

        tail_strength * exp(-0.1 * i / effective_rank).

    The low rank part of the profile can be considered the structured
    signal part of the data while the tail can be considered the noisy
    part of the data that cannot be summarized by a low number of linear
    components (singular vectors).

    This kind of singular profiles is often seen in practice, for instance:
     - gray level pictures of faces
     - TF-IDF vectors of text documents crawled from the web

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.

    n_features : int, optional (default=100)
        The number of features.

    effective_rank : int, optional (default=10)
        The approximate number of singular vectors required to explain most of
        the data by linear combinations.

    tail_strength : float between 0.0 and 1.0, optional (default=0.5)
        The relative importance of the fat noisy tail of the singular values
        profile.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The matrix.
    """
    generator = check_random_state(random_state)
    n = min(n_samples, n_features)

    # Random (ortho normal) vectors
    u, _ = linalg.qr(generator.randn(n_samples, n, chunk_size=chunk_size))
    v, _ = linalg.qr(generator.randn(n_features, n, chunk_size=chunk_size))

    # Index of the singular values
    singular_ind = mt.arange(n, dtype=mt.float64, chunk_size=chunk_size)

    # Build the singular profile by assembling signal and noise components
    low_rank = ((1 - tail_strength) *
                mt.exp(-1.0 * (singular_ind / effective_rank) ** 2))
    tail = tail_strength * mt.exp(-0.1 * singular_ind / effective_rank)
    s = mt.identity(n) * (low_rank + tail)

    return mt.dot(mt.dot(u, s), v.T)
