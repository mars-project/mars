# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

from ..utils import check_random_state
from ..datasource import tensor as astensor
from .lu import lu
from .qr import qr
from .svd import svd
from .utils import svd_flip


# ---------------------------------------------------------------------
# Original implementation is in `sklearn.utils.extmath.randomized_svd`.
# ---------------------------------------------------------------------


def randomized_range_finder(A, size, n_iter,
                            power_iteration_normalizer='auto',
                            random_state=None):
    r"""Computes an orthonormal matrix whose range approximates the range of A.

    .. versionadded:: 0.1.3

    Parameters
    ----------
    A : 2D tensor
        The input data tensor

    size : integer
        Size of the return tensor

    n_iter : integer
        Number of power iterations used to stabilize the result

    power_iteration_normalizer : 'auto' (default), 'QR', 'LU', 'none'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter` <= 2 and switches to LU otherwise.

    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    Returns
    -------
    Q : 2D array
        A (size x size) projection matrix, the range of which
        approximates well the range of the input matrix A.

    Notes
    -----

    Follows Algorithm 4.3 of
    Finding structure with randomness: Stochastic algorithms for constructing
    approximate matrix decompositions
    Halko, et al., 2009 (arXiv:909) https://arxiv.org/pdf/0909.4061.pdf

    An implementation of a randomized algorithm for principal component
    analysis
    A. Szlam et al. 2014
    """
    random_state = check_random_state(random_state)

    # Generating normal random vectors with shape: (A.shape[1], size)
    Q = random_state.normal(size=(A.shape[1], size))
    if A.dtype.kind == 'f':
        # Ensure f32 is preserved as f32
        Q = Q.astype(A.dtype, copy=False)

    # Deal with "auto" mode
    if power_iteration_normalizer == 'auto':
        if n_iter <= 2:
            power_iteration_normalizer = 'none'
        else:
            power_iteration_normalizer = 'LU'

    # Perform power iterations with Q to further 'imprint' the top
    # singular vectors of A in Q
    for _ in range(n_iter):
        if power_iteration_normalizer == 'none':
            Q = A.dot(Q)
            Q = A.T.dot(Q)
        elif power_iteration_normalizer == 'LU':
            # TODO: directly get Q when lu supports `permute_l`
            p, l, _ = lu(A.dot(Q))
            Q = p.dot(l)
            p, l, _ = lu(A.T.dot(Q))
            Q = p.dot(l)
        elif power_iteration_normalizer == 'QR':
            Q, _ = qr(A.dot(Q))
            Q, _ = qr(A.T.dot(Q))

    # Sample the range of A using by linear projection of Q
    # Extract an orthonormal basis
    Q, _ = qr(A.dot(Q))
    return Q


def randomized_svd(M, n_components, n_oversamples=10, n_iter='auto',
                   power_iteration_normalizer='auto', transpose='auto',
                   flip_sign=True, random_state=0):
    r"""
    Computes a truncated randomized SVD

    .. versionadded:: 0.1.4

    Parameters
    ----------
    M : Tensor
        tensor to decompose
    n_components : int
        Number of singular values and vectors to extract.
    n_oversamples : int (default is 10)
        Additional number of random vectors to sample the range of M so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of M is n_components + n_oversamples. Smaller
        number can improve speed but can negatively impact the quality of
        approximation of singular vectors and singular values.
    n_iter : int or 'auto' (default is 'auto')
        Number of power iterations. It can be used to deal with very noisy
        problems. When 'auto', it is set to 4, unless `n_components` is small
        (< .1 * min(X.shape)) `n_iter` in which case is set to 7.
        This improves precision with few components.
    power_iteration_normalizer : 'auto' (default), 'QR', 'LU', 'none'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter` <= 2 and switches to LU otherwise.
    transpose : True, False or 'auto' (default)
        Whether the algorithm should be applied to M.T instead of M. The
        result should approximately be the same. The 'auto' mode will
        trigger the transposition if M.shape[1] > M.shape[0] since this
        implementation of randomized SVD tend to be a little faster in that
        case.
    flip_sign : boolean, (True by default)
        The output of a singular value decomposition is only unique up to a
        permutation of the signs of the singular vectors. If `flip_sign` is
        set to `True`, the sign ambiguity is resolved by making the largest
        loadings for each component in the left singular vectors positive.
    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.
    Notes
    -----
    This algorithm finds a (usually very good) approximate truncated
    singular value decomposition using randomization to speed up the
    computations. It is particularly fast on large matrices on which
    you wish to extract only a small number of components. In order to
    obtain further speed up, `n_iter` can be set <=2 (at the cost of
    loss of precision).
    References
    ----------
    * Finding structure with randomness: Stochastic algorithms for constructing
      approximate matrix decompositions
      Halko, et al., 2009 https://arxiv.org/abs/0909.4061
    * A randomized algorithm for the decomposition of matrices
      Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert
    * An implementation of a randomized algorithm for principal component
      analysis
      A. Szlam et al. 2014
    """
    M = astensor(M)
    random_state = check_random_state(random_state)
    n_random = n_components + n_oversamples
    n_samples, n_features = M.shape

    if n_iter == 'auto':
        # Check if the number of iterations is explicitly specified
        # Adjust n_iter. 7 was found a good compromise for PCA.
        # https://github.com/scikit-learn/scikit-learn/pull/5299
        n_iter = 7 if n_components < .1 * min(M.shape) else 4

    if transpose == 'auto':
        transpose = n_samples < n_features
    if transpose:
        # this implementation is a bit faster with smaller shape[1]
        M = M.T

    Q = randomized_range_finder(M, n_random, n_iter,
                                power_iteration_normalizer, random_state)
    # project M to the (k + p) dimensional space using the basis vectors
    B = Q.T.dot(M)

    # compute the SVD on the thin matrix: (k + p) wide
    Uhat, s, V = svd(B)

    U = Q.dot(Uhat)

    if flip_sign:
        if not transpose:
            U, V = svd_flip(U, V)
        else:
            # In case of transpose u_based_decision=false
            # to actually flip based on u and not v.
            U, V = svd_flip(U, V, u_based_decision=False)

    if transpose:
        # transpose back the results according to the input convention
        return V[:n_components, :].T, s[:n_components], U[:, :n_components].T
    else:
        return U[:, :n_components], s[:n_components], V[:n_components, :]
