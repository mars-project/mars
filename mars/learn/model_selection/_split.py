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

import numbers
from abc import ABCMeta, abstractmethod
from itertools import chain
from math import ceil, floor

import numpy as np

from ...core import ExecutableTuple
from ... import tensor as mt
from ...tensor.utils import check_random_state
from ..utils import shuffle as shuffle_arrays
from ..utils.validation import indexable, _num_samples


def train_test_split(*arrays, **options):
    """Split arrays or matrices into random train and test subsets

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.

    test_size : float, int or None, optional (default=None)
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25.

    train_size : float, int, or None, (default=None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    shuffle : boolean, optional (default=True)
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.

    stratify : array-like or None (default=None)
        If not None, data is split in a stratified fashion, using this as
        the class labels.

    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> from mars.learn.model_selection import train_test_split
    >>> X, y = mt.arange(10).reshape((5, 2)), range(5)
    >>> X.execute()
    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7],
           [8, 9]])
    >>> list(y)
    [0, 1, 2, 3, 4]

    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.33, random_state=42)
    ...
    >>> X_train.execute()
    array([[8, 9],
           [0, 1],
           [4, 5]])
    >>> y_train.execute()
    array([4, 0, 2])
    >>> X_test.execute()
    array([[2, 3],
           [6, 7]])
    >>> y_test.execute()
    array([1, 3])

    >>> train_test_split(y, shuffle=False)
    [array([0, 1, 2]), array([3, 4])]

    """

    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")
    test_size = options.pop('test_size', None)
    train_size = options.pop('train_size', None)
    random_state = options.pop('random_state', None)
    stratify = options.pop('stratify', None)
    shuffle = options.pop('shuffle', True)
    session = options.pop('session', None)
    run_kwargs = options.pop('run_kwargs', None)

    if options:
        raise TypeError(f"Invalid parameters passed: {options}")

    arrays = indexable(*arrays, session=session, run_kwargs=run_kwargs)

    n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(n_samples, test_size, train_size,
                                              default_test_size=0.25)

    if shuffle is False:
        if stratify is not None:  # pragma: no cover
            raise ValueError(
                "Stratified train/test split is not implemented for "
                "shuffle=False")

        iterables = ((a[:n_train], a[n_train: n_train + n_test])
                     for a in arrays)
    else:
        if stratify is not None:  # pragma: no cover
            raise NotImplementedError('stratify is not implemented yet')
        else:
            shuffled_arrays = shuffle_arrays(
                *arrays, random_state=random_state)
            if not isinstance(shuffled_arrays, tuple):
                shuffled_arrays = (shuffled_arrays,)
            iterables = ((a[:n_train], a[n_train: n_train + n_test])
                         for a in shuffled_arrays)

    return list(ExecutableTuple(chain.from_iterable(iterables)).execute(
        session=session, **(run_kwargs or dict())))


def _validate_shuffle_split(n_samples, test_size, train_size,
                            default_test_size=None):
    """
    Validation helper to check if the test/test sizes are meaningful wrt to the
    size of the data (n_samples)
    """
    if test_size is None and train_size is None:
        test_size = default_test_size

    test_size_type = np.asarray(test_size).dtype.kind
    train_size_type = np.asarray(train_size).dtype.kind

    if (test_size_type == 'i' and (test_size >= n_samples or test_size <= 0)
            or test_size_type == 'f' and (test_size <= 0 or test_size >= 1)):
        raise ValueError(f'test_size={test_size} should be either positive and smaller'
                         f' than the number of samples {n_samples} or a float in the '
                         '(0, 1) range')

    if (train_size_type == 'i' and (train_size >= n_samples or train_size <= 0)
            or train_size_type == 'f' and (train_size <= 0 or train_size >= 1)):
        raise ValueError(f'train_size={train_size} should be either positive and smaller'
                         f' than the number of samples {n_samples} or a float in the '
                         '(0, 1) range')

    if train_size is not None and train_size_type not in ('i', 'f'):  # pragma: no cover
        raise ValueError(f"Invalid value for train_size: {train_size}")
    if test_size is not None and test_size_type not in ('i', 'f'):
        raise ValueError(f"Invalid value for test_size: {test_size}")

    if (train_size_type == 'f' and test_size_type == 'f' and
            train_size + test_size > 1):
        raise ValueError(
            f'The sum of test_size and train_size = {train_size + test_size}, '
            'should be in the (0, 1) range. Reduce test_size and/or train_size.')

    if test_size_type == 'f':
        n_test = ceil(test_size * n_samples)
    elif test_size_type == 'i':
        n_test = float(test_size)

    if train_size_type == 'f':
        n_train = floor(train_size * n_samples)
    elif train_size_type == 'i':  # pragma: no cover
        n_train = float(train_size)

    if train_size is None:
        n_train = n_samples - n_test
    elif test_size is None:
        n_test = n_samples - n_train

    if n_train + n_test > n_samples:  # pragma: no cover
        raise ValueError(f'The sum of train_size and test_size = {n_train + n_test}, '
                         f'should be smaller than the number of samples {n_samples}. '
                         'Reduce test_size and/or train_size.')

    n_train, n_test = int(n_train), int(n_test)

    if n_train == 0:  # pragma: no cover
        raise ValueError(
            f'With n_samples={n_samples}, test_size={test_size} and '
            f'train_size={train_size}, the resulting train set will '
            f'be empty. Adjust any of the aforementioned parameters.'
        )

    return n_train, n_test


class BaseCrossValidator(metaclass=ABCMeta):
    """Base class for all cross-validators

    Implementations must define `_iter_test_masks` or `_iter_test_indices`.
    """
    def split(self, X, y=None, groups=None):  # pragma: no cover
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        indices = mt.arange(_num_samples(X))
        for test_index in self._iter_test_masks(X, y, groups):
            train_index = indices[mt.logical_not(test_index)]
            test_index = indices[test_index]
            yield train_index, test_index

    # Since subclasses must implement either _iter_test_masks or
    # _iter_test_indices, neither can be abstract.
    def _iter_test_masks(self, X=None, y=None, groups=None):  # pragma: no cover
        """Generates boolean masks corresponding to test sets.

        By default, delegates to _iter_test_indices(X, y, groups)
        """
        for test_index in self._iter_test_indices(X, y, groups):
            test_mask = mt.zeros(_num_samples(X), dtype=bool)
            test_mask[test_index] = True
            yield test_mask

    def _iter_test_indices(self, X=None, y=None, groups=None):  # pragma: no cover
        """Generates integer indices corresponding to test sets."""
        raise NotImplementedError

    @abstractmethod
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""


class _BaseKFold(BaseCrossValidator, metaclass=ABCMeta):
    """Base class for KFold, GroupKFold, and StratifiedKFold"""

    @abstractmethod
    def __init__(self, n_splits, *, shuffle, random_state):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError('The number of folds must be of Integral type. '
                             '%s of type %s was passed.'
                             % (n_splits, type(n_splits)))
        n_splits = int(n_splits)

        if n_splits <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=2 or more,"
                " got n_splits={0}.".format(n_splits))

        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False;"
                            " got {0}".format(shuffle))

        if not shuffle and random_state is not None:  # None is the default
            raise ValueError(
                'Setting a random_state has no effect since shuffle is '
                'False. You should leave '
                'random_state to its default (None), or set shuffle=True.',
            )

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits


class KFold(_BaseKFold):
    """K-Folds cross-validator

    Provides train/test indices to split data in train/test sets. Split
    dataset into k consecutive folds (without shuffling by default).

    Each fold is then used once as a validation while the k - 1 remaining
    folds form the training set.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.

    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.
        Note that the samples within each split will not be shuffled.

    random_state : int, RandomState instance or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold. Otherwise, this
        parameter has no effect.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> from mars.learn.model_selection import KFold
    >>> X = mt.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = mt.array([1, 2, 3, 4])
    >>> kf = KFold(n_splits=2)
    >>> kf.get_n_splits(X)
    2
    >>> print(kf)
    KFold(n_splits=2, random_state=None, shuffle=False)
    >>> for train_index, test_index in kf.split(X):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    TRAIN: [2 3] TEST: [0 1]
    TRAIN: [0 1] TEST: [2 3]

    Notes
    -----
    The first ``n_samples % n_splits`` folds have size
    ``n_samples // n_splits + 1``, other folds have size
    ``n_samples // n_splits``, where ``n_samples`` is the number of samples.

    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting `random_state`
    to an integer.

    See Also
    --------
    StratifiedKFold : Takes group information into account to avoid building
        folds with imbalanced class distributions (for binary or multiclass
        classification tasks).

    GroupKFold : K-fold iterator variant with non-overlapping groups.

    RepeatedKFold : Repeats K-Fold n times.
    """
    def __init__(self, n_splits=5, *, shuffle=False,
                 random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        if self.n_splits > n_samples:
            raise ValueError(
                ("Cannot have number of splits n_splits={0} greater"
                 " than the number of samples: n_samples={1}.")
                .format(self.n_splits, n_samples))

        indices = mt.arange(n_samples)
        if self.shuffle:
            check_random_state(self.random_state).shuffle(indices)

        n_splits = self.n_splits
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
        fold_sizes[:n_samples % n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            train_index = mt.concatenate([indices[:start], indices[stop:]])
            yield train_index, indices[start:stop]
            current = stop
