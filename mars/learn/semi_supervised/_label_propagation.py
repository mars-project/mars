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

import warnings
from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning

from ... import tensor as mt
from ...core import ExecutableTuple
from ..base import ClassifierMixin
from ..metrics.pairwise import rbf_kernel
from ..neighbors.unsupervised import NearestNeighbors
from ..utils import check_array
from ..utils.multiclass import check_classification_targets
from ..utils.validation import check_is_fitted, check_X_y


class BaseLabelPropagation(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for label propagation module.

    Parameters
    ----------
    kernel : {'knn', 'rbf', callable}
        String identifier for kernel function to use or the kernel function
        itself. Only 'rbf' and 'knn' strings are valid inputs. The function
        passed should take two inputs, each of shape [n_samples, n_features],
        and return a [n_samples, n_samples] shaped weight matrix

    gamma : float
        Parameter for rbf kernel

    n_neighbors : integer > 0
        Parameter for knn kernel

    alpha : float
        Clamping factor

    max_iter : integer
        Change maximum number of iterations allowed

    tol : float
        Convergence tolerance: threshold to consider the system at steady
        state
    """
    def __init__(self, kernel='rbf', gamma=20, n_neighbors=7,
                 alpha=1, max_iter=30, tol=1e-3):

        self.max_iter = max_iter
        self.tol = tol

        # kernel parameters
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors

        # clamping factor
        self.alpha = alpha

        self.nn_fit = None

    def _get_kernel(self, X, y=None):
        if self.kernel == "rbf":
            if y is None:
                return rbf_kernel(X, X, gamma=self.gamma)
            else:
                return rbf_kernel(X, y, gamma=self.gamma)
        elif self.kernel == "knn":
            if self.nn_fit is None:
                self.nn_fit = NearestNeighbors(self.n_neighbors).fit(X)
            if y is None:
                return self.nn_fit.kneighbors_graph(self.nn_fit._fit_X,
                                                    self.n_neighbors,
                                                    mode='connectivity')
            else:
                return self.nn_fit.kneighbors(y, return_distance=False)
        elif callable(self.kernel):
            if y is None:
                return self.kernel(X, X)
            else:
                return self.kernel(X, y)
        else:  # pragma: no cover
            raise ValueError(f"{self.kernel} is not a valid kernel. Only rbf and knn"
                             " or an explicit function "
                             " are supported at this time.")

    @abstractmethod
    def _build_graph(self):  # pragma: no cover
        raise NotImplementedError("Graph construction must be implemented"
                                  " to fit a label propagation model.")

    def predict(self, X, session=None, run_kwargs=None):
        """Performs inductive inference across the model.

        Parameters
        ----------
        X : array_like, shape = [n_samples, n_features]

        Returns
        -------
        y : array_like, shape = [n_samples]
            Predictions for input data
        """
        probas = self.predict_proba(X, session=session, run_kwargs=run_kwargs)
        result = mt.tensor(self.classes_)[mt.argmax(probas, axis=1)].ravel()
        result.execute(session=session, **(run_kwargs or dict()))
        return result

    def predict_proba(self, X, session=None, run_kwargs=None):
        """Predict probability for each possible outcome.

        Compute the probability estimates for each single sample in X
        and each possible outcome seen during training (categorical
        distribution).

        Parameters
        ----------
        X : array_like, shape = [n_samples, n_features]

        Returns
        -------
        probabilities : Tensor, shape = [n_samples, n_classes]
            Normalized probability distributions across
            class labels
        """

        check_is_fitted(self, 'X_')

        X_2d = check_array(X, accept_sparse=True)
        weight_matrices = self._get_kernel(self.X_, X_2d)
        if self.kernel == 'knn':
            probabilities = mt.array([
                mt.sum(self.label_distributions_[weight_matrix], axis=0)
                for weight_matrix in weight_matrices])
        else:
            weight_matrices = weight_matrices.T
            probabilities = mt.dot(weight_matrices, self.label_distributions_)
        normalizer = mt.atleast_2d(mt.sum(probabilities, axis=1)).T
        probabilities /= normalizer
        probabilities.execute(session=session, **(run_kwargs or dict()))
        return probabilities

    def fit(self, X, y, session=None, run_kwargs=None):
        """Fit a semi-supervised label propagation model based

        All the input data is provided matrix X (labeled and unlabeled)
        and corresponding label matrix y with a dedicated marker value for
        unlabeled samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            A {n_samples by n_samples} size matrix will be created from this

        y : array_like, shape = [n_samples]
            n_labeled_samples (unlabeled points are marked as -1)
            All unlabeled samples will be transductively assigned labels

        Returns
        -------
        self : returns an instance of self.
        """
        X, y = check_X_y(X, y)
        self.X_ = X
        to_run = [check_classification_targets(y)]

        # actual graph construction (implementations should override this)
        graph_matrix = self._build_graph()

        # label construction
        # construct a categorical distribution for classification only
        classes = mt.unique(y, aggregate_size=1).to_numpy(
            session=session, **(run_kwargs or dict()))
        classes = (classes[classes != -1])
        self.classes_ = classes

        n_samples, n_classes = len(y), len(classes)

        alpha = self.alpha
        # add check when we support LabelSpreading
        # if self._variant == 'spreading' and \
        #         (alpha is None or alpha <= 0.0 or alpha >= 1.0):
        #     raise ValueError('alpha=%s is invalid: it must be inside '
        #                      'the open interval (0, 1)' % alpha)
        y = mt.asarray(y)
        unlabeled = y == -1

        # initialize distributions
        self.label_distributions_ = mt.zeros((n_samples, n_classes))
        for label in classes:
            self.label_distributions_[y == label, classes == label] = 1

        y_static = mt.copy(self.label_distributions_)
        if self._variant == 'propagation':
            # LabelPropagation
            y_static[unlabeled] = 0
        else:  # pragma: no cover
            # LabelSpreading
            y_static *= 1 - alpha

        l_previous = mt.zeros((self.X_.shape[0], n_classes))

        unlabeled = unlabeled[:, mt.newaxis]

        for self.n_iter_ in range(self.max_iter):
            cond = mt.abs(self.label_distributions_ - l_previous).sum() < self.tol

            to_run.append(cond)
            ExecutableTuple(to_run).execute(
                session=session, **(run_kwargs or dict()))
            # clear
            to_run = []

            if cond.fetch(session=session):
                break

            l_previous = self.label_distributions_
            self.label_distributions_ = graph_matrix.dot(self.label_distributions_)

            if self._variant == 'propagation':
                normalizer = mt.sum(
                    self.label_distributions_, axis=1)[:, mt.newaxis]
                self.label_distributions_ /= normalizer
                self.label_distributions_ = mt.where(unlabeled,
                                                     self.label_distributions_,
                                                     y_static)
            else:  # pragma: no cover
                # clamp
                self.label_distributions_ = mt.multiply(
                    alpha, self.label_distributions_) + y_static

            to_run.append(self.label_distributions_)
        else:
            warnings.warn(
                f'max_iter={self.max_iter} was reached without convergence.',
                category=ConvergenceWarning
            )
            self.n_iter_ += 1

        normalizer = mt.sum(self.label_distributions_, axis=1)[:, mt.newaxis]
        self.label_distributions_ /= normalizer

        # set the transduction item
        transduction = mt.tensor(self.classes_)[mt.argmax(self.label_distributions_,
                                                          axis=1)]
        self.transduction_ = transduction.ravel()
        ExecutableTuple([self.label_distributions_, self.transduction_]).execute(
            session=session, **(run_kwargs or dict()))
        return self


class LabelPropagation(BaseLabelPropagation):
    """Label Propagation classifier

    Read more in the :ref:`User Guide <label_propagation>`.

    Parameters
    ----------
    kernel : {'knn', 'rbf', callable}
        String identifier for kernel function to use or the kernel function
        itself. Only 'rbf' and 'knn' strings are valid inputs. The function
        passed should take two inputs, each of shape [n_samples, n_features],
        and return a [n_samples, n_samples] shaped weight matrix.

    gamma : float
        Parameter for rbf kernel

    n_neighbors : integer > 0
        Parameter for knn kernel

    max_iter : integer
        Change maximum number of iterations allowed

    tol : float
        Convergence tolerance: threshold to consider the system at steady
        state

    Attributes
    ----------
    X_ : array, shape = [n_samples, n_features]
        Input array.

    classes_ : array, shape = [n_classes]
        The distinct labels used in classifying instances.

    label_distributions_ : array, shape = [n_samples, n_classes]
        Categorical distribution for each item.

    transduction_ : array, shape = [n_samples]
        Label assigned to each item via the transduction.

    n_iter_ : int
        Number of iterations run.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import datasets
    >>> from mars.learn.semi_supervised import LabelPropagation
    >>> label_prop_model = LabelPropagation()
    >>> iris = datasets.load_iris()
    >>> rng = np.random.RandomState(42)
    >>> random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
    >>> labels = np.copy(iris.target)
    >>> labels[random_unlabeled_points] = -1
    >>> label_prop_model.fit(iris.data, labels)
    LabelPropagation(...)

    References
    ----------
    Xiaojin Zhu and Zoubin Ghahramani. Learning from labeled and unlabeled data
    with label propagation. Technical Report CMU-CALD-02-107, Carnegie Mellon
    University, 2002 http://pages.cs.wisc.edu/~jerryzhu/pub/CMU-CALD-02-107.pdf

    See Also
    --------
    LabelSpreading : Alternate label propagation strategy more robust to noise
    """

    _variant = 'propagation'

    def __init__(self, kernel='rbf', gamma=20, n_neighbors=7,
                 max_iter=1000, tol=1e-3):
        super().__init__(kernel=kernel, gamma=gamma,
                         n_neighbors=n_neighbors, max_iter=max_iter,
                         tol=tol, alpha=None)

    def _build_graph(self):
        """Matrix representing a fully connected graph between each sample

        This basic implementation creates a non-stochastic affinity matrix, so
        class distributions will exceed 1 (normalization may be desired).
        """
        if self.kernel == 'knn':
            self.nn_fit = None
        affinity_matrix = self._get_kernel(self.X_)
        normalizer = affinity_matrix.sum(axis=0)
        affinity_matrix /= normalizer[:, mt.newaxis]
        return affinity_matrix

    def fit(self, X, y, session=None, run_kwargs=None):
        return super().fit(X, y, session=session, run_kwargs=run_kwargs)
