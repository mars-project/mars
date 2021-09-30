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
from sklearn.base import ClusterMixin
try:
    from sklearn.cluster import _hierarchical_fast as _hierarchical
except:
    raise ImportError

from ... import tensor as mt
from ..base import BaseEstimator
from ..utils.validation import check_memory
from ..utils.validation import _num_samples
from ._agglomerative_operand import ward_tree, CutTree


_TREE_BUILDERS = dict(
    ward=ward_tree,
    # complete=_complete_linkage,
    # average=_average_linkage,
    # single=_single_linkage,
)


class AgglomerativeClustering(ClusterMixin, BaseEstimator):
    """Agglomerative Clustering.
    
    Read more in the :ref:`User Guide <agglomerative clustering>`.

    Parameters
    ----------

    n_clusters : int or None, default=2
        The number of clusters to find. It must be ``None`` if
        ``distance_threshold`` is not ``None``.

    affinity : str or callable, default='euclidean'
        Metric used to compute the linkage. Can be "euclidean", "l1", "l2",
        "manhattan", "cosine", or "precomputed".
        If linkage is "ward", only "euclidean" is accepted.
        If "precomputed", a distance matrix (instead of a similarity matrix)
        is needed as input for the fit method.

    connectivity : array-like or callable, default=None
        Connectivity matrix. Defines for each sample the neighboring
        samples following a given structure of the data.

    compute_full_tree : 'auto' or bool, default='auto'
        Stop early the construction of the tree at ``n_clusters``. This is
        useful to decrease computation time if the number of clusters is not
        small compared to the number of samples. This option is useful only
        when specifying a connectivity matrix.

    linkage : 'ward'
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.

    distance_threshold : float, default=None
        The linkage distance threshold above which, clusters will not be
        merged. If not ``None``, ``n_clusters`` must be ``None`` and
        ``compute_full_tree`` must be ``True``.

    compute_distances : bool, default=False
        Computes distances between clusters even if `distance_threshold` is not
        used. This can be used to make dendrogram visualization, but introduces
        a computational and memory overhead.

    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm. If ``distance_threshold=None``, 
        it will be equal to the given ``n_clusters``.

    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point.

    n_leaves_ : int
        Number of leaves in the hierarchical tree.

    n_connected_components_ : int
        The estimated number of connected components in the graph.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    children_ : array-like of shape (n_samples-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.

    distances_ : array-like of shape (n_nodes-1,)
        Distances between nodes in the corresponding place in `children_`.
        Only computed if `distance_threshold` is used or `compute_distances`
        is set to `True`.

    Examples
    --------

    >>> from mars.learn.cluster import AgglomerativeClustering
    >>> import mars.tensor as mt
    >>> X = mt.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> clustering = AgglomerativeClustering().fit(X)
    >>> clustering
    AgglomerativeClustering()
    >>> clustering.labels_
    array([1, 1, 1, 0, 0, 0])
    """

    def __init__(self, n_clusters=2, *, affinity="euclidean", connectivity=None,
                 compute_full_tree="auto", linkage="ward", distance_threshold=None,
                compute_distances=False,):

        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.connectivity = connectivity
        self.compute_full_tree = compute_full_tree
        self.linkage = linkage
        self.affinity = affinity
        self.compute_distances = compute_distances

    def _check_params(self):

        if self.n_clusters is not None and self.n_clusters <= 0:
            raise ValueError(f"n_clusters should be an integer greater than 0."
                             f" {self.n_clusters} was provided.")

        if not ((self.n_clusters is None) ^ (self.distance_threshold is None)):
            raise ValueError("Exactly one of n_clusters and "
                             "distance_threshold has to be set, and the other "
                             "needs to be None.")

        if (self.distance_threshold is not None
                and not self.compute_full_tree):
            raise ValueError("compute_full_tree must be True if "
                             "distance_threshold is set.")

        if self.linkage == "ward" and self.affinity != "euclidean":
            raise ValueError(f"{self.affinity} was provided as affinity. "
                             f"Ward can only work with euclidean distances.")

        if self.linkage not in _TREE_BUILDERS:
            raise ValueError(f"Unknown linkage type {self.linkage}. "
                             f"Valid options are {_TREE_BUILDERS.keys()}")


    def fit(self, X, y=None, session=None, run_kwargs=None):
        """Compute agglomerative clustering.

        Parameters
        ----------
        X : X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
            Fitted estimator.
        """
        X = self._validate_data(X, accept_sparse=False,
                                dtype=[np.float64, np.float32],
                                order='C', copy=False,
                                accept_large_sparse=False)

        if np.isnan(_num_samples(X)):  # pragma: no cover
            X.execute(session=session, **(run_kwargs or dict()))
        # memory = check_memory(self.memory)

        self._check_params()


        # print("\033[32m mars fit memory\033[0m", memory, X) ==

        tree_builder = _TREE_BUILDERS[self.linkage]
        
        connectivity = self.connectivity
        if self.connectivity is not None:
            if callable(self.connectivity):
                connectivity = self.connectivity(X)
            connectivity = self._validate_data(connectivity, accept_sparse=True,
                                dtype=[np.float64, np.float32],
                                order='C', copy=False,
                                accept_large_sparse=True)
            connectivity.execute(session=session, **(run_kwargs or dict()))
            # print("\033[32m mars fit 接受connectivity\033[0m", type(connectivity), connectivity.shape) ==


        n_samples = len(X)
        # print("\033[32m mars fit n_samples\033[0m", n_samples) ==

        compute_full_tree = self.compute_full_tree
        if self.connectivity is None:
            compute_full_tree = True
        if compute_full_tree == 'auto':
            if self.distance_threshold is not None:
                compute_full_tree = True
            else:
                # Early stopping is likely to give a speed up only for
                # a large number of clusters. The actual threshold
                # implemented here is heuristic
                compute_full_tree = self.n_clusters < max(100, .02 * n_samples)
        n_clusters = self.n_clusters
        if compute_full_tree:
            n_clusters = None

        # print("\033[32m mars fit n_clusters\033[0m", n_clusters) ==

        # Construct the tree
        kwargs = {}
        if self.linkage != 'ward':
            kwargs['linkage'] = self.linkage
            kwargs['affinity'] = self.affinity

        distance_threshold = self.distance_threshold

        return_distance = (
            (distance_threshold is not None) or self.compute_distances
        )

        # print("\033[32m mars fit tree_builder参数 X, connectivity, n_clusters, return_distance\n\033[0m", X, connectivity, n_clusters, return_distance) ==
        # print("\033[32m mars fit tree_builder参数 type connectivity\n\033[0m", type(connectivity)) ==

        # ============== 开始构建树 =================================

        # out = memory.cache(tree_builder)==
        # print("\033[32m mars fit memory.cache的返回\033[0m", out)==


        # out = memory.cache(tree_builder)(X, connectivity=connectivity, ==
        #                                  n_clusters=n_clusters,
        #                                  return_distance=return_distance,
        #                                  **kwargs)
        out = tree_builder(X, connectivity=connectivity, n_clusters=n_clusters,
                           return_distance=return_distance, 
                           session=session, run_kwargs=run_kwargs)
        (self.children_, self.n_connected_components_,
                self.n_leaves_, parents) = out[:4]

        if return_distance:
            self.distances_ = out[-1]
            # print("self.distances_", self.distances_)==
        # print("\033[32m mars fit 构建树结束\033[0m", out)==

        # ============= 构建树结束 ===================================

        # print(type(self.children_))==
        # print(max(self.children_))==


        if self.distance_threshold is not None:  # distance_threshold is used
            self.n_clusters_ = mt.count_nonzero(                            # TODO(mimeku): 这个部分没有测试
                self.distances_ >= distance_threshold) + 1
            self.n_clusters_.execute()
            self.n_clusters_ = self.n_clusters_.fetch()
            # print("\033[32m mars fit mt.count_\033[0m", self.n_clusters_)==
        else:  # n_clusters is used
            self.n_clusters_ = self.n_clusters

        # print("\033[32m mars cut前\033")==

        # Cut the tree
        if compute_full_tree:
            # self.labels_ = _hc_cut(self.n_clusters_, self.children_,
            #                        self.n_leaves_)
            # print("\033[32m mars cut tree\033[0m")==
            # print(type(self.n_clusters_), type(self.n_leaves_), type(n_samples), type(self.children_))==
            # print(self.children_)


            # print(max(self.children_[-1])) ==

            cut_op = CutTree(children=self.children_, n_clusters=self.n_clusters_, n_leaves=self.n_leaves_, n_samples=n_samples)
            ret = cut_op()
            # print("\033[32m mars fit cuttree\033[0m", ret) ==
            label = ret
            
            label.execute()
            # print(label) ==
            self.labels_ = label

        else:

            # print("\033[32m mars cut else\033") ==

            labels = _hierarchical.hc_get_heads(parents.to_numpy(), copy=False)
            # copy to avoid holding a reference on the original array
            labels = mt.copy(labels[:n_samples])
            # Reassign cluster numbers
            self.labels_ = mt.searchsorted(mt.unique(labels), labels)
            self.labels_.execute()

        
        return self


    def fit_predict(self, X, y=None):
        """Fit and return the result of each sample's clustering assignment.
        In addition to fitting, this method also return the result of the
        clustering assignment for each sample in the training set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or \
                (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``affinity='precomputed'``.
        y : Ignored
            Not used, present here for API consistency by convention.
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        return super().fit_predict(X, y)


