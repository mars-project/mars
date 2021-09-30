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

try:
    from ._kmeans import KMeans, k_means
    from ._mini_batch_k_means import MiniBatchKMeans, mini_batch_k_means
    from ._agglomerative import AgglomerativeClustering

    def _install():
        from ._k_means_common import KMeansInertia, KMeansRelocateEmptyClusters
        from ._k_means_elkan_iter import KMeansElkanInitBounds, \
            KMeansElkanUpdate, KMeansElkanPostprocess
        from ._k_means_init import KMeansPlusPlusInit
        from ._k_means_lloyd_iter import KMeansLloydUpdate, KMeansLloydPostprocess
        from ._mini_batch_k_means_operand import MiniBatchReassignCluster, \
            MiniBatchUpdate
        from ._agglomerative_operand import FixConnectivity, BuildWardTree, \
            CutTree

        del KMeansInertia, KMeansRelocateEmptyClusters, KMeansElkanInitBounds, \
            KMeansElkanUpdate, KMeansElkanPostprocess, KMeansPlusPlusInit, \
            KMeansLloydUpdate, KMeansLloydPostprocess
        del MiniBatchUpdate, MiniBatchReassignCluster, FixConnectivity, \
            BuildWardTree, CutTree

    _install()
    del _install
except ImportError:
    KMeans = None
    k_means = None
    MiniBatchKMeans = None
    mini_batch_k_means = None
    AgglomerativeClustering = None
