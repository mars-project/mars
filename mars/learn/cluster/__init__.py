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

    def _install():
        from ._k_means_common import KMeansInertia, KMeansRelocateEmptyClusters
        from ._k_means_elkan_iter import KMeansElkanInitBounds, \
            KMeansElkanUpdate, KMeansElkanPostprocess
        from ._k_means_init import KMeansPlusPlusInit
        from ._k_means_lloyd_iter import KMeansLloydUpdate, KMeansLloydPostprocess

        del KMeansInertia, KMeansRelocateEmptyClusters, KMeansElkanInitBounds, \
            KMeansElkanUpdate, KMeansElkanPostprocess, KMeansPlusPlusInit, \
            KMeansLloydUpdate, KMeansLloydPostprocess

    _install()
    del _install
except ImportError:
    KMeans = None
    k_means = None
