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

from .euclidean import euclidean_distances
from .haversine import haversine_distances
from .manhattan import manhattan_distances
from .cosine import cosine_distances, cosine_similarity
from .pairwise import pairwise_distances, PAIRWISE_DISTANCE_FUNCTIONS
from .pairwise_distances_topk import pairwise_distances_topk
from .rbf_kernel import rbf_kernel
