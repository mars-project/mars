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

from .average import average
from .corrcoef import corrcoef
from .cov import cov
from .digitize import digitize, TensorDigitize
from .ptp import ptp
from .histogram import histogram_bin_edges, TensorHistogramBinEdges, \
    histogram, TensorHistogram
from .quantile import quantile
from .percentile import percentile
from .median import median


def _install():
    from ..core import Tensor, TensorData

    for cls in (Tensor, TensorData):
        setattr(cls, 'ptp', ptp)


_install()
del _install
