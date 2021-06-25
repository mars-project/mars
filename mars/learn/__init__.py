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

# register operands
# import torch first, or some issue emerges,
# see https://github.com/pytorch/pytorch/issues/2575
from .contrib import pytorch, tensorflow, xgboost, lightgbm, statsmodels
from .metrics import pairwise
from . import cluster
from . import preprocessing
from . import proxima
from . import neighbors
from . import utils

for _mod in [xgboost, tensorflow, pytorch, lightgbm, proxima, neighbors, statsmodels]:
    _mod.register_op()

del _mod, pairwise, preprocessing, utils
