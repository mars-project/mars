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

from .run_script import run_pytorch_script
from .dataset import MarsDataset
<<<<<<< HEAD
from .sampler import MarsSequentialSampler, MarsRandomSampler
=======
from .sampler import MarsDistributedSampler, MarsSequentialSampler, MarsRandomSampler
>>>>>>> 10bb3397127f0c91e7be73ea235820106c39dcec


def register_op():
    from .run_script import RunPyTorch
    del RunPyTorch
