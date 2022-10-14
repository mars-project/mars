# Copyright 1999-2022 Alibaba Group Holding Ltd.
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

import ray

original_remote_function_options = ray.remote_function.RemoteFunction.options


def _wrap_original_remote_function_options(*args, **kwargs):
    assert kwargs["num_cpus"] == 5, "expect num_cpus==5"
    return original_remote_function_options(*args, **kwargs)


ray.remote_function.RemoteFunction.options = _wrap_original_remote_function_options
