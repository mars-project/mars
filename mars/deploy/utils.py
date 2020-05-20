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

import time


def wait_services_ready(selectors, min_counts, count_fun, timeout=None):
    readies = [0] * len(selectors)
    start_time = time.time()
    while True:
        all_satisfy = True
        for idx, selector in enumerate(selectors):
            if readies[idx] < min_counts[idx]:
                all_satisfy = False
                readies[idx] = count_fun(selector)
                break
        if all_satisfy:
            break
        if timeout and timeout + start_time < time.time():
            raise TimeoutError('Wait kubernetes cluster start timeout')
        time.sleep(1)
