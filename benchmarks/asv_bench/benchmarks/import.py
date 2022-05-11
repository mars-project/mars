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

import subprocess
import sys

# make sure necessary pyc files generated
import mars.dataframe as md
import mars.tensor as mt
del md, mt


class ImportPackageSuite:
    """
    Benchmark that times performance of chunk graph builder
    """
    def time_import_mars(self):
        proc = subprocess.Popen([sys.executable, "-c", "import mars"])
        proc.wait(120)

    def time_import_mars_tensor(self):
        proc = subprocess.Popen([sys.executable, "-c", "import mars.tensor"])
        proc.wait(120)

    def time_import_mars_dataframe(self):
        proc = subprocess.Popen([sys.executable, "-c", "import mars.dataframe"])
        proc.wait(120)
