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

import mars.remote as mr
from mars import new_session, execute


class EmptyRemotesExecutionSuite:
    """
    Benchmark that times running a number of empty subtasks
    """

    def setup(self):
        self.session = new_session(default=True)

    def teardown(self):
        self.session.stop_server()

    def time_remotes(self):
        def empty_fun(_i):
            pass

        remotes = [mr.spawn(empty_fun, args=(i,)) for i in range(1000)]
        execute(*remotes, session=self.session)
