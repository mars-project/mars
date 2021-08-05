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

import os

from .core import MarsRequestHandler


class IndexHandler(MarsRequestHandler):
    def _get_index_page(self):
        try:
            return self._index_page
        except AttributeError:
            index_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'index.html'
            )
            with open(index_file, 'r') as file_obj:
                self._index_page = file_obj.read()
            return self._index_page

    def get(self):
        self.write(self._get_index_page())


handlers = {
    '/': IndexHandler,
}
