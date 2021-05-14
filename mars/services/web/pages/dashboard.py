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

from ..core import MarsRequestHandler, get_jinja_env

_jinja_env = get_jinja_env()

_template_html = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <title>Mars UI</title>
    <meta name="viewport" content="minimum-scale=1, initial-scale=1, width=device-width" />
  </head>
  <body>
    <script async src="/static/bundle.js"></script>
    <div id="root"></div>
  </body>
</html>
""".strip()


class DashboardHandler(MarsRequestHandler):
    def get(self):
        self.write(_template_html)


handlers = {
    '/': DashboardHandler,
}
