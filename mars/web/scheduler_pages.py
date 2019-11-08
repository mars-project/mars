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

from .server import MarsRequestHandler, register_web_handler, get_jinja_env

_jinja_env = get_jinja_env()


class SchedulerListHandler(MarsRequestHandler):
    def get(self):
        schedulers_info = self.web_api.get_schedulers_info()

        template = _jinja_env.get_template('scheduler_pages/list.html')
        self.write(template.render(scheduler_metrics=schedulers_info))


class SchedulerHandler(MarsRequestHandler):
    def get(self, endpoint):
        schedulers_info = self.web_api.get_schedulers_info()

        template = _jinja_env.get_template('scheduler_pages/detail.html')
        self.write(template.render(
            endpoint=endpoint,
            scheduler_metrics=schedulers_info[endpoint],
        ))


register_web_handler('/scheduler', SchedulerListHandler)
register_web_handler('/scheduler/(?P<endpoint>[^/]+)', SchedulerHandler)
