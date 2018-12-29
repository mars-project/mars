# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

from .server import register_ui_handler, get_jinja_env
from ..utils import to_str
from .server import MarsWebAPI

_jinja_env = get_jinja_env()


def scheduler_list(doc, schedulers_info):
    doc.title = 'Mars UI'

    doc.template_variables['scheduler_metrics'] = schedulers_info
    doc.template = _jinja_env.get_template('scheduler_list.html')


def scheduler_detail(doc, schedulers_info, endpoint):
    doc.title = 'Mars UI'

    doc.template_variables['endpoint'] = endpoint
    doc.template_variables['scheduler_metrics'] = schedulers_info[endpoint]
    doc.template = _jinja_env.get_template('scheduler_detail.html')


def _route(scheduler_ip, doc):
    web_api = MarsWebAPI(scheduler_ip)
    schedulers_info = web_api.get_schedulers_info()

    endpoint = doc.session_context.request.arguments.get('endpoint')
    if not endpoint:
        return scheduler_list(doc, schedulers_info)
    else:
        return scheduler_detail(doc, schedulers_info, to_str(endpoint[0]))


register_ui_handler('/scheduler', _route)
