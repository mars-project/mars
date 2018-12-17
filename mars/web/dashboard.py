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

from .server import register_ui_handler, get_jinja_env, MarsWebAPI


def dashboard(scheduler_ip, doc):
    doc.title = 'Mars UI'

    web_api = MarsWebAPI(scheduler_ip)
    infos = web_api.get_schedulers_info()
    doc.template_variables['infos'] = infos
    jinja_env = get_jinja_env()
    doc.template = jinja_env.get_template('dashboard.html')


register_ui_handler('/', dashboard)
