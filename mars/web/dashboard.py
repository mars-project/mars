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

_jinja_env = get_jinja_env()


def dashboard(scheduler_ip, doc):
    doc.title = 'Mars UI'

    web_api = MarsWebAPI(scheduler_ip)
    scheduler_infos = web_api.get_schedulers_info()
    worker_infos = web_api.get_workers_meta()

    scheduler_summary = {
        'count': len(scheduler_infos),
        'cpu_used': sum(si['cpu_used'] for si in scheduler_infos.values()),
        'cpu_total': sum(si['cpu_total'] for si in scheduler_infos.values()),
        'memory_used': sum(si['memory_used'] for si in scheduler_infos.values()),
        'memory_total': sum(si['memory_total'] for si in scheduler_infos.values()),
        'git_branches': set(si['git_info'] for si in scheduler_infos.values()),
    }
    worker_summary = {
        'count': len(worker_infos),
        'cpu_used': sum(wi['hardware']['cpu_used'] for wi in worker_infos.values()),
        'cpu_total': sum(wi['hardware']['cpu_total'] for wi in worker_infos.values()),
        'memory_used': sum(wi['hardware']['memory_used'] for wi in worker_infos.values()),
        'memory_total': sum(wi['hardware']['memory_total'] for wi in worker_infos.values()),
        'git_branches': set(wi['details']['git_info'] for wi in worker_infos.values()),
    }

    doc.template_variables['scheduler_summary'] = scheduler_summary
    doc.template_variables['worker_summary'] = worker_summary
    doc.template = _jinja_env.get_template('dashboard.html')


register_ui_handler('/', dashboard)
