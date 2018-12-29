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

from collections import defaultdict

from .server import register_ui_handler, get_jinja_env
from ..utils import to_str
from .server import MarsWebAPI

_jinja_env = get_jinja_env()


def worker_list(doc, workers_meta):
    doc.title = 'Mars UI'

    doc.template_variables['worker_metrics'] = workers_meta
    doc.template = _jinja_env.get_template('worker_list.html')


def worker_detail(doc, workers_meta, endpoint):
    doc.title = 'Mars UI'

    progress_data = workers_meta[endpoint]['progress']
    progresses = dict()
    for session_data in progress_data.values():
        for g in session_data.values():
            if g['stage'] not in progresses:
                progresses[g['stage']] = dict(operands=defaultdict(lambda: 0), total=0)
            progresses[g['stage']]['operands'][g['operands']] += 1
            progresses[g['stage']]['total'] += 1
    for k in progresses:
        operands = sorted(progresses[k]['operands'].items(), key=lambda p: p[0])
        progresses[k]['operands'] = ', '.join('%s (%d)' % (k, v) for k, v in operands)

    doc.template_variables['endpoint'] = endpoint
    doc.template_variables['worker_metrics'] = workers_meta[endpoint]
    doc.template_variables['progresses'] = progresses
    doc.template = _jinja_env.get_template('worker_detail.html')


def _route(scheduler_ip, doc):
    web_api = MarsWebAPI(scheduler_ip)
    workers_meta = web_api.get_workers_meta()

    endpoint = doc.session_context.request.arguments.get('endpoint')
    if not endpoint:
        return worker_list(doc, workers_meta)
    else:
        return worker_detail(doc, workers_meta, to_str(endpoint[0]))


register_ui_handler('/worker', _route)
