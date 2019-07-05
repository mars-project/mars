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

from .server import MarsRequestHandler, get_jinja_env, register_web_handler

_jinja_env = get_jinja_env()


class WorkerListHandler(MarsRequestHandler):
    def get(self):
        workers_meta = self.web_api.get_workers_meta()

        template = _jinja_env.get_template('worker_list.html')
        self.write(template.render(worker_metrics=workers_meta))


class WorkerHandler(MarsRequestHandler):
    def get(self, endpoint):
        workers_meta = self.web_api.get_workers_meta()

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

        template = _jinja_env.get_template('worker_detail.html')
        self.write(template.render(
            endpoint=endpoint,
            worker_metrics=workers_meta[endpoint],
            progresses=progresses,
        ))


register_web_handler('/worker', WorkerListHandler)
register_web_handler('/worker/(?P<endpoint>[^/]+)', WorkerHandler)
