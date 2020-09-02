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

from collections import OrderedDict

from bokeh.models import ColumnDataSource, Legend
from bokeh.plotting import figure

from .server import MarsWebAPI, MarsRequestHandler, get_jinja_env, \
    register_bokeh_app, register_web_handler
from ..actors import new_client, ActorNotExist
from ..scheduler import OperandState
from ..utils import to_str


_actor_client = new_client()
_jinja_env = get_jinja_env()

PROGRESS_APP_NAME = 'bk_task_progress'
_base_palette = ['#e0f9ff', '#81d4dd', '#ff9e3b', '#399e34', '#2679b2', '#b3de8e',
                 '#e01f27', '#b3b3cc', '#f0f0f5']


class TaskListHandler(MarsRequestHandler):
    def get(self):
        sessions = self.web_api.get_tasks_info()

        template = _jinja_env.get_template('task_pages/list.html')
        self.write_rendered(template, sessions=sessions)


class TaskHandler(MarsRequestHandler):
    def get(self, session_id, graph_key):
        session_name = session_id

        template = _jinja_env.get_template('task_pages/progress.html')
        task_progress_script = self.bokeh_server_document(
            PROGRESS_APP_NAME, arguments=dict(session_id=session_id, task_id=graph_key))
        self.write_rendered(
            template,
            session_id=session_id,
            session_name=session_name,
            task_id=graph_key,
            task_progress_script=task_progress_script,
        )

    @staticmethod
    def task_progress_app(doc, scheduler_ip=None):
        session_id = to_str(doc.session_context.request.arguments.get('session_id')[0])
        task_id = to_str(doc.session_context.request.arguments.get('task_id')[0])
        web_api = MarsWebAPI(scheduler_ip)

        states = list(OperandState.__members__.values())

        ops, stats, progress = web_api.get_task_detail(session_id, task_id)
        source = ColumnDataSource(data=stats)
        cols = list(stats)[1:]
        p = figure(y_range=ops, plot_height=500, plot_width=800, x_range=(0, 100),
                   title="Total Progress: %0.2f%%" % progress)
        renderers = p.hbar_stack(cols, y='ops', height=0.9, color=_base_palette[0:len(states)],
                                 source=source)

        legend_items = [('\0' + s, [r]) for s, r in zip(cols, renderers)]
        legend = Legend(items=legend_items, location='bottom_right',
                        orientation='horizontal', label_text_font_size='10px')

        p.add_layout(legend, 'below')

        p.ygrid.grid_line_color = None
        p.yaxis.major_tick_line_color = None
        p.yaxis.minor_tick_line_color = None
        p.axis.minor_tick_line_color = None
        p.outline_line_color = None

        doc.add_root(p)

        def _refresher():
            try:
                _, new_stats, new_progress = web_api.get_task_detail(session_id, task_id)
            except ActorNotExist:  # pragma: no cover
                doc.remove_periodic_callback(cb)
                p.title.text = "Graph not found, session may be stopped"
                return
            p.title.text = "Total Progress: %0.2f%%" % new_progress
            source.data = new_stats

            if new_progress >= 100.0:
                doc.remove_periodic_callback(cb)

        if progress < 100.0:
            cb = doc.add_periodic_callback(_refresher, 5000)


class TaskRunningNodesHandler(MarsRequestHandler):
    def get(self, session_id, graph_key):
        session_name = session_id

        running_info = self.web_api.get_operand_info(session_id, graph_key, OperandState.RUNNING)
        sorted_info = OrderedDict(
            sorted((item for item in running_info.items()), key=lambda it: (it[1].get('op_name'), it[0])))

        template = _jinja_env.get_template('task_pages/runnings.html')
        self.write_rendered(
            template,
            session_id=session_id,
            session_name=session_name,
            task_id=graph_key,
            running_info=sorted_info
        )


register_web_handler('/session', TaskListHandler)
register_web_handler('/session/(?P<session_id>[^/]+)/graph/(?P<graph_key>[^/]+)',
                     TaskHandler)
register_web_handler('/session/(?P<session_id>[^/]+)/graph/(?P<graph_key>[^/]+)/running_nodes',
                     TaskRunningNodesHandler)

register_bokeh_app('/' + PROGRESS_APP_NAME, TaskHandler.task_progress_app)
