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

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

from .server import register_ui_handler, get_jinja_env
from ..scheduler.utils import OperandState
from ..utils import to_str
from ..actors import new_client
from .server import MarsWebAPI


_actor_client = new_client()
_jinja_env = get_jinja_env()

_base_palette = ['#e0f9ff', '#81d4dd', '#ff9e3b', '#399e34', '#2679b2', '#b3de8e',
                 '#e01f27', '#b3b3cc', '#f0f0f5']


def task_list(doc, web_api):
    sessions = web_api.get_tasks_info()

    doc.title = 'Mars UI'
    doc.template_variables['sessions'] = sessions
    doc.template = _jinja_env.get_template('task_list.html')


def task_operand(doc, cluster_info, session_id, task_id):
    session_name = session_id
    states = list(OperandState.__members__.values())

    ops, stats, progress = cluster_info.get_task_detail(session_id, task_id)
    source = ColumnDataSource(stats)
    cols = list(stats)[1:]
    p = figure(y_range=[''] + ops, plot_height=500, plot_width=800, x_range=(0, 100),
               title="Task Progresses")
    p.hbar_stack(cols, y='ops', height=0.9, color=_base_palette[0:len(states)],
                 source=source, legend=['\0' + s for s in cols])

    p.legend.background_fill_alpha = 0.5
    p.legend.location = 'bottom_left'
    p.legend.orientation = "horizontal"
    p.legend.label_text_font_size = '10px'

    p.y_range.range_padding = 0.1
    p.ygrid.grid_line_color = None
    p.yaxis.major_tick_line_color = None
    p.yaxis.minor_tick_line_color = None
    p.axis.minor_tick_line_color = None
    p.outline_line_color = None

    doc.add_root(p)

    doc.title = 'Mars UI'
    doc.template_variables['session_id'] = session_id
    doc.template_variables['session_name'] = session_name
    doc.template_variables['task_id'] = task_id
    doc.template_variables['progress'] = progress
    doc.template = _jinja_env.get_template('task_operands.html')


def _route(scheduler_ip, doc):
    web_api = MarsWebAPI(scheduler_ip)
    session_id = doc.session_context.request.arguments.get('session_id')
    task_id = doc.session_context.request.arguments.get('task_id')
    if session_id and task_id:
        task_operand(doc, web_api, to_str(session_id[0]), to_str(task_id[0]))
    else:
        task_list(doc, web_api)


register_ui_handler('/task', _route)
