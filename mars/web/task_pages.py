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
from ..scheduler.utils import OperandState
from ..scheduler import GraphActor, SessionManagerActor, GraphMetaActor
from ..compat import six
from ..utils import to_str
from ..actors import new_client

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

_actor_client = new_client()
_jinja_env = get_jinja_env()

_base_palette = ['#e0f9ff', '#81d4dd', '#ff9e3b', '#399e34', '#2679b2', '#b3de8e',
                 '#e01f27', '#b3b3cc', '#f0f0f5']


def task_list(doc, cluster_info):
    uid = SessionManagerActor.default_name()
    sessions_ref = _actor_client.actor_ref(uid, address=cluster_info.get_scheduler(uid))

    sessions = dict()
    for session_id, session_ref in six.iteritems(sessions_ref.get_sessions()):
        sessions[session_id] = dict()
        session_desc = sessions[session_id]
        session_desc['id'] = session_id
        session_desc['name'] = session_id
        session_desc['tasks'] = dict()
        session_ref = _actor_client.actor_ref(session_ref)
        for graph_key, graph_ref in six.iteritems(session_ref.get_graph_refs()):
            task_desc = dict()

            graph_meta_uid = GraphMetaActor.gen_name(session_id, graph_key)
            scheduler_address = cluster_info.get_scheduler(graph_meta_uid)
            graph_meta_ref = _actor_client.actor_ref(graph_meta_uid, address=scheduler_address)
            state = graph_meta_ref.get_state()
            if state == 'PREPARING':
                task_desc['state'] = state.lower()
                session_desc['tasks'][graph_key] = task_desc
                continue

            graph_ref = _actor_client.actor_ref(graph_ref)
            task_desc['id'] = graph_key
            task_desc['state'] = graph_ref.get_state().value
            start_time, end_time, graph_size = graph_ref.get_graph_info()
            task_desc['start_time'] = start_time
            task_desc['end_time'] = end_time or 'N/A'
            task_desc['graph_size'] = graph_size or 'N/A'

            session_desc['tasks'][graph_key] = task_desc

    doc.title = 'Mars UI'
    doc.template_variables['sessions'] = sessions
    doc.template = _jinja_env.get_template('task_list.html')


def task_operand(doc, cluster_info, session_id, task_id):
    session_name = session_id
    states = list(OperandState.__members__.values())

    graph_uid = GraphActor.gen_name(session_id, task_id)
    scheduler_address = cluster_info.get_scheduler(graph_uid)
    graph_ref = _actor_client.actor_ref(graph_uid, address=scheduler_address)
    ops, stats, progress = graph_ref.calc_stats()
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


def _route(cluster_info, doc):
    session_id = doc.session_context.request.arguments.get('session_id')
    task_id = doc.session_context.request.arguments.get('task_id')
    if session_id and task_id:
        task_operand(doc, cluster_info, to_str(session_id[0]), to_str(task_id[0]))
    else:
        task_list(doc, cluster_info)


register_ui_handler('/task', _route)
