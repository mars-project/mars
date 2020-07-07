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

import time
from collections import defaultdict, OrderedDict
from datetime import datetime

import numpy as np
from bokeh.embed import server_document
from bokeh.models import ColumnDataSource, Legend
from bokeh.plotting import figure

from ..utils import to_str
from .server import MarsRequestHandler, MarsWebAPI, get_jinja_env, \
    register_web_handler, register_bokeh_app

_jinja_env = get_jinja_env()
_worker_host_cache = dict()

TIMELINE_APP_NAME = 'bk_worker_timeline'
_TIMELINE_PALETTE = ['#0072B2', '#E69F00', '#F0E442', '#009E73', '#56B4E9',
                     '#D55E00', '#CC79A7', '#000000']

_dtype_datetime64_us = np.datetime64(datetime.now(), 'us').dtype


class WorkerListHandler(MarsRequestHandler):
    def get(self):
        workers_meta = self.web_api.get_workers_meta()
        for ep, meta in workers_meta.items():
            _worker_host_cache[ep] = meta['details']['host_name']

        template = _jinja_env.get_template('worker_pages/list.html')
        self.write(template.render(worker_metrics=workers_meta))


class WorkerHandler(MarsRequestHandler):
    def get(self, endpoint):
        workers_meta = self.web_api.get_workers_meta()
        for ep, meta in workers_meta.items():
            _worker_host_cache[ep] = meta['details']['host_name']

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

        template = _jinja_env.get_template('worker_pages/detail.html')
        self.write(template.render(
            endpoint=endpoint,
            worker_metrics=workers_meta[endpoint],
            progresses=progresses,
        ))


class EventUpdater(object):
    def __init__(self):
        self.owner_to_ticker = OrderedDict()
        self.unfinished_to_indexes = dict()
        self.base_indexes = defaultdict(lambda: 0)

        self.min_left = datetime.max
        self.max_right = datetime.min
        self.min_bottom = 0xffffffff
        self.max_top = 0

    def update_events(self, event_list, cur_time=None):
        import pandas as pd
        from ..worker.events import ProcedureEventType

        cur_time = cur_time or time.time()
        cur_time_dt = datetime.fromtimestamp(cur_time)

        unfinished_to_indexes = self.unfinished_to_indexes
        owner_to_ticker = self.owner_to_ticker
        base_indexes = self.base_indexes

        event_dfs = dict()
        patches = defaultdict(list)
        new_unfinished = dict()

        lefts = defaultdict(list)
        rights = defaultdict(list)
        tops = defaultdict(list)
        bottoms = defaultdict(list)

        for event in event_list:
            ev_type = event.event_type

            try:
                ev_type, idx = unfinished_to_indexes.pop(event.event_id)
                patches[ev_type].append((idx, datetime.fromtimestamp(event.time_end)))
                continue
            except KeyError:
                pass

            try:
                ticker = owner_to_ticker[event.owner]
            except KeyError:
                ticker = owner_to_ticker[event.owner] = 1 + len(owner_to_ticker)

            bottoms[ev_type].append(ticker - 0.5)
            tops[ev_type].append(ticker + 0.5)
            lefts[ev_type].append(datetime.fromtimestamp(event.time_start))

            if event.time_end is None:
                new_unfinished[event.event_id] = (
                    ev_type, base_indexes[ev_type] + len(rights[ev_type])
                )
                rights[ev_type].append(cur_time_dt)
            else:
                rights[ev_type].append(datetime.fromtimestamp(event.time_end))

        for ev_type in ProcedureEventType.__members__.values():
            df = event_dfs[ev_type] = pd.DataFrame(dict(
                left=np.array(lefts[ev_type], dtype=_dtype_datetime64_us),
                right=np.array(rights[ev_type], dtype=_dtype_datetime64_us),
                top=np.array(tops[ev_type], dtype=np.float32),
                bottom=np.array(bottoms[ev_type], dtype=np.float32),
            ), columns=['left', 'right', 'top', 'bottom'])
            base_indexes[ev_type] += len(df)
            self.min_left = min(self.min_left, df.left.min())
            self.max_right = max(self.max_right, df.right.max())
            self.min_bottom = min(self.min_bottom, df.bottom.min())
            self.max_top = max(self.max_top, df.top.max())

        for ev_type, idx in unfinished_to_indexes.values():
            patches[ev_type].append((idx, cur_time_dt))
            self.max_right = max(self.max_right, cur_time_dt)
        unfinished_to_indexes.update(new_unfinished)

        return event_dfs, patches

    @property
    def x_range(self):
        if self.min_left > self.max_right:
            return None
        return self.min_left, self.max_right

    @property
    def y_range(self):
        if self.min_bottom > self.max_top:
            return None
        return self.min_bottom, self.max_top


class WorkerTimelineHandler(MarsRequestHandler):
    def get(self, endpoint):
        template = _jinja_env.get_template('worker_pages/timeline.html')
        worker_timeline_script = server_document(
            '%s://%s/%s' % (self.request.protocol, self.request.host, TIMELINE_APP_NAME),
            arguments=dict(endpoint=endpoint))
        self.write(template.render(
            endpoint=endpoint,
            host_name=_worker_host_cache.get(endpoint, endpoint),
            worker_timeline_script=worker_timeline_script,
        ))

    @staticmethod
    def timeline_app(doc, scheduler_ip=None):
        from ..worker.events import EventCategory, ProcedureEventType

        worker_ep = to_str(doc.session_context.request.arguments.get('endpoint')[0])
        web_api = MarsWebAPI(scheduler_ip)

        last_query_time = time.time()
        events = web_api.query_worker_events(
            worker_ep, EventCategory.PROCEDURE, time_end=last_query_time)

        updater = EventUpdater()
        data_sources = dict()

        dfs, _ = updater.update_events(events)

        p = figure(x_axis_type='datetime', plot_height=500, plot_width=800)
        renderers = []
        for idx, ev_type in enumerate(ProcedureEventType.__members__.values()):
            ds = data_sources[ev_type] = ColumnDataSource(dfs[ev_type])
            renderers.append(p.quad(left='left', right='right', top='top', bottom='bottom', source=ds,
                                    color=_TIMELINE_PALETTE[idx]))

        p.axis.major_tick_in = 0
        p.ygrid.grid_line_color = None
        p.outline_line_color = None

        old_owners = set()

        def _update_axes():
            x_range, y_range = updater.x_range, updater.y_range
            if x_range is None or y_range is None:
                return

            p.title.text = 'Event time range: (%s - %s)' % tuple(t.strftime('%Y-%m-%d %H:%M:%S') for t in x_range)
            if set(updater.owner_to_ticker.keys()) - old_owners:
                p.yaxis.ticker = list(range(1, 1 + len(updater.owner_to_ticker)))
                p.yaxis.major_label_overrides = dict((v, k) for k, v in updater.owner_to_ticker.items())
                old_owners.update(updater.owner_to_ticker.keys())

        _update_axes()

        legend_items = [('\0' + s, [r]) for s, r in zip(ProcedureEventType.__members__.keys(), renderers)]
        legend = Legend(items=legend_items, location='bottom_right',
                        orientation='horizontal', label_text_font_size='10px')
        p.add_layout(legend, 'below')

        doc.add_root(p)

        def _refresher():
            nonlocal last_query_time
            query_time = time.time()
            events = web_api.query_worker_events(
                worker_ep, EventCategory.PROCEDURE, time_start=last_query_time, time_end=query_time)
            last_query_time = query_time

            dfs, patches = updater.update_events(events)
            for ev_type, df in dfs.items():
                if len(df):
                    data_sources[ev_type].stream(df)

            for ev_type, patch in patches.items():
                if patch:
                    data_sources[ev_type].patch(dict(right=patch))

            _update_axes()

        doc.add_periodic_callback(_refresher, 5000)


register_web_handler('/worker', WorkerListHandler)
register_web_handler('/worker/(?P<endpoint>[^/]+)', WorkerHandler)
register_web_handler('/worker/(?P<endpoint>[^/]+)/timeline', WorkerTimelineHandler)

register_bokeh_app('/' + TIMELINE_APP_NAME, WorkerTimelineHandler.timeline_app)
