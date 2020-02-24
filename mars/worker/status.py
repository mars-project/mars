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

import copy
import os
import time
import logging
from collections import defaultdict

from .utils import WorkerActor, ExpMeanHolder
from .. import resource
from ..config import options
from ..node_info import gather_node_info

logger = logging.getLogger(__name__)


class StatusReporterActor(WorkerActor):
    def __init__(self, endpoint, with_gpu=True):
        super().__init__()
        self._endpoint = endpoint
        self._with_gpu = with_gpu

        self._upload_status = False

        self._status_ref = None
        self._resource_ref = None

    def post_create(self):
        from ..scheduler import ResourceActor

        super().post_create()
        self._status_ref = self.ctx.actor_ref(StatusActor.default_uid())
        self._resource_ref = self.get_actor_ref(ResourceActor.default_uid())
        self.ref().collect_status(_tell=True)

    def enable_status_upload(self):
        self._upload_status = True

    def collect_status(self):
        """
        Collect worker status and write to kvstore
        """
        meta_dict = dict()
        try:
            if not self._upload_status:
                return

            cpu_percent = resource.cpu_percent()
            disk_io = resource.disk_io_usage()
            net_io = resource.net_io_usage()
            if cpu_percent is None or disk_io is None or net_io is None:
                return
            hw_metrics = dict()
            hw_metrics['cpu'] = max(0.0, resource.cpu_count() - cpu_percent / 100.0)
            hw_metrics['cpu_used'] = cpu_percent / 100.0
            hw_metrics['cpu_total'] = resource.cpu_count()

            cuda_info = resource.cuda_info() if self._with_gpu else None
            if cuda_info:
                hw_metrics['cuda'] = cuda_info.gpu_count
                hw_metrics['cuda_total'] = cuda_info.gpu_count

            hw_metrics['disk_read'] = disk_io[0]
            hw_metrics['disk_write'] = disk_io[1]
            hw_metrics['net_receive'] = net_io[0]
            hw_metrics['net_send'] = net_io[1]

            iowait = resource.iowait()
            if iowait is not None:
                hw_metrics['iowait'] = iowait

            mem_stats = resource.virtual_memory()
            hw_metrics['memory'] = int(mem_stats.available)

            hw_metrics['memory_used'] = int(mem_stats.used)
            hw_metrics['memory_total'] = int(mem_stats.total)

            cache_allocations = self._status_ref.get_cache_allocations()
            cache_total = cache_allocations.get('total', 0)
            hw_metrics['cached_total'] = int(cache_total)
            hw_metrics['cached_hold'] = int(cache_allocations.get('hold', 0))

            mem_quota_allocations = self._status_ref.get_mem_quota_allocations()
            mem_quota_total = mem_quota_allocations.get('total', 0)
            mem_quota_allocated = mem_quota_allocations.get('allocated', 0)
            hw_metrics['mem_quota'] = int(mem_quota_total - mem_quota_allocated)
            hw_metrics['mem_quota_used'] = int(mem_quota_allocated)
            hw_metrics['mem_quota_total'] = int(mem_quota_total)
            hw_metrics['mem_quota_hold'] = int(mem_quota_allocations.get('hold', 0))

            if options.worker.spill_directory:
                if isinstance(options.worker.spill_directory, str):
                    spill_dirs = options.worker.spill_directory.split(':')
                else:
                    spill_dirs = options.worker.spill_directory
                if spill_dirs and 'disk_stats' not in hw_metrics:
                    hw_metrics['disk_stats'] = dict()
                disk_stats = hw_metrics['disk_stats']

                agg_disk_used = 0.0
                agg_disk_total = 0.0
                for spill_dir in spill_dirs:
                    if not os.path.exists(spill_dir):
                        continue
                    if spill_dir not in disk_stats:
                        disk_stats[spill_dir] = dict()

                    disk_usage = resource.disk_usage(spill_dir)
                    disk_stats[spill_dir]['disk_total'] = disk_usage.total
                    agg_disk_total += disk_usage.total
                    disk_stats[spill_dir]['disk_used'] = disk_usage.used
                    agg_disk_used += disk_usage.used
                hw_metrics['disk_used'] = agg_disk_used
                hw_metrics['disk_total'] = agg_disk_total

            cuda_card_stats = resource.cuda_card_stats() if self._with_gpu else None
            if cuda_card_stats:
                hw_metrics['cuda_stats'] = [dict(
                    product_name=stat.product_name,
                    gpu_usage=stat.gpu_usage,
                    temperature=stat.temperature,
                    fb_memory_total=stat.fb_mem_info.total,
                    fb_memory_used=stat.fb_mem_info.used,
                ) for stat in cuda_card_stats]

            meta_dict = dict()
            meta_dict['hardware'] = hw_metrics
            meta_dict['update_time'] = time.time()
            meta_dict['stats'] = dict()
            meta_dict['slots'] = dict()

            status_data = self._status_ref.get_stats()
            for k, v in status_data.items():
                meta_dict['stats'][k] = v

            slots_data = self._status_ref.get_slots()
            for k, v in slots_data.items():
                meta_dict['slots'][k] = v

            meta_dict['progress'] = self._status_ref.get_progress()
            meta_dict['details'] = gather_node_info()

            if options.vineyard.socket:  # pragma: no cover
                import vineyard
                client = vineyard.connect(options.vineyard.socket)
                meta_dict['vineyard'] = {'instance_id': client.instance_id}

            self._resource_ref.set_worker_meta(self._endpoint, meta_dict)
        except Exception as ex:
            logger.error('Failed to save status: %s. repr(meta_dict)=%r', str(ex), meta_dict)
        finally:
            self.ref().collect_status(_tell=True, _delay=1)


class StatusActor(WorkerActor):
    def __init__(self, endpoint, with_gpu=True):
        super().__init__()
        self._speed_holders = defaultdict(ExpMeanHolder)
        self._endpoint = endpoint
        self._with_gpu = with_gpu
        self._reporter_ref = None
        self._stats = dict()
        self._slots = dict()
        self._progress = dict()

        self._mem_quota_allocations = {}
        self._cache_allocations = {}

    def post_create(self):
        super().post_create()
        self._reporter_ref = self.ctx.create_actor(
            StatusReporterActor, self._endpoint, with_gpu=self._with_gpu,
            uid=StatusReporterActor.default_uid())

    def pre_destroy(self):
        self.ctx.destroy_actor(self._reporter_ref)

    def enable_status_upload(self):
        self._reporter_ref.enable_status_upload(_tell=True)

    def get_stats(self, items=None):
        if not items:
            return self._stats
        else:
            return dict((k, self._stats[k]) for k in items if k in self._stats)

    def update_stats(self, update_dict):
        self._stats.update(update_dict)

    def get_slots(self):
        return copy.deepcopy(self._slots)

    def update_slots(self, slots_dict):
        self._slots.update(slots_dict)

    def get_progress(self):
        return copy.deepcopy(self._progress)

    def update_progress(self, session_id, graph_key, op_name, state):
        """
        Update statuses of operands
        :param session_id: session id
        :param graph_key: graph key
        :param op_name: operand name
        :param state: operand execution state
        """
        session_id = str(session_id)
        graph_key = str(graph_key)

        try:
            session_dict = self._progress[session_id]
        except KeyError:
            session_dict = self._progress[session_id] = dict()
        try:
            graph_dict = session_dict[graph_key]
        except KeyError:
            graph_dict = session_dict[graph_key] = dict()
        graph_dict.update(dict(operands=op_name, stage=state.name))

    def remove_progress(self, session_id, graph_key):
        try:
            del self._progress[session_id][graph_key]
            if not self._progress[session_id]:
                del self._progress[session_id]
        except KeyError:
            pass

    def update_mean_stats(self, item, value):
        """
        Update sequence statistics. Moment statistics will be computed automatically
        :param item: statistics item
        :param value: statistics value
        """
        stats_item = self._speed_holders[item]
        stats_item.put(value)
        self._stats[item] = dict(count=stats_item.count(), mean=stats_item.mean(),
                                 std=stats_item.std())

    def set_cache_allocations(self, value):
        self._cache_allocations = value

    def get_cache_allocations(self):
        return self._cache_allocations

    def set_mem_quota_allocations(self, value):
        self._mem_quota_allocations = value

    def get_mem_quota_allocations(self):
        return self._mem_quota_allocations
