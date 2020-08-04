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

import logging
import math
import os
import time
from collections import OrderedDict
from enum import Enum

from ..actors import ActorNotExist
from ..cluster_info import ClusterInfoActor, HasClusterInfoActor
from ..config import options
from ..errors import WorkerProcessStopped
from ..promise import PromiseActor
from ..utils import build_exc_info

logger = logging.getLogger(__name__)


class ExecutionState(Enum):
    ALLOCATING = 'allocating'
    PREPARING_INPUTS = 'preparing_inputs'
    CALCULATING = 'calculating'
    STORING = 'storing'


class WorkerClusterInfoActor(ClusterInfoActor):
    @classmethod
    def default_uid(cls):
        return 'w:0:%s' % cls.__name__


class WorkerHasClusterInfoActor(HasClusterInfoActor):
    cluster_info_uid = WorkerClusterInfoActor.default_uid()


class WorkerActor(WorkerHasClusterInfoActor, PromiseActor):
    """
    Base class of all worker actors, providing necessary utils
    """
    def __init__(self):
        super().__init__()
        self._proc_id = None

    @classmethod
    def default_uid(cls):
        return 'w:0:{0}'.format(cls.__name__)

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())
        try:
            self.set_cluster_info_ref()
        except ActorNotExist:
            pass
        self._init_shared_store()
        self._proc_id = self.ctx.distributor.distribute(self.uid)

    def _init_shared_store(self):
        import pyarrow.plasma as plasma
        from .storage.sharedstore import PlasmaSharedStore, PlasmaKeyMapActor

        mapper_ref = self.ctx.actor_ref(uid=PlasmaKeyMapActor.default_uid())
        try:
            self._plasma_client = plasma.connect(options.worker.plasma_socket)
        except TypeError:  # pragma: no cover
            self._plasma_client = plasma.connect(options.worker.plasma_socket, '', 0)
        self._shared_store = PlasmaSharedStore(self._plasma_client, mapper_ref)

    @property
    def proc_id(self):
        return self._proc_id

    @property
    def shared_store(self):
        return self._shared_store

    @property
    def storage_client(self):
        if not getattr(self, '_storage_client', None):
            from .storage import StorageClient
            self._storage_client = StorageClient(self)
        return self._storage_client

    def get_meta_client(self):
        from ..scheduler.chunkmeta import ChunkMetaClient
        return ChunkMetaClient(self.ctx, self._cluster_info_ref, has_local_cache=False)

    def handle_actors_down(self, halt_refs):
        """
        Handle process down event
        :param halt_refs: actor refs in halt processes
        """
        handled_refs = self.reject_promise_refs(halt_refs, *build_exc_info(WorkerProcessStopped))
        logger.debug('Process halt detected. Affected promises %r rejected.',
                     [ref.uid for ref in handled_refs])

    def register_actors_down_handler(self):
        from .daemon import WorkerDaemonActor

        daemon_ref = self.ctx.actor_ref(WorkerDaemonActor.default_uid())
        if self.ctx.has_actor(daemon_ref):
            daemon_ref.register_actor_callback(self.ref(), self.handle_actors_down.__name__,
                                               _tell=True)


class ExpMeanHolder(object):
    """
    Collector of statistics of a given series. The value decays by _factor as time elapses.
    """
    def __init__(self, factor=0.8):
        self._factor = factor
        self._count = 0
        self._v_divided = 0
        self._v_divisor = 0
        self._v2_divided = 0

    def put(self, value):
        self._count += 1
        self._v_divided = self._v_divided * self._factor + value
        self._v_divisor = self._v_divisor * self._factor + 1
        self._v2_divided = self._v2_divided * self._factor + value ** 2

    def count(self):
        return self._count

    def mean(self):
        if self._count == 0:
            return 0
        return self._v_divided * 1.0 / self._v_divisor

    def var(self):
        if self._count == 0:
            return 0
        return self._v2_divided * 1.0 / self._v_divisor - self.mean() ** 2

    def std(self):
        return math.sqrt(self.var())


class ExpiringCache(dict):
    def __init__(self, *args, **kwargs):
        expire_time = kwargs.pop('_expire_time', options.worker.callback_preserve_time)
        super().__init__(*args, **kwargs)

        self._expire_time = expire_time
        self._insert_times = OrderedDict()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key in self._insert_times:
            self._insert_times[key] = time.time()
            self._insert_times.move_to_end(key)
            return

        clean_keys = []
        self._insert_times[key] = time.time()
        last_finish_time = time.time() - self._expire_time
        for k, t in self._insert_times.items():
            if t < last_finish_time:
                clean_keys.append(k)
            else:
                break
        for k in clean_keys:
            self.pop(k, None)

    def __delitem__(self, item):
        self._insert_times.pop(item, None)
        self.pop(item, None)


def concat_operand_keys(graph, sep=','):
    from ..operands import Fetch
    graph_op_dict = OrderedDict()
    for c in graph:
        if isinstance(c.op, Fetch):
            continue
        if c.op.stage is None:
            graph_op_dict[c.op.key] = type(c.op).__name__
        else:
            graph_op_dict[c.op.key] = '%s:%s' % (type(c.op).__name__, c.op.stage.name)
    keys = sep.join(graph_op_dict.keys())
    graph_ops = sep.join(graph_op_dict.values())
    return keys, graph_ops


def get_chunk_key(key):
    return key[0] if isinstance(key, tuple) else key


def build_quota_key(session_id, data_key, owner):
    owner = str(owner)
    if isinstance(data_key, tuple):
        return data_key + (session_id, owner)
    return data_key, session_id, owner


def change_quota_key_owner(quota_key, owner):
    owner = str(owner)
    return quota_key[:-1] + (owner,)


def parse_spill_dirs(dir_repr):
    """
    Parse paths from a:b to list while resolving asterisks in path
    """
    import glob

    def _validate_dir(path):
        try:
            if not os.path.isdir(path):
                os.makedirs(path)
            touch_file = os.path.join(path, '.touch')
            open(touch_file, 'wb').close()
            os.unlink(touch_file)
            return True
        except OSError:  # pragma: no cover
            logger.exception('Fail to access directory %s', path)
            return False

    if dir_repr is None:
        return []
    elif isinstance(dir_repr, list):
        return dir_repr

    final_dirs = []
    for pattern in dir_repr.split(os.path.pathsep):
        pattern = pattern.strip()
        if not pattern:
            continue
        sub_patterns = pattern.split(os.path.sep)
        pos = 0
        while pos < len(sub_patterns) and '*' not in sub_patterns[pos]:
            pos += 1
        if pos == len(sub_patterns):
            final_dirs.append(pattern)
            continue
        left_pattern = os.path.sep.join(sub_patterns[:pos + 1])
        for match in glob.glob(left_pattern):
            final_dirs.append(os.path.sep.join([match] + sub_patterns[pos + 1:]))
    return sorted(d for d in final_dirs if _validate_dir(d))
