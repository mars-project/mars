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

import os
import signal
from collections import defaultdict

import psutil

from .utils import WorkerActor


class WorkerDaemonActor(WorkerActor):
    """
    Actor handling termination of worker processes. If a process terminates
    unprecedentedly, this actor restarts actors previously started
    """
    def __init__(self):
        super(WorkerDaemonActor, self).__init__()
        self._proc_actors = defaultdict(dict)
        self._proc_pids = dict()
        self._killed_pids = set()
        self._callbacks = []

    def create_actor(self, *args, **kwargs):
        """
        Create an actor and record its creation args for recovery
        """
        ref = self.ctx.create_actor(*args, **kwargs)
        proc_idx = self.ctx.distributor.distribute(ref.uid)
        ref_key = (ref.uid, ref.address)
        self._proc_actors[proc_idx][ref_key] = (ref_key, args, kwargs, False)
        return ref

    def register_child_actor(self, actor_ref):
        proc_idx = self.ctx.distributor.distribute(actor_ref.uid)
        ref_key = (actor_ref.uid, actor_ref.address)
        self._proc_actors[proc_idx][ref_key] = (ref_key, None, None, True)

    def register_process(self, actor_ref, pid):
        """
        Register a process id
        """
        proc_idx = self.ctx.distributor.distribute(actor_ref.uid)
        self._proc_pids[proc_idx] = pid

    def register_callback(self, actor_ref, func):
        """
        Register a callback on an actor for handling process down
        :param actor_ref: ActorRef to handle the callback
        :param func: function name of the callback
        """
        self._callbacks.append((actor_ref.uid, actor_ref.address, func))

    def kill_actor_process(self, actor_ref):
        """
        Kill a process given the ref of an actor on it
        """
        proc_idx = self.ctx.distributor.distribute(actor_ref.uid)
        try:
            pid = self._proc_pids[proc_idx]
            self._killed_pids.add(pid)
            os.kill(pid, signal.SIGKILL)
        except (KeyError, OSError):
            pass

    def is_actor_process_alive(self, actor_ref):
        """
        Check if the process holding the actor is still alive
        """
        proc_idx = self.ctx.distributor.distribute(actor_ref.uid)
        pid = self._proc_pids[proc_idx]
        if pid in self._killed_pids:
            return False
        try:
            psutil.Process(pid)
        except psutil.NoSuchProcess:
            return False
        else:
            return True

    def handle_process_down(self, proc_indices):
        """
        When process down is detected,
        :param proc_indices: indices of processes in Mars Worker
        """
        # recreate actors given previous records
        refs = set()
        for proc_idx in proc_indices:
            for actor_args in self._proc_actors[proc_idx].values():
                ref_key, args, kw, is_child = actor_args
                refs.add(ref_key)
                if not is_child:
                    self.ctx.create_actor(*args, **kw)

        # invoke registered callbacks
        for cb in self._callbacks:
            uid, addr, func = cb
            ref = self.ctx.actor_ref(uid, address=addr)
            clean_refs = [self.ctx.actor_ref(u, address=None) for u, _ in refs] \
                + [self.ctx.actor_ref(u, address=a) for u, a in refs]
            getattr(ref, func)(clean_refs, _tell=True)
