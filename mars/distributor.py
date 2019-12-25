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

import functools
import logging

from .actors import Distributor
from .lib.mmh3 import hash as mmh_hash
from .utils import to_binary, to_str

logger = logging.getLogger(__name__)


class MarsDistributor(Distributor):
    """
    We use the following format for uids of actors:

        <prefix>:<location_spec>:<name>

    ``prefix`` denote the owner of the actor, currently ``s``
    stands for schedulers and ``w`` stands for workers.
    ``location-spec`` stands for the process an actor should
    be created in. When the spec is an integer, the index of the
    process is ``location-spec % n_process``. When the spec
    is ``h<offset>``, distributor will first compute hash for
    the whole uid (including prefixes) and the index of the
    process is ``offset + hash % (n_process - offset)``.
    """
    def __init__(self, n_process, default_prefix=None):
        super().__init__(n_process)
        self._default_prefix = to_str(default_prefix)

    @functools.lru_cache(100)
    def distribute(self, uid):
        if not isinstance(uid, str) or self.n_process == 1:
            return 0
        uid = to_str(uid)
        id_parts = uid.split(':')
        try:
            if len(id_parts) == 3:
                if id_parts[1].startswith('h'):
                    offset = int(id_parts[1][1:] or '0')
                    # get process id by hashing uid
                    allocate_id = mmh_hash(to_binary(uid)) % (self.n_process - offset) + offset
                    return allocate_id
                else:
                    # to tell distributor the fixed process id
                    return (int(id_parts[1]) + self.n_process) % self.n_process
        except ValueError:
            pass

        if self._default_prefix is not None:
            return self.distribute(self._default_prefix + repr(uid).replace(':', '__'))
        else:
            raise ValueError('Malformed actor uid: %s' % uid)

    def make_same_process(self, uid, uid_rel, delta=0):
        rel_proc_id = self.distribute(uid_rel)
        id_parts = uid.split(':')
        if len(id_parts) == 3:
            id_parts[1] = str((rel_proc_id + delta + self.n_process) % self.n_process)
            return ':'.join(id_parts)
        elif self._default_prefix is not None:
            return self.make_same_process(self._default_prefix + repr(uid).replace(':', '__'),
                                          uid_rel, delta)
        else:
            raise ValueError('Malformed actor uid: %s' % uid)
