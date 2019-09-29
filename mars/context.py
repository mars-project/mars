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

import threading
from collections import namedtuple
import sys

from .compat import Enum


_context_factory = threading.local()


def get_context():
    return getattr(_context_factory, 'current', None)


class RunningMode(Enum):
    local = 0
    local_cluster = 1
    distributed = 2


class ContextBase(object):
    """
    Context will be used as a global object to detect the environment,
    and fetch meta, etc, mostly used in the server side, not for user.
    """

    @property
    def running_mode(self):
        """
        Get the running mode, could be local, local_cluster or distributed.
        """
        raise NotImplementedError

    def __enter__(self):
        _context_factory.current = self

    def __exit__(self, *_):
        _context_factory.current = None

    # ---------------
    # Meta relative
    # ---------------

    def get_chunk_metas(self, chunk_keys):
        """
        Get chunk metas according to the given chunk keys.

        :param chunk_keys: chunk keys
        :return: List of chunk metas
        """
        raise NotImplementedError

    # ---------------
    # Graph relative
    # ---------------

    def submit_chunk_graph(self, graph, result_keys):
        """
        Submit fine-grained graph to execute.

        :param graph: fine-grained graph to execute
        :param result_keys: result chunk keys
        :return: Future
        """
        raise NotImplementedError

    def submit_tileable_graph(self, graph, result_keys):
        """
        Submit coarse-grained graph to execute.

        :param graph: coarse-grained graph to execute
        :param result_keys: result tileable keys
        :return: Future
        """
        raise NotImplementedError


ChunkMeta = namedtuple('ChunkMeta', ['chunk_size', 'chunk_shape', 'workers'])


class LocalContext(ContextBase):

    def __init__(self, local_session):
        self._local_session = local_session

    @property
    def running_mode(self):
        return RunningMode.local

    def get_chunk_metas(self, chunk_keys):
        chunk_result = self._local_session.executor.chunk_result
        metas = []
        for chunk_key in chunk_keys:
            chunk_data = chunk_result[chunk_key]
            if hasattr(chunk_data, 'nbytes'):
                # ndarray
                size = chunk_data.nbytes
                shape = chunk_data.shape
            elif hasattr(chunk_data, 'memory_usage'):
                # DataFrame
                size = chunk_data.memory_usage(deep=True)
                shape = chunk_data.shape
            else:
                # other
                size = sys.getsizeof(chunk_data)
                shape = ()

            metas.append(ChunkMeta(chunk_size=size, chunk_shape=shape, workers=None))

        return metas
