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

import sys
import socket
import platform
import logging
import os

logger = logging.getLogger(__name__)

from . import resource
from .utils import git_info, readable_size
from .actors import FunctionActor
from .compat import six


try:
    import numpy as np
except ImportError:
    np = None
try:
    import scipy
except ImportError:
    scipy = None
try:
    import cupy as cp
except ImportError:
    cp = None

_collectors = dict()


def register_collector(collector):
    _collectors[id(collector)] = collector


def gather_node_info():
    from .lib.mkl_interface import mkl_get_version
    mem_stats = resource.virtual_memory()

    node_info = dict()
    node_info['command_line'] = ' '.join(sys.argv)
    node_info['platform'] = platform.platform()
    node_info['host_name'] = socket.gethostname()
    node_info['sys_version'] = sys.version
    node_info['cpu_info'] = 'Used: %f\nTotal: %d' % (resource.cpu_percent() / 100.0, resource.cpu_count())
    node_info['memory_info'] = 'Used: %s\nTotal: %s' % (readable_size(mem_stats.used),
                                                        readable_size(mem_stats.total))

    for collector in _collectors.values():
        node_info.update(collector())

    if np is None:
        node_info['numpy_info'] = 'Not installed'
    else:
        sio = six.StringIO()
        sio.write('Version: %s\n' % np.__version__)
        if hasattr(np, '__mkl_version__') and mkl_get_version:
            mkl_version = mkl_get_version()
            sio.write('MKL Version: %d.%d.%d\n' % (mkl_version.major, mkl_version.minor, mkl_version.update))
        node_info['numpy_info'] = sio.getvalue().strip()

    if scipy is None:
        node_info['scipy_info'] = 'Not installed'
    else:
        node_info['scipy_info'] = 'Version: %s' % scipy.__version__

    git = git_info()
    if git:
        node_info['git_info'] = '%s %s' % (git[0], git[1])
    else:
        node_info['git_info'] = 'Not available'

    return node_info


class NodeInfoActor(FunctionActor):
    def __init__(self):
        super(NodeInfoActor, self).__init__()
        self._node_info = None

    @classmethod
    def default_name(cls):
        return 's:' + cls.__name__

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())

        self.ref().gather_info()

    def gather_info(self):
        self._node_info = gather_node_info()
        self.ref().gather_info(_tell=True, _delay=1)

    def get_info(self):
        return self._node_info
