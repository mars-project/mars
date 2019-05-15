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

import logging
import platform
import socket
import sys
import time

from . import resource
from .compat import six
from .utils import git_info

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None
try:
    import scipy
except ImportError:  # pragma: no cover
    scipy = None
try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

logger = logging.getLogger(__name__)


def gather_node_info():
    from .lib.mkl_interface import mkl_get_version
    mem_stats = resource.virtual_memory()

    node_info = {
        'command_line': ' '.join(sys.argv),
        'platform': platform.platform(),
        'host_name': socket.gethostname(),
        'sys_version': sys.version,
        'cpu_used': resource.cpu_percent() / 100.0,
        'cpu_total': resource.cpu_count(),
        'memory_used': mem_stats.used,
        'memory_total': mem_stats.total,
        'update_time': time.time(),
    }

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
