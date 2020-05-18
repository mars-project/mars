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
import os
import platform
import socket
import sys
import time

from . import resource
from .utils import git_info, lazy_import

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None
try:
    import scipy
except ImportError:  # pragma: no cover
    scipy = None
try:
    import pandas
except ImportError:  # pragma: no cover
    pandas = None

cp = lazy_import('cupy', globals=globals(), rename='cp')
cudf = lazy_import('cudf', globals=globals())

logger = logging.getLogger(__name__)

_is_initial = True


def gather_node_info():
    from .lib.mkl_interface import mkl_get_version
    from .lib.nvutils import NVError
    global _is_initial

    if _is_initial:
        _is_initial = False
        resource.cpu_percent()

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

    if 'MARS_K8S_POD_NAME' in os.environ:
        node_info['k8s_pod_name'] = os.environ['MARS_K8S_POD_NAME']
    if 'CONTAINER_ID' in os.environ:
        node_info['yarn_container_id'] = os.environ['CONTAINER_ID']

    try:
        cuda_info = resource.cuda_info()
    except NVError:
        logger.exception('NVError encountered, cannot gather CUDA devices.')
        cuda_info = None
    if cuda_info:
        node_info['cuda_info'] = 'Driver: %s\nCUDA: %s\nProducts: %s\n' % \
                                 (cuda_info.driver_version, cuda_info.cuda_version,
                                  ', '.join(cuda_info.products))

    package_lines = []
    if np is not None:
        ver_str = 'numpy==' + np.__version__
        if hasattr(np, '__mkl_version__') and mkl_get_version:
            mkl_version = mkl_get_version()
            ver_str += ' (mkl: %d.%d.%d)' % (mkl_version.major, mkl_version.minor, mkl_version.update)
        package_lines.append(ver_str)
    if scipy is not None:
        package_lines.append('scipy==%s' % scipy.__version__)
    if pandas is not None:
        package_lines.append('pandas==%s' % pandas.__version__)
    if cp is not None:
        package_lines.append('cupy==%s' % cp.__version__)
    if cudf is not None:
        package_lines.append('cudf==%s' % cudf.__version__)

    node_info['package_info'] = '\n'.join(package_lines)

    git = git_info()
    if git:
        node_info['git_info'] = '%s %s' % (git[0], git[1])
    else:
        node_info['git_info'] = 'Not available'

    return node_info
