# Copyright 1999-2021 Alibaba Group Holding Ltd.
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
import warnings
import functools

from pkg_resources import iter_entry_points

logger = logging.getLogger(__name__)


# from https://github.com/numba/numba/blob/master/numba/core/entrypoints.py
# Must put this here to avoid extensions re-triggering initialization
@functools.lru_cache(maxsize=None)
def init_extension_entrypoints():
    """Execute all `mars_extensions` entry points with the name `init`
    If extensions have already been initialized, this function does nothing.
    """
    for entry_point in iter_entry_points("mars_extensions", "init"):
        logger.info("Loading extension: %s", entry_point)
        try:
            func = entry_point.load()
            func()
        except Exception as e:
            msg = "Mars extension module '{}' failed to load due to '{}({})'."
            warnings.warn(
                msg.format(entry_point.module_name, type(e).__name__, str(e)),
                stacklevel=2,
            )
            logger.info("Extension loading failed for: %s", entry_point)
