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

"""
This utility module dumps code of remote traceback and loads them
into local linecache. This enables displaying codes of remote
tracebacks correctly.
"""

import linecache
import os
import types
from collections import defaultdict


def dump_traceback_code(tb: types.TracebackType, number_of_lines_of_context: int = 5):
    """
    Dump codes before and after lines of tracebacks.

    Parameters
    ----------
    tb: types.TracebackType
        Traceback object
    number_of_lines_of_context: int
        Total number of lines around the code
    Returns
    -------
    result: dict
        Dumped code lines of traceback
    """
    results = defaultdict(lambda: dict(fragments=[]))

    while tb:
        file_name = tb.tb_frame.f_code.co_filename
        if linecache.getline(file_name, tb.tb_lineno):  # pragma: no branch
            code_lines = linecache.cache[file_name][2]
            left_range = max(tb.tb_lineno - number_of_lines_of_context // 2 - 1, 0)
            right_range = min(left_range + number_of_lines_of_context, len(code_lines))

            cache_data = linecache.cache[file_name]
            fragment = cache_data[2][left_range:right_range]
            results[file_name]["fragments"].append(
                dict(left=left_range, right=right_range, code=fragment)
            )
            results[file_name].update(
                dict(size=cache_data[0], lines=len(cache_data[2]))
            )
        tb = tb.tb_next
    return dict(results)


def load_traceback_code(code_frags: dict, cache: dict = None):
    """
    Load dumped codes for remote tracebacks.

    Parameters
    ----------
    code_frags: dict
        Dumped codes for remote traceback.
    cache: dict
        Target for codes to be dumped, for test purpose only.
        Production code should keep this field as None.
    """
    if cache is not None:
        real_cache = False
    else:
        real_cache = True
        cache = linecache.cache

    for file_name, profile in code_frags.items():
        if real_cache and os.path.exists(file_name):
            # skip rewriting caches of existing files
            continue

        if file_name not in cache:
            # keep field 1 (mtime) as None to ensure lazy cache
            cache[file_name] = (
                profile["size"],
                None,
                [""] * profile["lines"],
                file_name,
            )
        for fragment in profile["fragments"]:
            left_range, right_range = fragment["left"], fragment["right"]
            cache[file_name][2][left_range:right_range] = fragment["code"]
