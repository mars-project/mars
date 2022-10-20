#!/usr/bin/env python
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
Workaround for https://github.com/pytest-dev/pytest-cov/issues/406 :
we remove coverage files without cython tracers.
"""

import glob
import logging
import os
import sqlite3

logger = logging.getLogger(__name__)


def check_coverage_file(file_name):
    try:
        conn = sqlite3.connect(file_name)
        tracers = list(conn.execute("SELECT * FROM tracer"))
        if len(tracers) < 1:
            raise ValueError("File containing no tracers")
    except Exception as exc:  # noqa: E722
        logger.warning(
            "Failed to resolve coverage file %s due to error %r", file_name, exc
        )
        os.unlink(file_name)


def main():
    for cov_file in glob.glob(".coverage.*"):
        check_coverage_file(cov_file)


if __name__ == "__main__":
    main()
