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

import os
import sys
from pathlib import PurePath

_MATCH_FILES = [
    "*.py",
    "*.pyx",
]
_IGNORES = [
    "mars/learn/**/*.pyx",
    "mars/lib/**/*.py",
    "mars/lib/*.pyx",
    "mars/_version.py",
]


def main():
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    miss_files = []
    for root, _dirs, files in os.walk(os.path.join(root_path, "mars")):
        for fn in files:
            rel_path = os.path.relpath(os.path.join(root, fn), root_path)
            if any(PurePath(rel_path).match(patt) for patt in _IGNORES):
                continue
            if all(not PurePath(rel_path).match(patt) for patt in _MATCH_FILES):
                continue

            file_path = os.path.join(root, fn)
            with open(file_path, "rb") as input_file:
                file_lines = [
                    line
                    for line in input_file.read().split(b"\n")
                    if line.startswith(b"#")
                ]
            comments = b"\n".join(file_lines)
            if b"Copyright" not in comments:
                miss_files.append(rel_path)
    if miss_files:
        file_list = "\n    ".join(miss_files)
        sys.stderr.write(
            f"Please add missing copyright header for files:\n    {file_list}\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
