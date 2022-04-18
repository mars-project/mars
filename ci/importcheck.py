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

import ast
import os
import sys
import textwrap
from pathlib import PurePath
from typing import List, NamedTuple, Optional, Tuple

_IGNORES = [
    "mars/lib/**/*.py",
    "conftest.py",
]


class CheckResult(NamedTuple):
    path: str
    lines: List
    absolute_imports: List[Tuple[int, int]]
    head_disorder: Optional[Tuple[int, int]]
    block_disorders: Optional[List[Tuple[int, int]]]

    @property
    def has_faults(self) -> bool:
        return (
            bool(self.absolute_imports)
            or bool(self.head_disorder)
            or bool(self.block_disorders)
        )


def _check_absolute_import(node: ast.AST) -> List[Tuple[int, int]]:
    res = set()
    if isinstance(node, ast.Import):
        for import_name in node.names:
            if import_name.name.startswith("mars."):
                res.add((node.lineno, node.end_lineno))
    elif isinstance(node, ast.ImportFrom):
        if node.level == 0 and node.module.startswith("mars."):
            res.add((node.lineno, node.end_lineno))
    elif getattr(node, "body", []):
        for body_item in node.body:
            res.update(_check_absolute_import(body_item))
    return sorted(res)


def check_imports(file_path) -> CheckResult:
    with open(file_path, "rb") as src_file:
        body = src_file.read()
        lines = body.splitlines()
        parsed = ast.parse(body, filename=file_path)
    # scan for imports
    abs_faults = _check_absolute_import(parsed)

    return CheckResult(file_path, lines, abs_faults, None, None)


def _extract_line_block(lines: List, lineno: int, end_lineno: int, indent: str):
    grab_lines = "\n".join(line.decode() for line in lines[lineno - 1 : end_lineno])
    return textwrap.indent(textwrap.dedent(grab_lines), indent)


def format_results(results: List[CheckResult], root_path):
    rel_import_count = sum(len(res.absolute_imports) for res in results)
    if rel_import_count > 0:
        print(
            f"Do not use absolute imports for mars module in "
            f"code ({rel_import_count}):",
            file=sys.stderr,
        )
        for res in results:
            if not res.absolute_imports:
                continue
            rel_path = os.path.relpath(res.path, root_path)
            print("  " + rel_path, file=sys.stderr)
            for lineno, end_lineno in res.absolute_imports:
                print(f"    Line {lineno}-{end_lineno}", file=sys.stderr)
                print(
                    _extract_line_block(res.lines, lineno, end_lineno, "      "),
                    file=sys.stderr,
                )


def main():
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results = []
    for root, _dirs, files in os.walk(os.path.join(root_path, "mars")):
        for fn in files:
            if "/tests" in root and not fn.startswith("test_"):
                # allow test auxiliary files to use full imports
                continue
            abs_path = os.path.join(root, fn)
            rel_path = os.path.relpath(abs_path, root_path)

            if not fn.endswith(".py"):
                continue
            if any(PurePath(rel_path).match(patt) for patt in _IGNORES):
                continue

            check_result = check_imports(abs_path)
            if check_result.has_faults:
                results.append(check_result)
    if results:
        results = sorted(results, key=lambda x: x.path)
        format_results(results, root_path)
        sys.exit(1)


if __name__ == "__main__":
    main()
