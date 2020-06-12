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


def find_objects(nested, types):
    found = []
    stack = [nested]

    while len(stack) > 0:
        it = stack.pop()
        if isinstance(it, types):
            found.append(it)
            continue

        if isinstance(it, (list, tuple, set)):
            stack.extend(list(it)[::-1])
        elif isinstance(it, dict):
            stack.extend(list(it.values())[::-1])

    return found


def replace_inputs(nested, mapping):
    if not mapping:
        return nested

    if isinstance(nested, dict):
        vals = list(nested.values())
    else:
        vals = list(nested)

    new_vals = []
    for val in vals:
        if isinstance(val, (dict, list, tuple, set)):
            new_val = replace_inputs(val, mapping)
        else:
            try:
                new_val = mapping.get(val, val)
            except TypeError:
                new_val = val
        new_vals.append(new_val)

    if isinstance(nested, dict):
        return type(nested)((k, v) for k, v in zip(nested.keys(), new_vals))
    else:
        return type(nested)(new_vals)
