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

import os
import time
from typing import Callable, Dict, List, Union, TextIO

import yaml

DEFAULT_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'oscar/config.yml')


def wait_services_ready(selectors: List, min_counts: List[int],
                        count_fun: Callable, timeout=None):
    readies = [0] * len(selectors)
    start_time = time.time()
    while True:
        all_satisfy = True
        for idx, selector in enumerate(selectors):
            if readies[idx] < min_counts[idx]:
                all_satisfy = False
                readies[idx] = count_fun(selector)
                break
        if all_satisfy:
            break
        if timeout and timeout + start_time < time.time():
            raise TimeoutError('Wait cluster start timeout')
        time.sleep(1)


def load_service_config_file(path: Union[str, TextIO]) -> Dict:
    import mars
    mars_path = os.path.dirname(os.path.abspath(mars.__file__))

    cfg_stack = []  # type: List[Dict]
    cfg_file_set = set()
    if isinstance(path, str):
        path = os.path.abspath(path)

    while path is not None:
        if path in cfg_file_set:  # pragma: no cover
            raise ValueError('Recursive config inherit detected')

        if not hasattr(path, 'read'):
            with open(path) as file:
                cfg = yaml.safe_load(file)
        else:
            cfg = yaml.safe_load(path)
        cfg_stack.append(cfg)
        cfg_file_set.add(path)

        inherit_path = cfg.pop('inherits', None)
        if not inherit_path:
            path = None
        elif os.path.isfile(inherit_path):
            path = inherit_path
        elif inherit_path == '@default':
            path = DEFAULT_CONFIG_FILE
        elif inherit_path.startswith('@mars'):
            path = inherit_path.replace('@mars', mars_path)
        else:
            path = os.path.join(os.path.dirname(path), inherit_path)

    def _override_cfg(src: Union[Dict, List], override: Union[Dict, List]):
        if isinstance(override, dict):
            for key, val in override.items():
                if key not in src or not isinstance(val, (list, dict)):
                    src[key] = val
                else:
                    _override_cfg(src[key], override[key])
        else:
            src.extend(override)

    cfg = cfg_stack[-1]
    for new_cfg in cfg_stack[-2::-1]:
        _override_cfg(cfg, new_cfg)
    return cfg
