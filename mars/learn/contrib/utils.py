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

import sys

import numpy as np


def make_import_error_func(package_name):
    def _func(*_, **__):  # pragma: no cover
        raise ImportError(f'Cannot import {package_name}, please reinstall '
                          f'that package.')
    return _func


def pick_workers(workers, size):
    """
    Pick workers from a list.

    This method will try to pick workers as balanced as it can.

    1. If size <= len(workers), randomly pick workers from the list.
    2. If size > len(workers), just select all workers in a random order,
       then see the rest size, if it's still more than the workers size,
       return all workers in a random order, if not,
       randomly select workers from the list.

    :param workers: workers list
    :param size: number to pick from workers list
    :return: ndarray of selected workers whose length is `size`
    """
    result = np.empty(size, dtype=object)
    rest = size
    while rest > 0:
        start = size - rest
        to_pick_size = min(size - start, len(workers))
        result[start: start + to_pick_size] = \
            np.random.permutation(workers)[:to_pick_size]
        rest = rest - to_pick_size
    return result


def config_mod_getattr(mod_dict, globals_):
    def __getattr__(name):
        import importlib
        if name in mod_dict:
            mod_name, cls_name = mod_dict[name].rsplit('.', 1)
            mod = importlib.import_module(mod_name, globals_['__name__'])
            cls = globals_[name] = getattr(mod, cls_name)
            return cls
        else:  # pragma: no cover
            raise AttributeError(name)

    if sys.version_info[:2] < (3, 7):
        for _mod in mod_dict.keys():
            __getattr__(_mod)

    def __dir__():
        return sorted([n for n in globals_ if not n.startswith('_')]
                      + list(mod_dict))

    globals_.update({
        '__getattr__': __getattr__,
        '__dir__': __dir__,
        '__all__': list(__dir__()),
        '__warningregistry__': dict(),
    })
