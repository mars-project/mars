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

import functools
import inspect
import threading

from ..config import options


_internal_mode = threading.local()


def is_eager_mode():
    in_kernel = is_kernel_mode()
    if not in_kernel:
        return options.eager_mode
    else:
        # in kernel, eager mode always False
        return False


def is_kernel_mode():
    try:
        return bool(_internal_mode.kernel)
    except AttributeError:
        _internal_mode.kernel = None
        return False


def is_build_mode():
    return bool(getattr(_internal_mode, 'build', False))


class _EnterModeFuncWrapper:
    def __init__(self, mode_name_to_value):
        self.mode_name_to_value = mode_name_to_value

        # as the wrapper may enter for many times
        # record old values for each time
        self.mode_name_to_value_list = list()

    def __enter__(self):
        mode_name_to_old_value = dict()
        for mode_name, value in self.mode_name_to_value.items():
            # record mode's old values
            mode_name_to_old_value[mode_name] = \
                getattr(_internal_mode, mode_name, None)
            if value is None:
                continue
            # set value
            setattr(_internal_mode, mode_name, value)
        self.mode_name_to_value_list.append(mode_name_to_old_value)

    def __exit__(self, *_):
        mode_name_to_old_value = self.mode_name_to_value_list.pop()
        for mode_name in self.mode_name_to_value.keys():
            # set back old values
            setattr(_internal_mode, mode_name,
                    mode_name_to_old_value[mode_name])

    def __call__(self, func):
        if not inspect.iscoroutinefunction(func):
            # sync
            @functools.wraps(func)
            def _inner(*args, **kwargs):
                with self:
                    return func(*args, **kwargs)
        else:
            # async
            @functools.wraps(func)
            async def _inner(*args, **kwargs):
                with self:
                    return await func(*args, **kwargs)

        return _inner


def enter_mode(kernel=None, build=None):
    mode_name_to_value = {
        'kernel': kernel,
        'build': build,
    }

    return _EnterModeFuncWrapper(mode_name_to_value)
