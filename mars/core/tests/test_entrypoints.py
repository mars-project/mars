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
import types
import warnings
import pkg_resources


class _DummyClass(object):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "_DummyClass(%f, %f)" % self.value


def test_init_entrypoint():
    # FIXME: Python 2 workaround because nonlocal doesn't exist
    counters = {"init": 0}

    def init_function():
        counters["init"] += 1

    mod = types.ModuleType("_test_mars_extension")
    mod.init_func = init_function

    try:
        # will remove this module at the end of the test
        sys.modules[mod.__name__] = mod

        # We are registering an entry point using the "mars" package
        # ("distribution" in pkg_resources-speak) itself, though these are
        # normally registered by other packages.
        dist = "pymars"
        entrypoints = pkg_resources.get_entry_map(dist)
        my_entrypoint = pkg_resources.EntryPoint(
            "init",  # name of entry point
            mod.__name__,  # module with entry point object
            attrs=["init_func"],  # name of entry point object
            dist=pkg_resources.get_distribution(dist),
        )
        entrypoints.setdefault("mars_extensions", {})["init"] = my_entrypoint

        from .. import entrypoints

        # Allow reinitialization
        entrypoints.init_extension_entrypoints.cache_clear()

        entrypoints.init_extension_entrypoints()

        # was our init function called?
        assert counters["init"] == 1

        # ensure we do not initialize twice
        entrypoints.init_extension_entrypoints()
        assert counters["init"] == 1
    finally:
        # remove fake module
        if mod.__name__ in sys.modules:
            del sys.modules[mod.__name__]


def test_entrypoint_tolerance():
    # FIXME: Python 2 workaround because nonlocal doesn't exist
    counters = {"init": 0}

    def init_function():
        counters["init"] += 1
        raise ValueError("broken")

    mod = types.ModuleType("_test_mars_bad_extension")
    mod.init_func = init_function

    try:
        # will remove this module at the end of the test
        sys.modules[mod.__name__] = mod

        # We are registering an entry point using the "mars" package
        # ("distribution" in pkg_resources-speak) itself, though these are
        # normally registered by other packages.
        dist = "pymars"
        entrypoints = pkg_resources.get_entry_map(dist)
        my_entrypoint = pkg_resources.EntryPoint(
            "init",  # name of entry point
            mod.__name__,  # module with entry point object
            attrs=["init_func"],  # name of entry point object
            dist=pkg_resources.get_distribution(dist),
        )
        entrypoints.setdefault("mars_extensions", {})["init"] = my_entrypoint

        from .. import entrypoints

        # Allow reinitialization
        entrypoints.init_extension_entrypoints.cache_clear()

        with warnings.catch_warnings(record=True) as w:
            entrypoints.init_extension_entrypoints()

        bad_str = "Mars extension module '_test_mars_bad_extension'"
        for x in w:
            if bad_str in str(x):
                break
        else:
            raise ValueError("Expected warning message not found")

        # was our init function called?
        assert counters["init"] == 1

    finally:
        # remove fake module
        if mod.__name__ in sys.modules:
            del sys.modules[mod.__name__]
