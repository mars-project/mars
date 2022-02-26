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

import pytest

from ..api import init_metrics, Metrics, _metric_backend


@pytest.fixture
def init():
    init_metrics()


def test_init_metrics():
    init_metrics()
    assert _metric_backend == "console"
    init_metrics({"metric": {}})
    assert _metric_backend == "console"
    init_metrics({"metric": {"backend": "console"}})
    assert _metric_backend == "console"
    with pytest.raises(NotImplementedError):
        init_metrics({"metric": {"backend": "not_exist"}})


def test_counter(init):
    c = Metrics.counter("test_counter", "A test counter", ("service", "tenant"))
    assert c.name == "test_counter"
    assert c.description == "A test counter"
    assert c.tag_keys == ("service", "tenant")
    assert c.type == "counter"
    c.record(1, {"service": "mars", "tenant": "test"})
    c.record(2, {"service": "mars", "tenant": "test"})
    assert c.value == 3


def test_gauge():
    g = Metrics.gauge("test_gauge", "A test gauge")
    assert g.name == "test_gauge"
    assert g.description == "A test gauge"
    assert g.tag_keys == ()
    assert g.type == "gauge"
    g.record(1)
    assert g.value == 1
    g.record(2)
    assert g.value == 2


def test_meter():
    m = Metrics.meter("test_meter")
    assert m.name == "test_meter"
    assert m.description == ""
    assert m.tag_keys == ()
    assert m.type == "meter"
    m.record(1)
    assert m.value == 0
    m.record(2001)
    assert m.value > 0


def test_histogram():
    h = Metrics.histogram("test_histogram")
    assert h.name == "test_histogram"
    assert h.description == ""
    assert h.tag_keys == ()
    assert h.type == "histogram"
    h.record(1)
    assert h.value == 0
    for i in range(2002):
        h.record(1)
    assert h.value > 0
