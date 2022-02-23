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

from ..metric import Metric, Counter, Gauge, Meter, Histogram


def test_dummy_metric():
    Metric.__abstractmethods__ = set()

    class DummyMetric(Metric):
        pass

    m = DummyMetric("dummy_metric", "A test metric", ("service", "tenant"))
    assert isinstance(m, Metric)
    assert m.name == "dummy_metric"
    assert m.description == "A test metric"
    assert m.tag_keys == ("service", "tenant")
    assert m.type is None
    assert m._init() is None
    assert m.record() is None
    assert m._record() is None


def test_counter():
    Counter.__abstractmethods__ = set()
    c = Counter("test_counter", "A test counter", ("service", "tenant"))
    assert c.name == "test_counter"
    assert c.description == "A test counter"
    assert c.tag_keys == ("service", "tenant")
    assert c.type == "counter"
    assert c.record(1, {"service": "mars", "tenant": "test"}) is None


def test_gauge():
    Gauge.__abstractmethods__ = set()
    g = Gauge("test_gauge", "A test gauge")
    assert g.name == "test_gauge"
    assert g.description == "A test gauge"
    assert g.tag_keys == ()
    assert g.type == "gauge"
    assert g.record(1) is None


def test_meter():
    Meter.__abstractmethods__ = set()
    m = Meter("test_meter")
    assert m.name == "test_meter"
    assert m.description == ""
    assert m.tag_keys == ()
    assert m.type == "meter"
    assert m.record(1) is None


def test_histogram():
    Histogram.__abstractmethods__ = set()
    h = Histogram("test_histogram")
    assert h.name == "test_histogram"
    assert h.description == ""
    assert h.tag_keys == ()
    assert h.type == "histogram"
    assert h.record(1) is None
