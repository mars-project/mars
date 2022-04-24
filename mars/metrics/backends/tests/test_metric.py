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

from ..metric import (
    AbstractMetric,
    AbstractCounter,
    AbstractGauge,
    AbstractMeter,
    AbstractHistogram,
)


def test_illegal_arguments():
    class DummyMetric(AbstractMetric):
        pass

    DummyMetric.__abstractmethods__ = set()
    with pytest.raises(AssertionError):
        DummyMetric(1)

    with pytest.raises(AssertionError):
        DummyMetric("dummy_metric", 1)

    with pytest.raises(AssertionError):
        DummyMetric("dummy_metric", "A test metric", "service")

    with pytest.raises(AssertionError):
        DummyMetric("dummy_metric", "A test metric", ("service", 1))


def test_dummy_metric():
    class DummyMetric(AbstractMetric):
        pass

    DummyMetric.__abstractmethods__ = set()
    m = DummyMetric("dummy_metric", "A test metric", ("service", "tenant"))
    assert isinstance(m, AbstractMetric)
    assert m.name == "dummy_metric"
    assert m.description == "A test metric"
    assert m.tag_keys == ("service", "tenant")
    assert m.type is None
    assert m._init() is None
    assert m.record() is None
    assert m._record() is None


def test_counter():
    class DummyCounter(AbstractCounter):
        pass

    DummyCounter.__abstractmethods__ = set()
    c = DummyCounter("test_counter", "A test counter", ("service", "tenant"))
    assert c.name == "test_counter"
    assert c.description == "A test counter"
    assert c.tag_keys == ("service", "tenant")
    assert c.type == "counter"
    assert c.record(1, {"service": "mars", "tenant": "test"}) is None


def test_gauge():
    class DummyGauge(AbstractGauge):
        pass

    DummyGauge.__abstractmethods__ = set()
    g = DummyGauge("test_gauge", "A test gauge")
    assert g.name == "test_gauge"
    assert g.description == "A test gauge"
    assert g.tag_keys == ()
    assert g.type == "gauge"
    assert g.record(1) is None


def test_meter():
    class DummyMeter(AbstractMeter):
        pass

    DummyMeter.__abstractmethods__ = set()
    m = DummyMeter("test_meter")
    assert m.name == "test_meter"
    assert m.description == ""
    assert m.tag_keys == ()
    assert m.type == "meter"
    assert m.record(1) is None


def test_histogram():
    class DummyHistogram(AbstractHistogram):
        pass

    DummyHistogram.__abstractmethods__ = set()
    h = DummyHistogram("test_histogram")
    assert h.name == "test_histogram"
    assert h.description == ""
    assert h.tag_keys == ()
    assert h.type == "histogram"
    assert h.record(1) is None
