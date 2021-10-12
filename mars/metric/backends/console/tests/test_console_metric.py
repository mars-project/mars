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

from ..console_metric import CounterImpl, GaugeImpl, MeterImpl, HistogramImpl


def test_counter():
    c = CounterImpl("test_counter", "A test counter", ("service", "tenant"))
    c.record(1, {"service": "mars", "tenant": "test"})
    c.record(2, {"service": "mars", "tenant": "test"})
    assert c.name == "test_counter"
    assert c.tag_keys == ("service", "tenant")
    assert c.value == 3


def test_gauge():
    g = GaugeImpl("test_gauge", "A test gauge")
    g.record(1)
    assert g.name == "test_gauge"
    assert g.tag_keys == ()
    assert g.value == 1


def test_meter():
    m = MeterImpl("test_meter")
    m.record(1)
    assert m.name == "test_meter"
    assert m.tag_keys == ()


def test_histogram():
    h = HistogramImpl("test_histogram")
    h.record(1)
    assert h.name == "test_histogram"
    assert h.tag_keys == ()
