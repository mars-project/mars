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

from ..console_metric import Counter, Gauge, Meter, Histogram


def test_counter():
    c = Counter("test_counter", "A test counter", ("service", "tenant"))
    assert c.name == "test_counter"
    assert c.description == "A test counter"
    assert c.tag_keys == ("service", "tenant")
    assert c.type == "Counter"
    c.record(1, {"service": "mars", "tenant": "test"})
    c.record(2, {"service": "mars", "tenant": "test"})
    assert c.value == 3


def test_gauge():
    g = Gauge("test_gauge", "A test gauge")
    assert g.name == "test_gauge"
    assert g.description == "A test gauge"
    assert g.tag_keys == ()
    assert g.type == "Gauge"
    g.record(1)
    assert g.value == 1
    g.record(2)
    assert g.value == 2


def test_meter():
    m = Meter("test_meter")
    assert m.name == "test_meter"
    assert m.description == ""
    assert m.tag_keys == ()
    assert m.type == "Meter"
    m.record(1)
    assert m.value == 0
    m.record(2001)
    assert m.value > 0


def test_histogram():
    h = Histogram("test_histogram")
    assert h.name == "test_histogram"
    assert h.description == ""
    assert h.tag_keys == ()
    assert h.type == "Histogram"
    h.record(1)
    assert h.value == 0
    for i in range(2002):
        h.record(1)
    assert h.value > 0
