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

from .....tests.core import require_ray
from ..ray_metric import Counter, Gauge, Meter, Histogram


@require_ray
def test_record():
    c = Counter("test_counter")
    assert c.record(1) is None


@require_ray
def test_counter():
    c = Counter("test_counter", "A test counter", ("service", "tenant"))
    assert c.name == "test_counter"
    assert c.description == "A test counter"
    assert c.tag_keys == ("service", "tenant")
    assert c.type == "Counter"
    assert c.record(1, {"service": "mars", "tenant": "test"}) is None


@require_ray
def test_gauge():
    g = Gauge("test_gauge", "A test gauge")
    assert g.name == "test_gauge"
    assert g.description == "A test gauge"
    assert g.tag_keys == ()
    assert g.type == "Gauge"
    assert g.record(1) is None


@require_ray
def test_meter():
    m = Meter("test_meter")
    assert m.name == "test_meter"
    assert m.description == ""
    assert m.tag_keys == ()
    assert m.type == "Meter"
    assert m.record(1) is None


@require_ray
def test_histogram():
    h = Histogram("test_histogram")
    assert h.name == "test_histogram"
    assert h.description == ""
    assert h.tag_keys == ()
    assert h.type == "Histogram"
    assert h.record(1) is None
