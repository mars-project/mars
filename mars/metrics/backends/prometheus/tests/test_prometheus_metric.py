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
import time

try:
    import requests
except ImportError:
    requests = None

try:
    from prometheus_client import start_http_server
except ImportError:
    start_http_server = None

from .....utils import get_next_port
from ..prometheus_metric import Counter, Gauge, Histogram, Meter

_PROMETHEUS_CLIENT_PORT = get_next_port()


@pytest.fixture(scope="module")
def start_prometheus_http_server():
    if start_http_server:
        start_http_server(_PROMETHEUS_CLIENT_PORT)


def verify_metric(name, value, delta=1e-6):
    if start_http_server is None or requests is None:
        return
    resp = requests.get("http://127.0.0.1:{}".format(_PROMETHEUS_CLIENT_PORT)).text
    assert name in resp
    lines = resp.splitlines()
    for line in lines:
        if line.startswith(name):
            items = line.split(" ")
            assert len(items) == 2
            assert pytest.approx(float(items[1]), abs=delta) == value


def test_counter(start_prometheus_http_server):
    c = Counter("test_counter", "A test counter", ("service", "tenant"))
    assert c.name == "test_counter"
    assert c.description == "A test counter"
    assert set(["host", "pid"]).issubset(set(c.tag_keys))
    assert set(["service", "tenant"]).issubset(set(c.tag_keys))
    assert c.type == "Counter"
    c.record(1, {"service": "mars", "tenant": "test"})
    verify_metric("test_counter", 1.0)
    c.record(2, {"service": "mars", "tenant": "test"})
    verify_metric("test_counter", 3.0)


def test_gauge(start_prometheus_http_server):
    g = Gauge("test_gauge", "A test gauge")
    assert g.name == "test_gauge"
    assert g.description == "A test gauge"
    assert set(["host", "pid"]).issubset(set(g.tag_keys))
    assert g.type == "Gauge"
    g.record(0.1)
    verify_metric("test_gauge", 0.1)
    g.record(1.1)
    verify_metric("test_gauge", 1.1)


def test_meter(start_prometheus_http_server):
    m = Meter("test_meter")
    assert m.name == "test_meter"
    assert m.description == ""
    assert set(["host", "pid"]).issubset(set(m.tag_keys))
    assert m.type == "Meter"
    num = 3
    while num > 0:
        m.record(1)
        time.sleep(1)
        num -= 1
    verify_metric("test_meter", 1, 0.05)


def test_histogram(start_prometheus_http_server):
    h = Histogram("test_histogram")
    assert h.name == "test_histogram"
    assert h.description == ""
    assert set(["host", "pid"]).issubset(set(h.tag_keys))
    assert h.type == "Histogram"
    num = 3
    while num > 0:
        h.record(1)
        h.record(2)
        time.sleep(1)
        num -= 1
    verify_metric("test_histogram", 1.5, 0.15)
    num = 3
    while num > 0:
        h.record(3)
        time.sleep(1)
        num -= 1
    verify_metric("test_histogram", 3, 0.1)
