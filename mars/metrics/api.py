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

import logging

from contextlib import contextmanager
from enum import Enum
from queue import PriorityQueue
import time
from typing import Optional, Tuple, NamedTuple, Callable, List

from .backends.console import console_metric
from .backends.prometheus import prometheus_metric
from .backends.ray import ray_metric

logger = logging.getLogger(__name__)

_metric_backend = "console"
_backends_cls = {
    "console": console_metric,
    "prometheus": prometheus_metric,
    "ray": ray_metric,
}


def init_metrics(backend="console", port=0):
    backend = backend or "console"
    if backend not in _backends_cls:
        raise NotImplementedError(f"Do not support metric backend {backend}")
    global _metric_backend
    _metric_backend = backend
    if _metric_backend == "prometheus":
        try:
            from prometheus_client import start_http_server

            start_http_server(port)
            logger.info("Finished startup prometheus http server and port is %d", port)
        except ImportError:
            logger.warning(
                "Failed to start prometheus http server because there is no prometheus_client"
            )
    logger.info("Finished initialize the metrics with backend %s", _metric_backend)


class Metrics:
    """
    A factory to generate different types of metrics.

    Note:
        Counter, Meter and Histogram are not thread safe.

    Examples
    --------
    >>> c1 = counter('counter1', 'A counter')
    >>> c1.record(1)

    >>> c2 = counter('counter2', 'A counter', ('service', 'tenant'))
    >>> c2.record(1, {'service': 'mars', 'tenant': 'test'})

    >>> g1 = gauge('gauge1')
    >>> g1.record(1)

    >>> m1 = meter('meter1')
    >>> m1.record(1)

    >>> h1 = histogram('histogram1')
    >>> h1.record(1)
    """

    @staticmethod
    def counter(name, description: str = "", tag_keys: Optional[Tuple[str]] = None):
        return _backends_cls[_metric_backend].Counter(name, description, tag_keys)

    @staticmethod
    def gauge(name, description: str = "", tag_keys: Optional[Tuple[str]] = None):
        return _backends_cls[_metric_backend].Gauge(name, description, tag_keys)

    @staticmethod
    def meter(name, description: str = "", tag_keys: Optional[Tuple[str]] = None):
        return _backends_cls[_metric_backend].Meter(name, description, tag_keys)

    @staticmethod
    def histogram(name, description: str = "", tag_keys: Optional[Tuple[str]] = None):
        return _backends_cls[_metric_backend].Histogram(name, description, tag_keys)


class Percentile:
    class PercentileType(Enum):
        P99 = 1
        P95 = 2
        P90 = 3

    def __init__(self, capacity: int, window: int, callback: Callable[[float], None]):
        self._capacity = capacity
        self._window = window
        self._callback = callback
        self._min_heap = PriorityQueue()
        self._cur_num = 0

        if capacity <= 0 or window <= 0:
            raise ValueError(
                f"capacity or window expect to get a positive integer,"
                f"but capacity got: {capacity} and window got: {window}"
            )

    def record_data(self, value):
        store_value = -1 * value
        if self._min_heap.qsize() < self._capacity:
            self._min_heap.put(store_value)
        else:
            top_value = self._min_heap.get_nowait()
            store_value = store_value if top_value < store_value else top_value
            self._min_heap.put(store_value)

        self._cur_num += 1
        if self._cur_num % self._window == 0:
            self._callback(-1 * self._min_heap.get_nowait())
            self._cur_num = 0
            self._min_heap = PriorityQueue()

    @classmethod
    def build_p99(cls, callback: Callable[[float], None], window: int):
        return cls(int(window * 0.01), window, callback)

    @classmethod
    def build_p95(cls, callback: Callable[[float], None], window: int):
        return cls(int(window * 0.05), window, callback)

    @classmethod
    def build_p90(cls, callback: Callable[[float], None], window: int):
        return cls(int(window * 0.1), window, callback)


_percentile_builder = {
    Percentile.PercentileType.P99: Percentile.build_p99,
    Percentile.PercentileType.P95: Percentile.build_p95,
    Percentile.PercentileType.P90: Percentile.build_p90,
}


class PercentileArg(NamedTuple):
    percentile_type: Percentile.PercentileType
    callback: Callable[[float], None]
    window: int


@contextmanager
def record_time_cost_percentile(percentile_args: List[PercentileArg]):
    percentile_list = [
        _percentile_builder[percentile_type](callback, window)
        for percentile_type, callback, window in percentile_args
    ]
    st_time = time.time()

    yield

    cost_time = time.time() - st_time
    for percentile in percentile_list:
        percentile.record_data(cost_time)
