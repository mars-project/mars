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
import time
import weakref

from contextlib import contextmanager
from enum import Enum
from queue import PriorityQueue
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from .backends.console import console_metric
from .backends.metric import AbstractMetric
from .backends.prometheus import prometheus_metric
from .backends.ray import ray_metric

logger = logging.getLogger(__name__)

_init = False
_metric_backend = "console"
_backends_cls = {
    "console": console_metric,
    "prometheus": prometheus_metric,
    "ray": ray_metric,
}


_metrics_to_be_initialized = weakref.WeakSet()


def init_metrics(backend="console", config: Dict[str, Any] = None):
    global _init
    if _init is True:
        return

    backend = backend or "console"
    if backend not in _backends_cls:
        raise NotImplementedError(f"Do not support metric backend {backend}")
    global _metric_backend
    _metric_backend = backend
    if _metric_backend == "prometheus":
        try:
            from prometheus_client import start_http_server
            from ..utils import get_next_port

            port = config.get("port", 0) if config else 0
            port = port or get_next_port()
            start_http_server(port)
            logger.warning(
                "Finished startup prometheus http server and port is %d", port
            )
        except ImportError:
            logger.warning(
                "Failed to start prometheus http server because there is no prometheus_client"
            )
    _init = True
    for m in _metrics_to_be_initialized:
        cls = getattr(_backends_cls[_metric_backend], m.type)
        metric = cls(m.name, m.description, m.tag_keys)
        m.set_metric(metric)
    logger.info("Finished initialize the metrics of backend: %s.", _metric_backend)


def shutdown_metrics():
    global _metric_backend
    _metric_backend = "console"
    global _init
    _init = False
    logger.info("Shutdown metrics of backend: %s.", _metric_backend)


class _MetricWrapper(AbstractMetric):
    _metric: AbstractMetric
    _log_not_init_error: bool

    def __init__(
        self,
        name: str,
        description: str = "",
        tag_keys: Optional[Tuple[str]] = None,
        metric_type: str = "Counter",
    ):
        self._name = name
        self._description = description
        self._tag_keys = tag_keys or tuple()
        self._type = metric_type
        self._metric = None
        self._log_not_init_error = False

    @property
    def type(self):
        return self._type

    @property
    def value(self):
        assert (
            self._metric is not None
        ), "Metric is not initialized, please call `init_metrics()` before using metrics."
        return self._metric.value

    def set_metric(self, metric):
        assert metric is not None, "Argument metric is None, please check it."
        self._metric = metric

    def record(self, value=1, tags: Optional[Dict[str, str]] = None):
        if self._metric is not None:
            self._metric.record(value, tags)
        elif not self._log_not_init_error:
            self._log_not_init_error = True
            logger.warning(
                "Metric is not initialized, please call `init_metrics()` before using metrics."
            )


def gen_metric(func):
    def wrapper(name, descriptions: str = "", tag_keys: Optional[Tuple[str]] = None):
        if _init is True:
            return func(name, descriptions, tag_keys)
        else:
            logger.info(
                "Metric %s will be initialized when invoking `init_metrics()`.", name
            )
            metric = _MetricWrapper(
                name, descriptions, tag_keys, func.__name__.capitalize()
            )
            _metrics_to_be_initialized.add(metric)
            return metric

    return wrapper


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
    @gen_metric
    def counter(name, description: str = "", tag_keys: Optional[Tuple[str]] = None):
        logger.info(
            "Initializing a counter with name: %s, tag keys: %s, backend: %s",
            name,
            tag_keys,
            _metric_backend,
        )
        return _backends_cls[_metric_backend].Counter(name, description, tag_keys)

    @staticmethod
    @gen_metric
    def gauge(name, description: str = "", tag_keys: Optional[Tuple[str]] = None):
        logger.info(
            "Initializing a gauge whose name: %s, tag keys: %s, backend: %s",
            name,
            tag_keys,
            _metric_backend,
        )
        return _backends_cls[_metric_backend].Gauge(name, description, tag_keys)

    @staticmethod
    @gen_metric
    def meter(name, description: str = "", tag_keys: Optional[Tuple[str]] = None):
        logger.info(
            "Initializing a meter whose name: %s, tag keys: %s, backend: %s",
            name,
            tag_keys,
            _metric_backend,
        )
        return _backends_cls[_metric_backend].Meter(name, description, tag_keys)

    @staticmethod
    @gen_metric
    def histogram(name, description: str = "", tag_keys: Optional[Tuple[str]] = None):
        logger.info(
            "Initializing a histogram whose name: %s, tag keys: %s, backend: %s",
            name,
            tag_keys,
            _metric_backend,
        )
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
