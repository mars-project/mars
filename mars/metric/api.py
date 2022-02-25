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

from typing import Dict, Any, Optional, Tuple

from .backends.console import console_metric
from .backends.prometheus import prometheus_metric

logger = logging.getLogger(__name__)

_metric_backend = "console"
_backends_cls = {
    "console": console_metric,
    "prometheus": prometheus_metric,
}


def init_metrics(config: Dict[str, Any] = None):
    metric_config = config.get("metric", {}) if config else {}
    global _metric_backend
    _metric_backend = metric_config.get("backend", "console")
    if _metric_backend not in _backends_cls:
        raise NotImplementedError(f"Do not support metric backend {_metric_backend}")
    if _metric_backend == "prometheus":
        conf = metric_config.get("conf", {})
        port = int(conf.get("port", 0))
        try:
            from prometheus_client import start_http_server

            start_http_server(port)
            logger.info("Finished startup prometheus http server and port is %d", port)
        except ImportError:
            logger.info(
                "Do not startup prometheus http server because there is no prometheus_client"
            )
    logger.info(
        "Finished initialize the metrics, config is %s, backend is %s",
        config,
        _metric_backend,
    )


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
