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

import os
import socket

from typing import Optional, Dict

from ....utils import lazy_import
from ..metric import (
    AbstractMetric,
    AbstractCounter,
    AbstractGauge,
    AbstractMeter,
    AbstractHistogram,
)

pc = lazy_import("prometheus_client", rename="pc")


class PrometheusMetricMixin(AbstractMetric):
    def _init(self):
        # Prometheus metric name must match the regex `[a-zA-Z_:][a-zA-Z0-9_:]*`
        # `.` is a common character in metrics, so here replace it with `:`
        self._name = self._name.replace(".", ":")
        self._tag_keys = self._tag_keys + (
            "host",
            "pid",
        )
        self._tags = {"host": socket.gethostname(), "pid": os.getpid()}
        try:
            self._metric = (
                pc.Gauge(self._name, self._description, self._tag_keys) if pc else None
            )
        except ValueError:  # pragma: no cover
            self._metric = None

    def _record(self, value=1, tags: Optional[Dict[str, str]] = None):
        if self._metric:
            if tags is not None:
                tags.update(self._tags)
            else:
                tags = self._tags
            self._metric.labels(**tags).set(value)


class Counter(PrometheusMetricMixin, AbstractCounter):
    pass


class Gauge(PrometheusMetricMixin, AbstractGauge):
    pass


class Meter(PrometheusMetricMixin, AbstractMeter):
    pass


class Histogram(PrometheusMetricMixin, AbstractHistogram):
    pass
