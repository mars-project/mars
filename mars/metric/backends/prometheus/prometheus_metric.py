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

from typing import Optional, Dict

try:
    from prometheus_client import Gauge as PGauge
except ImportError:
    PGauge = None

from ..metric import (
    AbstractMetric,
    AbstractCounter,
    AbstractGauge,
    AbstractMeter,
    AbstractHistogram,
)


class PrometheusMetricMixin(AbstractMetric):
    def _init(self):
        self._metric = (
            PGauge(self._name, self._description, self._tag_keys) if PGauge else None
        )

    def _record(self, value=1, tags: Optional[Dict[str, str]] = None):
        if self._metric:
            if tags:
                self._metric.labels(**tags).set(value)
            else:
                self._metric.set(value)


class Counter(PrometheusMetricMixin, AbstractCounter):
    pass


class Gauge(PrometheusMetricMixin, AbstractGauge):
    pass


class Meter(PrometheusMetricMixin, AbstractMeter):
    pass


class Histogram(PrometheusMetricMixin, AbstractHistogram):
    pass
