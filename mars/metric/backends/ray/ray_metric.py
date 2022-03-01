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

from ....utils import lazy_import
from ..metric import (
    AbstractMetric,
    AbstractCounter,
    AbstractGauge,
    AbstractMeter,
    AbstractHistogram,
)

ray_metrics = lazy_import("ray.util.metrics")

# Note: Gauge `record` method is deprecated in ray 1.3.0 version, so here
# make it compatible with the old and new ray versions.
RAY_GAUGE_SET_AVAILABLE = (
    True if ray_metrics and hasattr(ray_metrics.Gauge, "set") else False
)


class RayMetricMixin(AbstractMetric):
    def _init(self):
        if ray_metrics is not None:  # pragma: no branch
            self._metric = ray_metrics.Gauge(
                self._name, self._description, self._tag_keys
            )

    def _record(self, value=1, tags: Optional[Dict[str, str]] = None):
        if RAY_GAUGE_SET_AVAILABLE:
            self._metric.set(value, tags)
        elif ray_metrics is not None:  # pragma: no branch
            self._metric.record(value, tags)


class Counter(RayMetricMixin, AbstractCounter):
    pass


class Gauge(RayMetricMixin, AbstractGauge):
    pass


class Meter(RayMetricMixin, AbstractMeter):
    pass


class Histogram(RayMetricMixin, AbstractHistogram):
    pass
