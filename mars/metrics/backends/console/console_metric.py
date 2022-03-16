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
from typing import Optional, Dict, Tuple

from ..metric import (
    AbstractMetric,
    AbstractCounter,
    AbstractGauge,
    AbstractHistogram,
    AbstractMeter,
)

logger = logging.getLogger(__name__)


class SimpleMetric:
    def __init__(
        self, name: str, description: str = "", tag_keys: Optional[Tuple[str]] = None
    ):
        self._name = name
        self._description = description
        self._tag_keys = tag_keys
        self._value = 0

    def update(self, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        self._value = value
        logger.debug(
            "Reporting metric with name: %s, description: %s, value: %s, tags: %s",
            self._name,
            self._description,
            value,
            tags,
        )

    @property
    def value(self):
        return self._value


class ConsoleMetricMixin(AbstractMetric):
    @property
    def value(self):
        return self._metric.value

    def _init(self):
        self._metric = SimpleMetric(self._name, self._description, self._tag_keys)

    def _record(self, value=1, tags: Optional[Dict[str, str]] = None):
        self._metric.update(value, tags)


class Counter(ConsoleMetricMixin, AbstractCounter):
    pass


class Gauge(ConsoleMetricMixin, AbstractGauge):
    pass


class Meter(ConsoleMetricMixin, AbstractMeter):
    pass


class Histogram(ConsoleMetricMixin, AbstractHistogram):
    pass
