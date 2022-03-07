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

import time

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

_THRESHOLD = 2000
_RECORDED_INTERVAL_SECS = 1


class AbstractMetric(ABC):
    """Base class of metrics."""

    _type = None

    def __init__(
        self, name: str, description: str = "", tag_keys: Optional[Tuple[str]] = None
    ):
        self._name = name
        self._description = description
        self._tag_keys = tuple(tag_keys) if tag_keys else tuple()
        self._init()

    @property
    def type(self):
        return self._type

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def tag_keys(self):
        return self._tag_keys

    @abstractmethod
    def _init(self):
        """Some initialization in subclass."""
        pass

    def record(self, value=1, tags: Optional[Dict[str, str]] = None):
        """A public method called by users."""
        pass

    @abstractmethod
    def _record(self, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """An internal method called by record() and should be
        implemented by different metric backends.
        """
        pass


class AbstractCounter(AbstractMetric):
    """A counter records the counts of events."""

    _type = "counter"

    def __init__(
        self, name: str, description: str = "", tag_keys: Optional[Tuple[str]] = None
    ):
        super().__init__(name, description, tag_keys)
        self._count = 0

    def record(self, value=1, tags: Optional[Dict[str, str]] = None):
        self._count += value
        self._record(self._count, tags)


class AbstractGauge(AbstractMetric):
    """A gauge represents a single numerical value that can be
    arbitrarily set.
    """

    _type = "gauge"

    def record(self, value=1, tags: Optional[Dict[str, str]] = None):
        self._record(value, tags)


class AbstractMeter(AbstractMetric):
    """A meter measures the rate at which a set of events occur."""

    _type = "meter"

    def __init__(
        self, name: str, description: str = "", tag_keys: Optional[Tuple[str]] = None
    ):
        super().__init__(name, description, tag_keys)
        self._count = 0
        self._last_time = time.time()

    def record(self, value=1, tags: Optional[Dict[str, str]] = None):
        self._count += value
        now = time.time()
        past = now - self._last_time
        if self._count >= _THRESHOLD or past >= _RECORDED_INTERVAL_SECS:
            qps = self._count / past
            self._record(qps, tags)
            self._last_time = now
            self._count = 0


class AbstractHistogram(AbstractMetric):
    """A histogram measures the distribution of values in a stream of data."""

    _type = "histogram"

    def __init__(
        self, name: str, description: str = "", tag_keys: Optional[Tuple[str]] = None
    ):
        super().__init__(name, description, tag_keys)
        self._data = list()
        self._last_time = time.time()

    def record(self, value=1, tags: Optional[Dict[str, str]] = None):
        self._data.append(value)
        now = time.time()
        if (
            len(self._data) >= _THRESHOLD
            or now - self._last_time >= _RECORDED_INTERVAL_SECS
        ):
            avg = sum(self._data) / len(self._data)
            self._record(avg, tags)
            self._data.clear()
            self._last_time = now
