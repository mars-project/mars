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

from .....tests.core import require_ray, mock
from .....utils import lazy_import
from ..utils import report_event

ray = lazy_import("ray")


@require_ray
@mock.patch("ray.report_event")
def test_report_event(fake_report_event, ray_start_regular):
    arguments = []

    def _report_event(*args):
        arguments.extend(args)

    fake_report_event.side_effect = _report_event
    severity, label, message = "WARNING", "test_label", "test_message"
    report_event(severity, label, message)
    assert arguments == [ray.EventSeverity.WARNING, label, message]
