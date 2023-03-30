# Copyright 2022 XProbe Inc.
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

from ...tests.core import require_cupy
from ...utils import lazy_import
from .. import nvutils


cupy = lazy_import("cupy")


@require_cupy
def test_nvutil():
    device_info = nvutils.get_device_info(0)
    assert device_info.uuid is not None

    # run something
    _ = cupy.ones(10)

    handle = nvutils.get_handle_by_index(0)
    assert nvutils._running_process_matches(handle)
    assert nvutils.get_cuda_context().has_context

    info = nvutils.get_index_and_uuid(0)
    info2 = nvutils.get_index_and_uuid(info.uuid)
    assert info.device_index == info2.device_index
    assert info.uuid == info2.uuid
