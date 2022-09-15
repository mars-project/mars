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

try:  # pragma: no cover
    # make sure adlfs is installed
    from adlfs import AzureBlobFileSystem as _AzureBlobFileSystem

    # make sure fsspec is installed
    from .fsspec_adapter import FsSpecAdapter

    del _AzureBlobFileSystem
except ImportError:
    FsSpecAdapter = None

if FsSpecAdapter is not None:  # pragma: no cover
    from .core import register_filesystem

    class AzureBlobFileSystem(FsSpecAdapter):
        def __init__(self, **kwargs):
            super().__init__("az", **kwargs)

    register_filesystem("az", AzureBlobFileSystem)
    register_filesystem("abfs", AzureBlobFileSystem)
else:
    AzureBlobFileSystem = None
