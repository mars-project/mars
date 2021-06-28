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

from gzip import GzipFile
from typing import BinaryIO

try:
    import lz4
    import lz4.frame
except ImportError:  # pragma: no cover
    lz4 = None


_compressions = {
    'gzip': lambda f: GzipFile(fileobj=f)
}

if lz4:
    _compressions['lz4'] = lz4.frame.open


def compress(file: BinaryIO,
             compress_type: str) -> BinaryIO:
    """
    Return a compressed file object.

    Parameters
    ----------
    file:
        file object.
    compress_type: str
       compression type.

    Returns
    -------
    compressed_file:
        compressed file object.
    """
    try:
        compress_ = _compressions[compress_type]
    except KeyError:  # pragma: no cover
        raise ValueError(f'Unknown compress type: {compress_type}, '
                         f'available include: {", ".join(_compressions)}')

    return compress_(file)
