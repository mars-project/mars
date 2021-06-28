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

from typing import Union, List, Dict, Any

from .core import Serializer, buffered

try:
    import pyarrow as pa
    pa_types = Union[pa.Table, pa.RecordBatch]
except ImportError:  # pragma: no cover
    pa = None
    pa_types = Any


class ArrowBatchSerializer(Serializer):
    serializer_name = 'arrow'

    @buffered
    def serialize(self, obj: pa_types, context: Dict):
        header = {}

        sink = pa.BufferOutputStream()
        writer = pa.RecordBatchStreamWriter(sink, obj.schema)
        if isinstance(obj, pa.Table):
            header['type'] = 'Table'
            writer.write_table(obj)
        else:
            header['type'] = 'Batch'
            writer.write_batch(obj)
        writer.close()

        buf = sink.getvalue()
        buffers = [buf]
        return header, buffers

    def deserialize(self, header: Dict, buffers: List, context: Dict):
        reader = pa.RecordBatchStreamReader(pa.BufferReader(buffers[0]))
        if header['type'] == 'Table':
            return reader.read_all()
        else:
            return reader.read_next_batch()


if pa is not None:  # pragma: no branch
    ArrowBatchSerializer.register(pa.Table)
    ArrowBatchSerializer.register(pa.RecordBatch)
