# Copyright 1999-2018 Alibaba Group Holding Ltd.
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


def concat_chunks_on_axis(chunks, axis=0):
    from ...tensor.core import CHUNK_TYPE as TENSOR_CHUNK_TYPE
    from ...dataframe.core import CHUNK_TYPE as DATAFRAME_CHUNK_TYPE

    if isinstance(chunks[0], TENSOR_CHUNK_TYPE):
        from ...tensor.utils import concat_chunks_on_axis
        return concat_chunks_on_axis(chunks, axis=axis)
    else:
        assert isinstance(chunks[0], DATAFRAME_CHUNK_TYPE)
        from ...dataframe.utils import concat_chunks_on_axis
        return concat_chunks_on_axis(chunks, axis=axis)


def get_fetch_op_cls(op):
    from ...tensor.core import CHUNK_TYPE as TENSOR_CHUNK_TYPE
    from ...dataframe.core import CHUNK_TYPE as DATAFRAME_CHUNK_TYPE

    if isinstance(op.outputs[0], TENSOR_CHUNK_TYPE):
        from ...tensor.utils import get_fetch_op_cls
        return get_fetch_op_cls(op)
    else:
        assert isinstance(op.outputs[0], DATAFRAME_CHUNK_TYPE)
        from ...dataframe.utils import get_fetch_op_cls
        return get_fetch_op_cls(op)


def get_fuse_op_cls(op):
    from ...tensor.core import CHUNK_TYPE as TENSOR_CHUNK_TYPE
    from ...dataframe.core import CHUNK_TYPE as DATAFRAME_CHUNK_TYPE

    if isinstance(op.outputs[0], TENSOR_CHUNK_TYPE):
        from ...tensor.utils import get_fuse_op_cls
        return get_fuse_op_cls(op)
    else:
        assert isinstance(op.outputs[0], DATAFRAME_CHUNK_TYPE)
        from ...dataframe.utils import get_fuse_op_cls
        return get_fuse_op_cls(op)
