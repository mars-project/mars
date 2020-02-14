# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

import numpy as np
try:
    from PIL import Image
except ImportError:
    Image = None

from ... import opcodes as OperandDef
from ...serialize import Int32Field, AnyField
from ...config import options
from ...filesystem import open_file, glob, file_size
from ...utils import ceildiv
from ..operands import TensorOperandMixin, TensorOperand


def _read_image(fpath):
    return np.asarray(Image.open(fpath))


class TensorImread(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.IMREAD

    _filepath = AnyField('filepath')
    _chunk_frames = Int32Field('chunk_frames')

    def __init__(self, filepath=None, chunk_frames=None, **kwargs):
        super().__init__(_filepath=filepath, _chunk_frames=chunk_frames, **kwargs)

    @property
    def filepath(self):
        return self._filepath

    @property
    def chunk_frames(self):
        return self._chunk_frames

    @classmethod
    def tile(cls, op):
        out_shape = op.outputs[0].shape
        paths = op.filepath if isinstance(op.filepath, (tuple, list)) else glob(op.filepath)
        n_chunks = ceildiv(len(paths), op.chunk_frames)
        if len(paths) > 1:
            chunks = []
            splits = []
            for i in range(n_chunks):
                chunk_op = op.copy().reset_key()
                chunk_op._filepath = paths[i * op.chunk_frames: (i + 1) * op.chunk_frames]
                file_nums = len(chunk_op._filepath)
                shape = (file_nums,) + out_shape[1:]
                chunk = chunk_op.new_chunk(None, shape=shape, index=(i,) + (0,) * (len(out_shape) - 1))
                chunks.append(chunk)
                splits.append(file_nums)
            nsplits = (tuple(splits),) + tuple((s,) for s in out_shape[1:])
        else:
            chunk_op = op.copy().reset_key()
            chunks = [chunk_op.new_chunk(None, shape=out_shape)]
            nsplits = tuple((s,) for s in out_shape)
        new_op = op.copy()
        return new_op.new_tensors(None, shape=out_shape, chunks=chunks, nsplits=nsplits)

    @classmethod
    def execute(cls, ctx, op):
        if isinstance(op.filepath, list):
            datas = []
            for path in op.filepath:
                with open_file(path, 'rb') as f:
                    datas.append(_read_image(f))
            ctx[op.outputs[0].key] = np.array(datas)
        else:
            with open_file(op.filepath, 'rb') as f:
                ctx[op.outputs[0].key] = np.array(_read_image(f))

    def __call__(self, shape):
        return self.new_tensor(None, shape)


def imread(path, chunk_frames=None):
    paths = path if isinstance(path, (tuple, list)) else glob(path)
    with open_file(paths[0], 'rb') as f:
        sample_data = _read_image(f)
        img_shape = sample_data.shape
        img_size = file_size(paths[0])
    if len(paths) > 1:
        shape = (len(paths), ) + img_shape
    else:
        shape = img_shape
    if chunk_frames is None:
        chunk_frames = int(options.chunk_store_limit / img_size)
    op = TensorImread(filepath=path, chunk_frames=chunk_frames)
    return op(shape=shape)
