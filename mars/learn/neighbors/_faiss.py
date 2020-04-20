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

import atexit
import os
import operator
import tempfile
from enum import Enum

import numpy as np
try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

from ... import opcodes as OperandDef
from ...context import RunningMode
from ...operands import OperandStage
from ...serialize import KeyField, StringField, Int64Field, \
    Int32Field, BoolField, Int8Field
from ...tiles import TilesError
from ...tensor import tensor as astensor
from ...tensor.core import TensorOrder
from ...tensor.random import RandomState
from ...tensor.array_utils import as_same_device, device
from ...tensor.utils import check_random_state, gen_random_seeds
from ...utils import check_chunks_unknown_shape, require_not_none, recursive_tile
from ..operands import LearnOperand, LearnOperandMixin, OutputType


class MemoryRequirementGrade(Enum):
    minimum = 0
    low = 1
    high = 2
    maximum = 3


if faiss is not None:
    METRIC_TO_FAISS_METRIC_TYPE = {
        'l2': faiss.METRIC_L2,
        'euclidean': faiss.METRIC_L2,
        'innerproduct': faiss.METRIC_INNER_PRODUCT,
        'cosine': faiss.METRIC_INNER_PRODUCT,
    }
else:  # pragma: no cover
    METRIC_TO_FAISS_METRIC_TYPE = {}


@require_not_none(faiss)
class FaissBuildIndex(LearnOperand, LearnOperandMixin):
    _op_type_ = OperandDef.FAISS_BUILD_INDEX

    _input = KeyField('input')
    _metric = StringField('metric')
    _faiss_index = StringField('faiss_index')
    _n_sample = Int64Field('n_sample')
    _seed = Int32Field('seed')
    _same_distribution = BoolField('same_distribution')
    _accuracy = BoolField('accuracy')
    _memory_require = Int8Field('memory_require',
                                on_serialize=operator.attrgetter('value'),
                                on_deserialize=MemoryRequirementGrade)
    # for test purpose, could be 'object', 'filename' or 'bytes'
    _return_index_type = StringField('return_index_type')

    def __init__(self, metric=None, faiss_index=None, n_sample=None, seed=None,
                 same_distribution=None, return_index_type=None,
                 accuracy=None, memory_require=None,
                 stage=None, output_types=None, gpu=None, **kw):
        super().__init__(_metric=metric, _faiss_index=faiss_index, _n_sample=n_sample,
                         _seed=seed, _same_distribution=same_distribution,
                         _return_index_type=return_index_type,
                         _accuracy=accuracy, _memory_require=memory_require, _gpu=gpu,
                         _stage=stage, _output_types=output_types, **kw)
        if self._output_types is None:
            self._output_types = [OutputType.object]

    @property
    def input(self):
        return self._input

    @property
    def metric(self):
        return self._metric

    @property
    def faiss_metric_type(self):
        return METRIC_TO_FAISS_METRIC_TYPE[self._metric]

    @property
    def faiss_index(self):
        return self._faiss_index

    @property
    def n_sample(self):
        return self._n_sample

    @property
    def seed(self):
        return self._seed

    @property
    def same_distribution(self):
        return self._same_distribution

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def memory_require(self):
        return self._memory_require

    @property
    def return_index_type(self):
        return self._return_index_type

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, X):
        return self.new_tileable([X])

    @classmethod
    def tile(cls, op):
        check_chunks_unknown_shape(op.inputs, TilesError)

        in_tensor = astensor(op.input, np.dtype(np.float32))._inplace_tile()
        if op.faiss_index == 'auto':
            faiss_index, n_sample = _gen_index_string_and_sample_count(
                in_tensor.shape, op.n_sample, op.accuracy, op.memory_require,
                gpu=op.gpu, **op.extra_params)
            op._n_sample = n_sample
        else:
            faiss_index, n_sample = op.faiss_index, op.n_sample

        if len(in_tensor.chunks) == 1:
            return cls._tile_one_chunk(op, faiss_index, n_sample)

        if in_tensor.chunk_shape[1] != 1:
            # make sure axis 1 has 1 chunk
            in_tensor = in_tensor.rechunk({1: in_tensor.shape[1]})._inplace_tile()
        return cls._tile_chunks(op, in_tensor, faiss_index, n_sample)

    @classmethod
    def _tile_one_chunk(cls, op, faiss_index, n_sample):
        in_chunk = op.input.chunks[0]
        chunk_op = op.copy().reset_key()
        chunk_op._faiss_index = faiss_index
        chunk_op._n_sample = n_sample
        chunk = chunk_op.new_chunk([in_chunk], index=in_chunk.index)

        new_op = op.copy()
        kw = op.outputs[0].params
        kw['chunks'] = [chunk]
        kw['nsplits'] = ((1,),)
        return new_op.new_tileables(op.inputs, kws=[kw])

    @classmethod
    def _tile_chunks(cls, op, in_tensor, faiss_index, n_sample):
        """
        If the distribution on each chunk is the same,
        refer to:
        https://github.com/facebookresearch/faiss/wiki/FAQ#how-can-i-distribute-index-building-on-several-machines

        1. train an IndexIVF* on a representative sample of the data, store it.
        2. for each node, load the trained index, add the local data to it, store the resulting populated index
        3. on a central node, load all the populated indexes and merge them.
        """
        faiss_index_ = faiss.index_factory(in_tensor.shape[1], faiss_index,
                                           op.faiss_metric_type)
        # Training on sample data when two conditions meet
        # 1. the index type requires for training, e.g. Flat does not require
        # 2. distributions of chunks are the same, in not,
        #    train separately on each chunk data
        need_sample_train = not faiss_index_.is_trained and op.same_distribution

        train_chunk = None
        if need_sample_train:
            # sample data to train
            rs = RandomState(op.seed)
            sampled_index = rs.choice(in_tensor.shape[0], size=n_sample,
                                      replace=False, chunk_size=n_sample)
            sample_tensor = recursive_tile(in_tensor[sampled_index])
            assert len(sample_tensor.chunks) == 1
            sample_chunk = sample_tensor.chunks[0]
            train_op = FaissTrainSampledIndex(faiss_index=faiss_index, metric=op.metric,
                                              return_index_type=op.return_index_type)
            train_chunk = train_op.new_chunk([sample_chunk])
        elif op.gpu:  # pragma: no cover
            # if not need train, and on gpu, just merge data together to train
            in_tensor = in_tensor.rechunk(in_tensor.shape)._inplace_tile()

        # build index for each input chunk
        build_index_chunks = []
        for i, chunk in enumerate(in_tensor.chunks):
            build_index_op = op.copy().reset_key()
            build_index_op._stage = OperandStage.map
            build_index_op._faiss_index = faiss_index
            if train_chunk is not None:
                build_index_chunk = build_index_op.new_chunk(
                    [chunk, train_chunk], index=(i,))
            else:
                build_index_chunk = build_index_op.new_chunk([chunk], index=(i,))
            build_index_chunks.append(build_index_chunk)

        out_chunks = []
        if need_sample_train:
            assert op.n_sample is not None
            # merge all indices into one, do only when trained on sample data
            out_chunk_op = op.copy().reset_key()
            out_chunk_op._faiss_index = faiss_index
            out_chunk_op._stage = OperandStage.agg
            out_chunk = out_chunk_op.new_chunk(build_index_chunks, index=(0,))
            out_chunks.append(out_chunk)
        else:
            out_chunks.extend(build_index_chunks)

        new_op = op.copy()
        return new_op.new_tileables(op.inputs, chunks=out_chunks,
                                    nsplits=((len(out_chunks),),))

    @classmethod
    def _execute_one_chunk(cls, ctx, op):
        (inp,), device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            # create index
            index = faiss.index_factory(inp.shape[1], op.faiss_index,
                                        op.faiss_metric_type)
            # GPU
            if device_id >= 0:  # pragma: no cover
                index = _index_to_gpu(index, device_id)

            # train index
            if not index.is_trained:
                assert op.n_sample is not None
                sample_indices = xp.random.choice(inp.shape[0],
                                                  size=op.n_sample, replace=False)
                sampled = inp[sample_indices]
                index.train(sampled)

            if op.metric == 'cosine':
                # faiss does not support cosine distances directly,
                # data needs to be normalize before adding to index,
                # refer to:
                # https://github.com/facebookresearch/faiss/wiki/FAQ#how-can-i-index-vectors-for-cosine-distance
                faiss.normalize_L2(inp)
            # add vectors to index
            if device_id >= 0:  # pragma: no cover
                # gpu
                inp = inp.astype(np.float32, copy=False)
                index.add_c(inp.shape[0], _swig_ptr_from_cupy_float32_array(inp))
            else:
                index.add(inp)

            ctx[op.outputs[0].key] = _store_index(ctx, op, index, device_id)

    @classmethod
    def _execute_map(cls, ctx, op):
        (data,), device_id, _ = as_same_device(
            [ctx[op.inputs[0].key]], device=op.device, ret_extra=True)
        index = ctx[op.inputs[1].key] if len(op.inputs) == 2 else None

        with device(device_id):
            if index is not None:
                # fetch the trained index
                trained_index = _load_index(ctx, op, index, device_id)
                return_index_type = _get_index_type(op.return_index_type, ctx)
                if return_index_type == 'object':
                    # clone a new one,
                    # because faiss does not ensure thread-safe for operations that change index
                    # https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls#thread-safety
                    trained_index = faiss.clone_index(trained_index)
            else:
                trained_index = faiss.index_factory(data.shape[1], op.faiss_index,
                                                    op.faiss_metric_type)
                if op.same_distribution:
                    # no need to train, just create index
                    pass
                else:
                    # distribution no the same, train on each chunk
                    trained_index.train(data)

                if device_id >= 0:  # pragma: no cover
                    trained_index = _index_to_gpu(trained_index, device_id)
            if op.metric == 'cosine':
                # faiss does not support cosine distances directly,
                # data needs to be normalize before adding to index,
                # refer to:
                # https://github.com/facebookresearch/faiss/wiki/FAQ#how-can-i-index-vectors-for-cosine-distance
                faiss.normalize_L2(data)

            # add data into index
            if device_id >= 0:  # pragma: no cover
                # gpu
                trained_index.add_c(data.shape[0], _swig_ptr_from_cupy_float32_array(data))
            else:
                trained_index.add(data)

            ctx[op.outputs[0].key] = _store_index(ctx, op, trained_index, device_id)

    @classmethod
    def _execute_agg(cls, ctx, op):
        device_id = op.device
        if device_id is None:
            device_id = -1
        inputs = [ctx[inp.key] for inp in op.inputs]

        with device(device_id):
            merged_index = None
            indexes = []
            for index in inputs:
                index = _load_index(ctx, op, index, device_id)
                indexes.append(index)
                assert hasattr(index, 'merge_from')
                if merged_index is None:
                    merged_index = index
                else:
                    merged_index.merge_from(index, index.ntotal)

            ctx[op.outputs[0].key] = _store_index(ctx, op, merged_index, device_id)

    @classmethod
    def execute(cls, ctx, op):
        if op.stage == OperandStage.map:
            cls._execute_map(ctx, op)
        elif op.stage == OperandStage.agg:
            cls._execute_agg(ctx, op)
        else:
            assert op.stage is None
            cls._execute_one_chunk(ctx, op)


def _get_index_type(return_index_type, ctx):
    if return_index_type is None:  # pragma: no cover
        if ctx.running_mode == RunningMode.local:
            return_index_type = 'object'
        elif ctx.running_mode == RunningMode.local_cluster:
            return_index_type = 'filename'
        else:
            return_index_type = 'bytes'
    return return_index_type


def _store_index(ctx, op, index, device_id):
    return_index_type = _get_index_type(op.return_index_type, ctx)

    if return_index_type == 'object':
        # no need to serialize
        return index
    elif return_index_type == 'filename':
        # save to file, then return filename
        if device_id >= 0:  # pragma: no cover
            # for gpu, convert to cpu first
            index = faiss.index_gpu_to_cpu(index)
        fn = tempfile.mkstemp('.index', prefix='faiss_')[1]
        faiss.write_index(index, fn)

        atexit.register(lambda: os.remove(fn))

        return fn
    else:
        if device_id >= 0:  # pragma: no cover
            # for gpu, convert to cpu first
            index = faiss.index_gpu_to_cpu(index)
        # distributed, save to file, then return in memory bytes
        fn = tempfile.mkstemp('.index', prefix='faiss_')[1]
        faiss.write_index(index, fn)
        try:
            with open(fn, 'rb') as f:
                return f.read()
        finally:
            os.remove(fn)


def _load_index(ctx, op, index, device_id):
    return_index_type = _get_index_type(op.return_index_type, ctx)

    if return_index_type == 'object':
        # local
        return index
    elif return_index_type == 'filename':
        # local cluster
        return faiss.read_index(index)
    else:
        # distributed
        fn = tempfile.mkstemp('.index', prefix='faiss_')[1]
        with open(fn, 'wb') as f:
            f.write(index)
        index = faiss.read_index(f.name)
        if device_id >= 0:  # pragma: no cover
            index = _index_to_gpu(index, device_id)
        return index


def _index_to_gpu(index, device_id):  # pragma: no cover
    res = faiss.StandardGpuResources()
    return faiss.index_cpu_to_gpu(res, device_id, index)


def _swig_ptr_from_cupy_float32_array(x):  # pragma: no cover
    assert x.flags.c_contiguous
    assert x.dtype == np.float32
    data_ptr = x.__cuda_array_interface__['data'][0]
    return faiss.cast_integer_to_float_ptr(data_ptr)


def _swig_ptr_from_cupy_int64_array(x):  # pragma: no cover
    assert x.flags.c_contiguous
    assert x.dtype == np.int64
    data_ptr = x.__cuda_array_interface__['data'][0]
    return faiss.cast_integer_to_long_ptr(data_ptr)


@require_not_none(faiss)
class FaissTrainSampledIndex(LearnOperand, LearnOperandMixin):
    _op_type_ = OperandDef.FAISS_TRAIN_SAMPLED_INDEX

    _input = KeyField('input')
    _metric = StringField('metric')
    _faiss_index = StringField('faiss_index')
    # for test purpose, could be 'object', 'filename' or 'bytes'
    _return_index_type = StringField('return_index_type')

    def __init__(self, faiss_index=None, metric=None,
                 return_index_type=None, output_types=None, **kw):
        super().__init__(_faiss_index=faiss_index, _metric=metric,
                         _return_index_type=return_index_type,
                         _output_types=output_types, **kw)
        if self._output_types is None:
            self._output_types = [OutputType.object]

    @property
    def input(self):
        return self._input

    @property
    def metric(self):
        return self._metric

    @property
    def faiss_metric_type(self):
        return METRIC_TO_FAISS_METRIC_TYPE[self.metric]

    @property
    def faiss_index(self):
        return self._faiss_index

    @property
    def return_index_type(self):
        return self._return_index_type

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    @classmethod
    def execute(cls, ctx, op):
        (data,), device_id, _ = as_same_device(
            [ctx[op.input.key]], device=op.device, ret_extra=True)

        with device(device_id):
            index = faiss.index_factory(data.shape[1], op.faiss_index,
                                        op.faiss_metric_type)

            if device_id >= 0:  # pragma: no cover
                # GPU
                index = _index_to_gpu(index, device_id)
                index.train_c(data.shape[0], _swig_ptr_from_cupy_float32_array(data))
            else:
                index.train(data)

            ctx[op.outputs[0].key] = _store_index(
                ctx, op, index, device_id)


def _gen_index_string_and_sample_count(shape, n_sample, accuracy, memory_require, gpu=False, **kw):
    """
    Generate index string and sample count according to guidance of faiss:
    https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
    """
    size, dim = shape
    memory_require = _get_memory_require(memory_require)

    if accuracy or size < 10 ** 5:
        # Flat is the only index that guarantees exact results
        # no need to train, thus sample count is None
        return 'Flat', None

    if memory_require == MemoryRequirementGrade.maximum and not gpu:
        x = kw.get('M', 32)  # get medium number by default
        if x < 4 or x > 64:
            raise ValueError('HNSWx requires M that between 4 and 64, '
                             'got {}'.format(x))
        return 'HNSW%d' % x, None

    if memory_require in (MemoryRequirementGrade.high, MemoryRequirementGrade.maximum):
        basement = '{},Flat'
    elif memory_require == MemoryRequirementGrade.low:
        x = kw.get('dim', dim // 2)
        basement = 'PCAR%d,{},SQ8' % x
    elif memory_require == MemoryRequirementGrade.minimum:
        x = kw.get('M', min(64, dim // 2))
        if x > 64:
            raise ValueError('PQx requires M <= 64, got {}'.format(x))
        y = kw.get('dim', None)
        if y is not None and y % x != 0:
            raise ValueError('OPQx_y requires dim is a multiple of M({}), '
                             'got dim: {}'.format(x, y))
        y = min(dim, 4 * x)
        y = x * (y // x)  # make sure y is a multiple of x
        basement = 'OPQ%(x)d_%(y)d,{},PQ%(x)d' % {'x': x, 'y': y}
    else:  # pragma: no cover
        raise ValueError('unknown memory require')

    # now choose the clustering options
    if size < 10 ** 6 or (size < 10 ** 7 and gpu):
        # < 1M, or <10M but need GPU
        k = kw.get('k', 5 * int(np.sqrt(size)))
        if k < 4 * int(np.sqrt(size)) or k > 16 * int(np.sqrt(size)):
            raise ValueError('k should be between 4 * sqrt(N) and 16 * sqrt(N), '
                             'got {}'.format(k))
        index_str = basement.format('IVF%d' % k)
        if n_sample is None:
            # 30 * k - 256 * k
            n_sample = min(30 * k, size)
    elif size < 10 ** 7 and not gpu:
        # 1M - 10M
        index_str = basement.format('IVF65536_HNSW32')
        if n_sample is None:
            # between 30 * 65536 and 256 * 65536
            n_sample = 32 * 65536
    elif size < 10 ** 8:
        index_str = basement.format('IVF65536_HNSW32')
        n_sample = 64 * 65536 if n_sample is None else n_sample
    else:
        index_str = basement.format('IVF1048576_HNSW32')
        n_sample = 64 * 65536 if n_sample is None else n_sample

    return index_str, n_sample


def _get_memory_require(memory_require):
    if isinstance(memory_require, str):
        return getattr(MemoryRequirementGrade, memory_require)
    elif isinstance(memory_require, MemoryRequirementGrade):
        return memory_require
    return MemoryRequirementGrade(memory_require)


@require_not_none(faiss)
def build_faiss_index(X, index_name='auto', n_sample=None, metric="euclidean",
                      random_state=None, same_distribution=True,
                      accuracy=False, memory_require=None, **kw):
    X = astensor(X)

    if metric not in METRIC_TO_FAISS_METRIC_TYPE:
        raise ValueError('unknown metric: {}'.format(metric))
    if index_name != 'auto':
        try:
            faiss.index_factory(X.shape[1], index_name,
                                METRIC_TO_FAISS_METRIC_TYPE[metric])
        except RuntimeError:
            raise ValueError('illegal faiss index: {}'.format(index_name))

    rs = check_random_state(random_state)
    if isinstance(rs, RandomState):
        rs = rs.to_numpy()
    seed = gen_random_seeds(1, rs)[0]
    if memory_require is None:
        memory_require = MemoryRequirementGrade.low
    else:
        memory_require = _get_memory_require(memory_require)
    op = FaissBuildIndex(faiss_index=index_name, metric=metric,
                         n_sample=n_sample, gpu=X.op.gpu, seed=seed,
                         same_distribution=same_distribution,
                         accuracy=accuracy, memory_require=memory_require, **kw)
    return op(X)


class FaissQuery(LearnOperand, LearnOperandMixin):
    _op_type_ = OperandDef.FAISS_QUERY

    _input = KeyField('input')
    _faiss_index = KeyField('faiss_index')
    _metric = StringField('metric')
    _n_neighbors = Int32Field('n_neighbors')
    _return_distance = BoolField('return_distance')
    _nprobe = Int64Field('nprobe')
    # for test purpose, could be 'object', 'filename' or 'bytes'
    _return_index_type = StringField('return_index_type')

    def __init__(self, faiss_index=None, metric=None, n_neighbors=None,
                 return_distance=None, return_index_type=None,
                 nprobe=None, output_types=None, gpu=None, **kw):
        super().__init__(_faiss_index=faiss_index, _n_neighbors=n_neighbors, _metric=metric,
                         _return_distance=return_distance, _output_types=output_types,
                         _nprobe=nprobe, _return_index_type=return_index_type, _gpu=gpu, **kw)
        if self._output_types is None:
            self._output_types = [OutputType.tensor] * self.output_limit

    @property
    def input(self):
        return self._input

    @property
    def faiss_index(self):
        return self._faiss_index

    @property
    def metric(self):
        return self._metric

    @property
    def n_neighbors(self):
        return self._n_neighbors

    @property
    def nprobe(self):
        return self._nprobe

    @property
    def return_distance(self):
        return self._return_distance

    @property
    def return_index_type(self):
        return self._return_index_type

    @property
    def output_limit(self):
        return 2 if self._return_distance else 1

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]
        if self._faiss_index is not None:
            self._faiss_index = self._inputs[1]

    def __call__(self, y):
        kws = []
        if self._return_distance:
            kws.append({'shape': (y.shape[0], self._n_neighbors),
                        'dtype': np.dtype(np.float32),
                        'order': TensorOrder.C_ORDER,
                        'type': 'distance'})
        kws.append({
            'shape': (y.shape[0], self._n_neighbors),
            'dtype': np.dtype(np.int64),
            'order': TensorOrder.C_ORDER,
            'type': 'indices'
        })
        return self.new_tileables([y, self._faiss_index], kws=kws)

    @classmethod
    def tile(cls, op):
        in_tensor = astensor(op.input)

        if in_tensor.chunk_shape[1] != 1:
            check_chunks_unknown_shape([in_tensor], TilesError)
            in_tensor = in_tensor.rechunk({1: in_tensor.shape[1]})._inplace_tile()

        out_chunks = [], []
        for chunk in in_tensor.chunks:
            chunk_op = op.copy().reset_key()
            chunk_kws = []
            if op.return_distance:
                chunk_kws.append({
                    'shape': (chunk.shape[0], op.n_neighbors),
                    'dtype': np.dtype(np.float32),
                    'order': TensorOrder.C_ORDER,
                    'index': chunk.index,
                    'type': 'distance'
                })
            chunk_kws.append({
                'shape': (chunk.shape[0], op.n_neighbors),
                'dtype': np.dtype(np.int64),
                'order': TensorOrder.C_ORDER,
                'index': chunk.index,
                'type': 'indices'
            })
            in_chunks = [chunk]
            in_chunks.extend(op.faiss_index.chunks)
            chunks = chunk_op.new_chunks(in_chunks, kws=chunk_kws)
            if op.return_distance:
                out_chunks[0].append(chunks[0])
            out_chunks[1].append(chunks[-1])

        new_op = op.copy()
        kws = [out.params for out in op.outputs]
        if op.return_distance:
            kws[0]['chunks'] = out_chunks[0]
            kws[0]['nsplits'] = (in_tensor.nsplits[0], (op.n_neighbors,))
        kws[-1]['chunks'] = out_chunks[1]
        kws[-1]['nsplits'] = (in_tensor.nsplits[0], (op.n_neighbors,))
        return new_op.new_tileables(op.inputs, kws=kws)

    @classmethod
    def execute(cls, ctx, op):
        (y,), device_id, xp = as_same_device(
            [ctx[op.input.key]], device=op.device, ret_extra=True)
        indexes = [_load_index(ctx, op, ctx[index.key], device_id)
                   for index in op.inputs[1:]]

        with device(device_id):
            y = xp.ascontiguousarray(y, dtype=np.float32)

            if len(indexes) == 1:
                index = indexes[0]
            else:
                index = faiss.IndexShards(indexes[0].d)
                [index.add_shard(ind) for ind in indexes]

            if op.metric == 'cosine':
                # faiss does not support cosine distances directly,
                # data needs to be normalize before searching,
                # refer to:
                # https://github.com/facebookresearch/faiss/wiki/FAQ#how-can-i-index-vectors-for-cosine-distance
                faiss.normalize_L2(y)

            if op.nprobe is not None:
                index.nprobe = op.nprobe

            if device_id >= 0:  # pragma: no cover
                n = y.shape[0]
                k = op.n_neighbors
                distances = xp.empty((n, k), dtype=xp.float32)
                indices = xp.empty((n, k), dtype=xp.int64)
                index.search_c(n, _swig_ptr_from_cupy_float32_array(y),
                               k, _swig_ptr_from_cupy_float32_array(distances),
                               _swig_ptr_from_cupy_int64_array(indices))
            else:
                distances, indices = index.search(y, op.n_neighbors)
            if op.return_distance:
                if index.metric_type == faiss.METRIC_L2:
                    # make it equivalent to `pairwise.euclidean_distances`
                    distances = xp.sqrt(distances, out=distances)
                elif op.metric == 'cosine':
                    # make it equivalent to `pairwise.cosine_distances`
                    distances = xp.subtract(1, distances, out=distances)
                ctx[op.outputs[0].key] = distances
            ctx[op.outputs[-1].key] = indices


@require_not_none(faiss)
def faiss_query(faiss_index, data, n_neighbors, return_distance=True, nprobe=None):
    data = astensor(data)
    op = FaissQuery(faiss_index=faiss_index, n_neighbors=n_neighbors,
                    metric=faiss_index.op.metric, return_distance=return_distance,
                    return_index_type=faiss_index.op.return_index_type,
                    nprobe=nprobe, gpu=data.op.gpu)
    ret = op(data)
    if not return_distance:
        return ret[0]
    return ret
