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

import functools
import gzip
import struct
import zlib
from collections import namedtuple
from distutils.version import LooseVersion
from enum import Enum
from io import BytesIO
import pickle  # nosec
import warnings

import numpy as np
import pandas as pd
try:
    import scipy.sparse as sps
except ImportError:  # pragma: no cover
    sps = None

from ..errors import SerializationFailed
from ..lib.sparse import SparseNDArray
from ..lib.groupby_wrapper import GroupByWrapper

try:
    import pyarrow
    try:
        from pyarrow.serialization import SerializationCallbackError
    except ImportError:
        from pyarrow.lib import SerializationCallbackError
except ImportError:  # pragma: no cover
    pyarrow = None
    SerializationCallbackError = Exception
try:
    from pyarrow.lib import ArrowNotImplementedError
except ImportError:  # pragma: no cover
    ArrowNotImplementedError = type('ArrowNotImplementedError', (Exception,), {})


try:
    import vineyard
except ImportError:  # pragma: no cover
    vineyard = None


BUFFER_SIZE = 256 * 1024


class DummyCompress(object):
    def __enter__(self):  # pragma: no cover
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # pragma: no cover
        pass

    @staticmethod
    def compress(data):  # pragma: no cover
        return data

    @staticmethod
    def decompress(data):  # pragma: no cover
        return data


try:
    import lz4.frame
    try:
        lz4.frame._compression.BUFFER_SIZE = BUFFER_SIZE
    except AttributeError:  # pragma: no cover
        pass
    lz4_open = functools.partial(lz4.frame.open, block_size=lz4.frame.BLOCKSIZE_MAX1MB)
    lz4_compress = functools.partial(lz4.frame.compress, block_size=lz4.frame.BLOCKSIZE_MAX1MB)
    lz4_compressobj = lz4.frame.LZ4FrameCompressor
    lz4_decompress = lz4.frame.decompress
    lz4_decompressobj = lz4.frame.LZ4FrameDecompressor
except ImportError:  # pragma: no cover
    lz4_open = None
    lz4_compress, lz4_compressobj = None, None
    lz4_decompress, lz4_decompressobj = None, None

gz_open = gzip.open
gz_compressobj = functools.partial(
    lambda level=-1: zlib.compressobj(level, zlib.DEFLATED, 16 + zlib.MAX_WBITS, zlib.DEF_MEM_LEVEL, 0)
)
gz_decompressobj = functools.partial(lambda: zlib.decompressobj(16 + zlib.MAX_WBITS))
gz_compress = gzip.compress
gz_decompress = gzip.decompress

SERIAL_VERSION = 0


class _EnumTagMixin:
    @property
    def tag(self):
        return self._tags[self]

    @classmethod
    def from_tag(cls, tag):
        try:
            rev_tags = cls._rev_tags
        except AttributeError:
            rev_tags = cls._rev_tags = {v: k for k, v in cls._tags.items()}
        return rev_tags[tag]

    def __gt__(self, other):
        return self._tags[self] > self._tags[other]


class CompressType(_EnumTagMixin, Enum):
    NONE = 'none'
    LZ4 = 'lz4'
    GZIP = 'gzip'
CompressType._tags = {  # noqa: E305
    CompressType.NONE: 0,
    CompressType.LZ4: 1,
    CompressType.GZIP: 2,
}


class SerialType(_EnumTagMixin, Enum):
    ARROW = 'arrow'
    PICKLE = 'pickle'
SerialType._tags = {  # noqa: E305
    SerialType.ARROW: 0,
    SerialType.PICKLE: 1,
}


compressors = {
    CompressType.LZ4: lz4_compress,
    CompressType.GZIP: gz_compress,
}
compressobjs = {
    CompressType.NONE: DummyCompress,
    CompressType.LZ4: lz4_compressobj,
    CompressType.GZIP: gz_compressobj,
}
decompressors = {
    CompressType.LZ4: lz4_decompress,
    CompressType.GZIP: gz_decompress,
}
decompressobjs = {
    CompressType.NONE: DummyCompress,
    CompressType.LZ4: lz4_decompressobj,
    CompressType.GZIP: gz_decompressobj,
}
compress_openers = {
    CompressType.LZ4: lz4_open,
    CompressType.GZIP: gz_open,
}


def get_supported_compressions():
    return set(k for k, v in decompressors.items() if v is not None)


def get_compressobj(compress):
    return compressobjs[compress]()


def get_decompressobj(compress):
    return decompressobjs[compress]()


def open_compression_file(file, compress):
    if compress != CompressType.NONE:
        file = compress_openers[compress](file, 'wb')
    return file


def open_decompression_file(file, compress):
    if compress != CompressType.NONE:
        file = compress_openers[compress](file, 'rb')
    return file


file_header = namedtuple('FileHeader', 'type version nbytes compress')
HEADER_LENGTH = 12


def read_file_header(file):
    if hasattr(file, 'read'):
        header_bytes = file.read(HEADER_LENGTH)
    else:
        header_bytes = file[:HEADER_LENGTH]
    type_ = header_bytes[0]
    version = header_bytes[1]
    nbytes, = struct.unpack('<Q', header_bytes[2:10])
    compress, = struct.unpack('<H', header_bytes[10:12])
    return file_header(SerialType.from_tag(type_), version, nbytes, CompressType.from_tag(compress))


def write_file_header(file, header):
    file.write(struct.pack('B', header.type.tag))
    file.write(struct.pack('B', header.version))
    file.write(struct.pack('<Q', header.nbytes))
    file.write(struct.pack('<H', header.compress.tag))


def peek_file_header(file):
    pos = file.tell()
    try:
        return read_file_header(file)
    finally:
        file.seek(pos)


_PANDAS_HAS_MGR = hasattr(pd.Series([0]), '_mgr')


def _patch_pandas_mgr(pd_obj):  # pragma: no cover
    # as pandas prior to 1.1.0 use _data instead of _mgr to hold BlockManager,
    # deserializing from high versions may produce mal-functioned pandas objects,
    # thus the patch is needed
    if _PANDAS_HAS_MGR:
        return pd_obj
    if hasattr(pd_obj, '_mgr') and isinstance(pd_obj, (pd.DataFrame, pd.Series)):
        pd_obj._data = pd_obj._mgr
        del pd_obj._mgr
    return pd_obj


def load(file):
    header = read_file_header(file)
    file = open_decompression_file(file, header.compress)

    try:
        buf = file.read()
    finally:
        if header.compress != CompressType.NONE:
            file.close()

    if header.type == SerialType.ARROW:
        return deserialize(memoryview(buf))
    else:
        return _patch_pandas_mgr(pickle.loads(buf))  # nosec


def loads(buf):
    mv = memoryview(buf)
    header = read_file_header(mv)
    compress = header.compress

    if compress == CompressType.NONE:
        data = buf[HEADER_LENGTH:]
    else:
        data = decompressors[compress](mv[HEADER_LENGTH:])

    if header.type == SerialType.ARROW:
        return deserialize(memoryview(data))
    else:
        return _patch_pandas_mgr(pickle.loads(data))  # nosec


def dump(obj, file, *, serial_type=None, compress=None, pickle_protocol=None):
    if serial_type is None:
        serial_type = SerialType.ARROW if pyarrow is not None else SerialType.PICKLE
    if compress is None:
        compress = CompressType.NONE
    try:
        if serial_type == SerialType.ARROW:
            serialized = serialize(obj)
            data_size = serialized.total_bytes
            write_file_header(file, file_header(serial_type, SERIAL_VERSION, data_size, compress))
            file = open_compression_file(file, compress)
            serialized.write_to(file)
        else:
            pickle_protocol = pickle_protocol or pickle.HIGHEST_PROTOCOL
            serialized = pickle.dumps(obj, protocol=pickle_protocol)
            data_size = len(serialized)
            write_file_header(file, file_header(serial_type, SERIAL_VERSION, data_size, compress))
            file = open_compression_file(file, compress)
            file.write(serialized)
    finally:
        if compress != CompressType.NONE:
            file.close()
    return


def dumps(obj, *, serial_type=None, compress=None, pickle_protocol=None):
    sio = BytesIO()
    dump(obj, sio, serial_type=serial_type, compress=compress, pickle_protocol=pickle_protocol)
    return sio.getvalue()


def _wrap_deprecates(fun):
    @functools.wraps(fun)
    def _wrapped(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            return fun(*args, **kwargs)

    if pyarrow is not None and LooseVersion(pyarrow.__version__) >= '2.0':
        return _wrapped
    return fun


@_wrap_deprecates
def serialize(data):
    if isinstance(data, pyarrow.SerializedPyObject):
        return data
    try:
        return pyarrow.serialize(data, mars_serialize_context())
    except (ArrowNotImplementedError, SerializationCallbackError):
        if isinstance(data, (complex, np.complexfloating)):
            return pyarrow.serialize(_ComplexWrapper(data), mars_serialize_context())
    except pyarrow.lib.ArrowInvalid:
        if isinstance(data, np.ndarray) and any(s < 0 for s in data.strides):
            # pyarrow does not support negative strides for now
            return pyarrow.serialize(data.copy(), mars_serialize_context())

    try:
        return pyarrow.serialize(_PickleWrapper(data), mars_serialize_context())
    except (ArrowNotImplementedError, SerializationCallbackError,
            pyarrow.lib.ArrowInvalid, pickle.PickleError):
        raise SerializationFailed(obj=data)


@_wrap_deprecates
def deserialize(data):
    result = pyarrow.deserialize(data, mars_serialize_context())
    return _patch_pandas_mgr(result)


_FreezeGrouping = namedtuple('_FreezeGrouping', 'name codes grouper ')


class _ComplexWrapper:
    def __init__(self, cmplx):
        if isinstance(cmplx, complex):
            self.dtype, self.real, self.imag = None, cmplx.real, cmplx.imag
        else:
            self.dtype, self.real, self.imag = cmplx.dtype.str, cmplx.real, cmplx.imag

    def serialize(self):
        return self.dtype, self.real, self.imag

    @classmethod
    def deserialize(cls, serialized):
        if serialized[0] is None:
            return complex(*serialized[1:])
        else:
            return np.dtype(serialized[0]).type(np.complex(*serialized[1:]))


class _PickleWrapper:
    def __init__(self, obj):
        self._obj = obj

    def serialize(self):
        return pickle.dumps(self._obj)

    @classmethod
    def deserialize(cls, serialized):
        return pickle.loads(serialized)  # nosec


def _serialize_groupby_wrapper(obj: GroupByWrapper):
    return obj.to_tuple(pickle_function=True, truncate=True)


def _deserialize_groupby_wrapper(serialized):
    return GroupByWrapper.from_tuple(serialized)


if pyarrow and LooseVersion(pyarrow.__version__) < LooseVersion('0.16'):
    def _serialize_sparse_nd_array(obj):
        return obj.data, obj.indices, obj.indptr, obj.shape

    def _deserialize_sparse_nd_array(serialized):
        data, indices, indptr, shape = serialized
        empty_arr = np.zeros(0, dtype=data.dtype)

        target_csr = sps.coo_matrix((empty_arr, (empty_arr,) * 2), dtype=data.dtype,
                                    shape=shape if len(shape) > 1 else (1, shape[0])).tocsr()

        target_csr.data, target_csr.indices, target_csr.indptr = data, indices, indptr
        return SparseNDArray(target_csr, shape=shape)
else:  # pragma: no cover
    def _serialize_sparse_nd_array(obj):
        return obj.raw, obj.shape

    def _deserialize_sparse_nd_array(serialized):
        data, shape = serialized
        return SparseNDArray(data, shape=shape)


def _serialize_pandas_interval(obj: pd.Interval):
    return [obj.left, obj.right, obj.closed]


def _deserialize_pandas_interval(data):
    return pd.Interval(data[0], data[1], data[2])


def _serialze_pandas_categorical(obj: pd.Categorical):
    return [obj.codes, obj.dtype]


def _deserialize_pandas_categorical(data):
    return pd.Categorical.from_codes(data[0], dtype=data[1])


def _serialize_pandas_categorical_dtype(obj: pd.CategoricalDtype):
    return [obj.categories, obj.ordered]


def _deserialize_pandas_categorical_dtype(data):
    return pd.CategoricalDtype(data[0], data[1])


def _serialize_arrow_dtype(obj):
    return type(obj), obj.name


def _deserialize_arrow_dtype(obj):
    tp, name = obj
    return tp.construct_from_string(name)


def _serialize_arrow_array(obj):
    return obj.dtype, obj._arrow_array.chunks


def _deserialize_arrow_array(obj):
    dtype, chunks = obj
    arrow_array = pyarrow.chunked_array(chunks)
    return dtype.construct_array_type()(arrow_array)


def _serialize_sparse_array(obj):
    return obj.sp_index.__reduce__(), obj.sp_values, obj.dtype


def _deserialzie_sparse_array(obj):
    (sparse_index_type, sparse_index_args), sparse_values, dtype = obj
    sparse_index = sparse_index_type(*sparse_index_args)
    return pd.arrays.SparseArray(sparse_values, sparse_index=sparse_index,
                                 dtype=dtype)


def _serialize_sparse_dtype(obj):
    return obj.subtype.str, obj.fill_value


def _deserialize_sparse_dtype(obj):
    dtype, fill_value = obj
    return pd.SparseDtype(dtype=dtype, fill_value=fill_value)


_serialize_context = None


def _apply_pyarrow_serialization_patch(serialization_context):  # pragma: no cover
    """
    Fix the bug about dtype serialization in pyarrow (pyarrow#4953).

    From the JIRA of arrow, the fixed version of this bug is 1.0, so we only apply
    the patch for pyarrow less than version 1.0.
    """
    import pyarrow

    try:
        # This function is available after numpy-0.16.0.
        # See also: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
        from numpy.lib.format import descr_to_dtype
    except ImportError:
        def descr_to_dtype(descr):
            '''
            descr may be stored as dtype.descr, which is a list of
            (name, format, [shape]) tuples where format may be a str or a tuple.
            Offsets are not explicitly saved, rather empty fields with
            name, format == '', '|Vn' are added as padding.
            This function reverses the process, eliminating the empty padding fields.
            '''
            if isinstance(descr, str):
                # No padding removal needed
                return np.dtype(descr)
            elif isinstance(descr, tuple):
                # subtype, will always have a shape descr[1]
                dt = descr_to_dtype(descr[0])
                return np.dtype((dt, descr[1]))
            fields = []
            offset = 0
            for field in descr:
                if len(field) == 2:
                    name, descr_str = field
                    dt = descr_to_dtype(descr_str)
                else:
                    name, descr_str, shape = field
                    dt = np.dtype((descr_to_dtype(descr_str), shape))

                # Ignore padding bytes, which will be void bytes with '' as name
                # Once support for blank names is removed, only "if name == ''" needed)
                is_pad = (name == '' and dt.type is np.void and dt.names is None)
                if not is_pad:
                    fields.append((name, dt, offset))

                offset += dt.itemsize

            names, formats, offsets = zip(*fields)
            # names may be (title, names) tuples
            nametups = (n if isinstance(n, tuple) else (None, n) for n in names)
            titles, names = zip(*nametups)
            return np.dtype({'names': names, 'formats': formats, 'titles': titles,
                             'offsets': offsets, 'itemsize': offset})

    def _serialize_numpy_array_list(obj):
        if obj.dtype.str != '|O':
            # Make the array c_contiguous if necessary so that we can call change
            # the view.
            if not obj.flags.c_contiguous:
                obj = np.ascontiguousarray(obj)
            return obj.view('uint8'), np.lib.format.dtype_to_descr(obj.dtype)
        else:
            return obj.tolist(), np.lib.format.dtype_to_descr(obj.dtype)

    def _deserialize_numpy_array_list(data):
        if data[1] != '|O':
            assert data[0].dtype == np.uint8
            return data[0].view(descr_to_dtype(data[1]))
        else:
            return np.array(data[0], dtype=np.dtype(data[1]))

    if LooseVersion(pyarrow.__version__) < LooseVersion('0.15'):
        serialization_context.register_type(
            np.ndarray, 'np.array',
            custom_serializer=_serialize_numpy_array_list,
            custom_deserializer=_deserialize_numpy_array_list)


def mars_serialize_context():
    from ..dataframe.arrays import ArrowArray, ArrowDtype

    global _serialize_context
    if _serialize_context is None:
        ctx = pyarrow.default_serialization_context()
        ctx.register_type(_PickleWrapper, 'mars.PickleWrapper',
                          custom_serializer=_PickleWrapper.serialize,
                          custom_deserializer=_PickleWrapper.deserialize)
        ctx.register_type(_ComplexWrapper, 'mars.ComplexWrapper',
                          custom_serializer=_ComplexWrapper.serialize,
                          custom_deserializer=_ComplexWrapper.deserialize)
        ctx.register_type(SparseNDArray, 'mars.SparseNDArray',
                          custom_serializer=_serialize_sparse_nd_array,
                          custom_deserializer=_deserialize_sparse_nd_array)
        ctx.register_type(GroupByWrapper, 'pandas.GroupByWrapper',
                          custom_serializer=_serialize_groupby_wrapper,
                          custom_deserializer=_deserialize_groupby_wrapper)
        ctx.register_type(pd.Interval, 'pandas.Interval',
                          custom_serializer=_serialize_pandas_interval,
                          custom_deserializer=_deserialize_pandas_interval)
        ctx.register_type(pd.Categorical, 'pandas.Categorical',
                          custom_serializer=_serialze_pandas_categorical,
                          custom_deserializer=_deserialize_pandas_categorical)
        ctx.register_type(pd.CategoricalDtype, 'pandas.CategoricalDtype',
                          custom_serializer=_serialize_pandas_categorical_dtype,
                          custom_deserializer=_deserialize_pandas_categorical_dtype)
        ctx.register_type(ArrowDtype, 'mars.dataframe.ArrowDtype',
                          custom_serializer=_serialize_arrow_dtype,
                          custom_deserializer=_deserialize_arrow_dtype)
        ctx.register_type(ArrowArray, 'mars.dataframe.ArrowArray',
                          custom_serializer=_serialize_arrow_array,
                          custom_deserializer=_deserialize_arrow_array)
        ctx.register_type(pd.arrays.SparseArray, 'pandas.arrays.SparseArray',
                          custom_serializer=_serialize_sparse_array,
                          custom_deserializer=_deserialzie_sparse_array)
        ctx.register_type(pd.SparseDtype, 'pandas.SparseDtype',
                          custom_serializer=_serialize_sparse_dtype,
                          custom_deserializer=_deserialize_sparse_dtype)
        _apply_pyarrow_serialization_patch(ctx)
        _serialize_context = ctx
    return _serialize_context
