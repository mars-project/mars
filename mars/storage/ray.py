from typing import Any, Dict, List, Tuple
from ..lib import sparse
from ..utils import lazy_import, implements, register_ray_serializer
from .base import StorageBackend, StorageLevel, ObjectInfo, register_storage_backend
from .core import BufferWrappedFileObject, StorageFileObject

ray = lazy_import("ray")


# TODO(fyrestone): make the SparseMatrix pickleable.

def _mars_sparse_matrix_serializer(value):
    return [value.shape, value.spmatrix]


def _mars_sparse_matrix_deserializer(obj) -> sparse.SparseNDArray:
    shape, spmatrix = obj
    return sparse.matrix.SparseMatrix(spmatrix, shape=shape)


def _register_sparse_matrix_serializer():
    # register a custom serializer for Mars SparseMatrix
    register_ray_serializer(sparse.matrix.SparseMatrix,
                            serializer=_mars_sparse_matrix_serializer,
                            deserializer=_mars_sparse_matrix_deserializer)


class RayFileLikeObject:
    def __init__(self):
        self._buffers = []
        self._size = 0

    def write(self, content: bytes):
        self._buffers.append(content)
        self._size += len(content)

    def readinto(self, buffer):
        read_bytes = 0
        for b in self._buffers:
            read_pos = read_bytes + len(b)
            buffer[read_bytes:read_pos] = b
            read_bytes = read_pos
        return read_bytes

    def close(self):
        self._buffers.clear()
        self._size = 0

    def tell(self):
        return self._size


class RayFileObject(BufferWrappedFileObject):
    def __init__(self, object_id: Any, mode: str):
        self._object_id = object_id
        super().__init__(mode, size=0)

    def _write_init(self):
        self._buffer = RayFileLikeObject()

    def _read_init(self):
        self._buffer = ray.get(self._object_id)
        self._mv = memoryview(self._buffer)
        self._size = len(self._buffer)

    def write(self, content: bytes):
        if not self._initialized:
            self._write_init()
            self._initialized = True

        return self._buffer.write(content)

    def _write_close(self):
        worker = ray.worker.global_worker
        metadata = ray.ray_constants.OBJECT_METADATA_TYPE_RAW
        worker.core_worker.put_file_like_object(metadata, self._buffer.tell(), self._buffer,
                                                self._object_id)

    def _read_close(self):
        pass


@register_storage_backend
class RayStorage(StorageBackend):
    name = 'ray'

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    @implements(StorageBackend.setup)
    async def setup(cls, **kwargs) -> Tuple[Dict, Dict]:
        ray.init(ignore_reinit_error=True)
        _register_sparse_matrix_serializer()
        return dict(), dict()

    @staticmethod
    @implements(StorageBackend.teardown)
    async def teardown(**kwargs):
        pass

    @property
    @implements(StorageBackend.level)
    def level(self) -> StorageLevel:
        # TODO(fyrestone): return StorageLevel.MEMORY & StorageLevel.DISK
        # if object spilling is available.
        return StorageLevel.MEMORY

    @implements(StorageBackend.get)
    async def get(self, object_id, **kwargs) -> object:
        if kwargs:  # pragma: no cover
            raise NotImplementedError(f'Got unsupported args: {",".join(kwargs)}')
        return await object_id

    @implements(StorageBackend.put)
    async def put(self, obj, importance=0) -> ObjectInfo:
        object_id = ray.put(obj)
        # We can't get the serialized bytes length from ray.put
        return ObjectInfo(object_id=object_id)

    @implements(StorageBackend.delete)
    async def delete(self, object_id):
        ray.internal.free(object_id)

    @implements(StorageBackend.object_info)
    async def object_info(self, object_id) -> ObjectInfo:
        # The performance of obtaining the object size is poor.
        return ObjectInfo(object_id=object_id)

    @implements(StorageBackend.open_writer)
    async def open_writer(self, size=None) -> StorageFileObject:
        new_id = ray.ObjectRef.from_random()
        ray_writer = RayFileObject(new_id, mode='w')
        return StorageFileObject(ray_writer, object_id=new_id)

    @implements(StorageBackend.open_reader)
    async def open_reader(self, object_id) -> StorageFileObject:
        ray_reader = RayFileObject(object_id, mode='r')
        return StorageFileObject(ray_reader, object_id=object_id)

    @implements(StorageBackend.list)
    async def list(self) -> List:
        raise NotImplementedError("Ray storage does not support list")
