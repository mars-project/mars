from typing import Any, Dict, List, Tuple
from ..serialization import AioSerializer, AioDeserializer
from ..utils import lazy_import, implements
from .base import StorageBackend, StorageLevel, ObjectInfo
from .core import BufferWrappedFileObject, StorageFileObject

ray = lazy_import("ray")
import ray


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
        # This ray.get may block the main loop.
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


class RayStorage(StorageBackend):
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    @implements(StorageBackend.setup)
    async def setup(cls, **kwargs) -> Tuple[Dict, Dict]:
        ray.init(ignore_reinit_error=True)
        return dict(), dict()

    @staticmethod
    @implements(StorageBackend.teardown)
    async def teardown(**kwargs):
        pass

    @property
    @implements(StorageBackend.level)
    def level(self) -> StorageLevel:
        return StorageLevel.MEMORY

    @implements(StorageBackend.get)
    async def get(self, object_id, **kwargs) -> object:
        ray_file = RayFileObject(object_id, mode='r')

        async with StorageFileObject(ray_file, object_id) as f:
            deserializer = AioDeserializer(f)
            return await deserializer.run()

    @implements(StorageBackend.put)
    async def put(self, obj, importance=0) -> ObjectInfo:
        serializer = AioSerializer(obj)
        buffers = await serializer.run()
        buffer_size = sum(getattr(buf, 'nbytes', len(buf))
                          for buf in buffers)

        object_id = ray.ObjectRef.from_random()
        ray_file = RayFileObject(object_id, mode='w')
        async with StorageFileObject(ray_file, object_id) as f:
            for buffer in buffers:
                await f.write(buffer)

        return ObjectInfo(size=buffer_size, object_id=object_id)

    @implements(StorageBackend.delete)
    async def delete(self, object_id):
        ray.internal.free(object_id)

    @implements(StorageBackend.object_info)
    async def object_info(self, object_id) -> ObjectInfo:
        ray_file = RayFileObject(object_id, mode='r')
        async with StorageFileObject(ray_file, object_id=object_id) as f:
            buf = await f.read()
            return ObjectInfo(size=len(buf), object_id=object_id)

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
        return []
