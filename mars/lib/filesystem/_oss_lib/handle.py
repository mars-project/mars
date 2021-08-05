from _pyio import IOBase
from _thread import allocate_lock as Lock
import io
from .common import *
import oss2


class OSSIOBase(IOBase):
	def __init__(self, path, mode):
		self._path = path
		self._bucket_name, self._key_name, self._access_key_id, self._access_key_secret, self._end_point = parse_osspath(self._path)
		self._bucket = self._get_bucket()
		self._current_pos = 0
		self._stream = None
		self._read_lock = Lock()
		self._size = None
		self._buffer = b""
		self._buffer_size = 1 * 1024
		self._mode = mode
	
	@property
	def mode(self):
		return self._mode
	
	def fileno(self) -> int:
		raise AttributeError()
	
	def _get_bucket(self):
		return oss2.Bucket(
			auth=oss2.Auth(access_key_id=self._access_key_id,
			               access_key_secret=self._access_key_secret),
			endpoint=self._end_point,
			bucket_name=self._bucket_name,
			)
	
	def _get_size(self):
		if self._size is None:
			self._size = int(oss_stat(self._path)["size"])
		return self._size
	
	def seek(self, pos, whence=0):
		if whence == 0:
			if pos < 0:
				raise OSError("Invalid argument")
			self._current_pos = pos
		elif whence == 2:
			self._current_pos = self._get_size() + pos
		elif whence == 1:
			check_pos = self._current_pos + pos
			if check_pos < 0:
				raise OSError("Invalid argument")
			else:
				self._current_pos = self._current_pos + pos
		else:
			raise ValueError('Parameter "whence" should be 0 or 1 or 2')
		if pos > 0 and self._current_pos > self._get_size() - 1:
			self._current_pos = self._get_size()
		# print(f"current_pos = {self._current_pos}")
		return self._current_pos
	
	def seekable(self):
		return True
	
	def read(self, size=-1):
		"""
		Read and return up to size bytes, where size is an int.

		If the argument is omitted, None, or negative, reads and
		returns all data until EOF.

		If the argument is positive, multiple raw reads may be issued to satisfy
		the byte count (unless EOF is reached first).

		Returns an empty bytes array on EOF.
		"""
		if self._current_pos == self._get_size() or size == 0:
			return bytearray()
		elif size < 0:
			obj = self._bucket.get_object(self._key_name, byte_range=(self._current_pos, None))
			self._current_pos = self._get_size()
		else:
			obj = self._bucket.get_object(self._key_name, byte_range=(self._current_pos, self._current_pos + size - 1))
			self._current_pos = self._current_pos + size
		content = obj.read()
		return content
	
	def readline(self, size=-1):
		# For backwards compatibility, a (slowish) readline().
		def nreadahead():
			# Read to the beginning of the next line
			read_to = min(self._get_size() - 1, self._current_pos + self._buffer_size - 1)
			buffer = self._bucket.get_object(self._key_name,
			                                 byte_range=(self._current_pos, read_to)).read()
			if not buffer:
				return 1
			n = (buffer.find(b"\n") + 1) or len(buffer)
			if size >= 0:
				n = min(n, size)
			return n
		
		if size is None:
			size = -1
		else:
			try:
				size_index = size.__index__
			except AttributeError:
				raise TypeError(f"{size!r} is not an integer")
			else:
				size = size_index()
		res = bytearray()
		while size < 0 or len(res) < size:
			b = self.read(nreadahead())
			if not b:
				break
			res += b
			if res.endswith(b"\n"):
				break
		return bytes(res)
	
	def readable(self):
		return True
	
	def writable(self):
		return False
