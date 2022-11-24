# -*- coding: utf-8 -*-
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

import dataclasses
import logging
import os
import subprocess
import sys
import uuid
from collections import namedtuple
from ctypes import (
    c_char,
    c_char_p,
    c_int,
    c_uint,
    c_ulonglong,
    byref,
    create_string_buffer,
    Structure,
    POINTER,
    CDLL,
)
from typing import List, Tuple, Optional, Union

from ..utils import parse_readable_size

logger = logging.getLogger(__name__)

# Some constants taken from cuda.h
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33
CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36

CU_NO_CUDA_CAPABLE_DEVICE_DETECTED = 100

# nvml constants
NVML_SUCCESS = 0
NVML_ERROR_UNINITIALIZED = 1
NVML_ERROR_INVALID_ARGUMENT = 2
NVML_ERROR_NOT_SUPPORTED = 3
NVML_ERROR_NO_PERMISSION = 4
NVML_ERROR_ALREADY_INITIALIZED = 5
NVML_ERROR_NOT_FOUND = 6
NVML_ERROR_INSUFFICIENT_SIZE = 7
NVML_ERROR_INSUFFICIENT_POWER = 8
NVML_ERROR_DRIVER_NOT_LOADED = 9
NVML_ERROR_TIMEOUT = 10
NVML_ERROR_IRQ_ISSUE = 11
NVML_ERROR_LIBRARY_NOT_FOUND = 12
NVML_ERROR_FUNCTION_NOT_FOUND = 13
NVML_ERROR_CORRUPTED_INFOROM = 14
NVML_ERROR_GPU_IS_LOST = 15
NVML_ERROR_RESET_REQUIRED = 16
NVML_ERROR_OPERATING_SYSTEM = 17
NVML_ERROR_LIB_RM_VERSION_MISMATCH = 18
NVML_ERROR_IN_USE = 19
NVML_ERROR_MEMORY = 20
NVML_ERROR_NO_DATA = 21
NVML_ERROR_VGPU_ECC_NOT_SUPPORTED = 22
NVML_ERROR_INSUFFICIENT_RESOURCES = 23
NVML_ERROR_FREQ_NOT_SUPPORTED = 24
NVML_ERROR_UNKNOWN = 999
NVML_TEMPERATURE_GPU = 0
NVML_DRIVER_NOT_LOADED = 9
NVML_DEVICE_UUID_V2_BUFFER_SIZE = 96
NVML_VALUE_NOT_AVAILABLE_ulonglong = c_ulonglong(-1)
NVML_DEVICE_MIG_DISABLE = 0x0
NVML_DEVICE_MIG_ENABLE = 0x1


class _CUuuid_t(Structure):
    _fields_ = [("bytes", c_char * 16)]


class _nvmlUtilization_t(Structure):
    _fields_ = [
        ("gpu", c_uint),
        ("memory", c_uint),
    ]


class _struct_nvmlDevice_t(Structure):
    pass  # opaque handle


_nvmlDevice_t = POINTER(_struct_nvmlDevice_t)


class _nvmlBAR1Memory_t(Structure):
    _fields_ = [
        ("total", c_ulonglong),
        ("free", c_ulonglong),
        ("used", c_ulonglong),
    ]


class _nvmlProcessInfo_t(Structure):
    _fields_ = [
        ("pid", c_uint),
        ("usedGpuMemory", c_ulonglong),
        ("gpuInstanceId", c_uint),
        ("computeInstanceId", c_uint),
    ]


## Alternative object
# Allows the object to be printed
# Allows mismatched types to be assigned
#  - like None when the Structure variant requires c_uint
class nvmlFriendlyObject:
    def __init__(self, dictionary):
        for x in dictionary:
            setattr(self, x, dictionary[x])

    def __str__(self):
        return self.__dict__.__str__()


def nvmlStructToFriendlyObject(struct):
    d = {}
    for x in struct._fields_:
        key = x[0]
        value = getattr(struct, key)
        # only need to convert from bytes if bytes, no need to check python version.
        d[key] = value.decode() if isinstance(value, bytes) else value
    obj = nvmlFriendlyObject(d)
    return obj


@dataclasses.dataclass
class CudaDeviceInfo:
    uuid: bytes = None
    device_index: int = None
    mig_index: int = None


@dataclasses.dataclass
class CudaContext:
    has_context: bool
    device_info: CudaDeviceInfo = None


_is_windows: bool = sys.platform.startswith("win")
_is_wsl: bool = "WSL_DISTRO_NAME" in os.environ


def _load_nv_library(*libnames):
    for lib in libnames:
        try:
            return CDLL(lib)
        except OSError:
            continue


_cuda_lib = _nvml_lib = None

_cu_device_info = namedtuple(
    "_cu_device_info", "index uuid name multiprocessors cuda_cores threads"
)
_nvml_driver_info = namedtuple("_nvml_driver_info", "driver_version cuda_version")
_nvml_device_status = namedtuple(
    "_nvml_device_status",
    "gpu_util mem_util temperature fb_total_mem fb_used_mem fb_free_mem",
)

_init_pid = None
_gpu_count = None
_driver_info = None
_device_infos = dict()

_no_device_warned = False


class NVError(Exception):
    def __init__(self, msg, *args, errno=None):
        self._errno = errno
        super().__init__(msg or "Unknown error", *args)

    def __str__(self):
        return f"({self._errno}) {super().__str__()}"

    @property
    def errno(self):
        return self._errno

    @property
    def message(self):
        return super().__str__()


class NVDeviceAPIError(NVError):
    pass


class NVMLAPIError(NVError):
    pass


def _cu_check_error(result):
    if result != CUDA_SUCCESS:
        _error_str = c_char_p()
        _cuda_lib.cuGetErrorString(result, byref(_error_str))
        err_value = _error_str.value.decode() if _error_str.value is not None else None
        raise NVDeviceAPIError(err_value, errno=result)


_nvmlErrorString = None


def _nvml_check_error(result):
    global _nvmlErrorString
    if _nvmlErrorString is None:
        _nvmlErrorString = _nvml_lib.nvmlErrorString
        _nvmlErrorString.restype = c_char_p

    if result != NVML_SUCCESS:
        _error_str = _nvmlErrorString(result)
        raise NVMLAPIError(_error_str.decode(), errno=result)


_cu_process_var_to_cores = {
    (1, 0): 8,
    (1, 1): 8,
    (1, 2): 8,
    (1, 3): 8,
    (2, 0): 32,
    (2, 1): 48,
}


def _cu_get_processor_cores(major, minor):
    return _cu_process_var_to_cores.get((major, minor), 192)


def _init_cp():
    global _cuda_lib, _no_device_warned
    if _init_pid == os.getpid():
        return

    libcuda_paths = ["libcuda.so", "libcuda.dylib", "cuda.dll", "nvcuda.dll"]
    if _is_wsl:
        libcuda_paths = ["/usr/lib/wsl/lib/libcuda.so"] + libcuda_paths
    _cuda_lib = _load_nv_library(*libcuda_paths)

    if _cuda_lib is None:
        return
    try:
        _cu_check_error(_cuda_lib.cuInit(0))
    except NVDeviceAPIError as ex:
        if ex.errno == CU_NO_CUDA_CAPABLE_DEVICE_DETECTED:
            _cuda_lib = None
            if not _no_device_warned:
                logger.warning("No CUDA device detected")
                _no_device_warned = True
        else:
            logger.exception("Failed to initialize libcuda.")
        return


def _init_nvml():
    global _nvml_lib, _no_device_warned
    if _init_pid == os.getpid():
        return

    nvml_paths = [
        "libnvidia-ml.so",
        "libnvidia-ml.so.1",
        "libnvidia-ml.dylib",
        "nvml.dll",
    ]
    if _is_windows:
        nvml_paths.append(
            os.path.join(
                os.getenv("ProgramFiles", "C:/Program Files"),
                "NVIDIA Corporation/NVSMI/nvml.dll",
            )
        )
    if _is_wsl:
        nvml_paths = ["/usr/lib/wsl/lib/libnvidia-ml.so.1"] + nvml_paths
    _nvml_lib = _load_nv_library(*nvml_paths)

    if _nvml_lib is None:
        return
    try:
        _nvml_check_error(_nvml_lib.nvmlInit_v2())
    except NVMLAPIError as ex:
        if ex.errno == NVML_DRIVER_NOT_LOADED:
            _nvml_lib = None
            if not _no_device_warned:
                logger.warning(
                    "Failed to load libnvidia-ml: %s, no CUDA device will be enabled",
                    ex.message,
                )
                _no_device_warned = True
        else:
            logger.exception("Failed to initialize libnvidia-ml.")
        return


def _init():
    global _init_pid

    _init_cp()
    _init_nvml()

    if _nvml_lib is not None and _cuda_lib is not None:
        _init_pid = os.getpid()


def get_device_count() -> int:
    global _gpu_count

    if _gpu_count is not None:
        return _gpu_count

    _init_nvml()
    if _nvml_lib is None:
        return None

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        devices = os.environ["CUDA_VISIBLE_DEVICES"].strip()
        if not devices or devices == "-1":
            _gpu_count = 0
        else:
            _gpu_count = len(devices.split(","))
    else:
        n_gpus = c_uint()
        _cu_check_error(_nvml_lib.nvmlDeviceGetCount(byref(n_gpus)))
        _gpu_count = n_gpus.value
    return _gpu_count


def _get_all_device_count() -> int:
    _init_nvml()
    if _nvml_lib is None:
        return None

    n_gpus = c_uint()
    _cu_check_error(_nvml_lib.nvmlDeviceGetCount(byref(n_gpus)))
    return n_gpus.value


def get_driver_info() -> _nvml_driver_info:
    global _driver_info

    _init_nvml()
    if _nvml_lib is None:
        return None
    if _driver_info is not None:
        return _driver_info

    version_buf = create_string_buffer(100)
    cuda_version = c_uint()

    _nvml_check_error(
        _nvml_lib.nvmlSystemGetDriverVersion(version_buf, len(version_buf))
    )
    _nvml_check_error(_nvml_lib.nvmlSystemGetCudaDriverVersion(byref(cuda_version)))

    _driver_info = _nvml_driver_info(
        driver_version=version_buf.value.decode(),
        cuda_version=".".join(str(v) for v in divmod(cuda_version.value, 1000)),
    )
    return _driver_info


def get_device_info(dev_index: int) -> _cu_device_info:
    try:
        return _device_infos[dev_index]
    except KeyError:
        pass

    _init()
    if _init_pid is None:
        return None

    device = c_int()
    name_buf = create_string_buffer(100)
    uuid_t = _CUuuid_t()
    cc_major = c_int()
    cc_minor = c_int()
    cores = c_int()
    threads_per_core = c_int()

    _cu_check_error(_cuda_lib.cuDeviceGet(byref(device), c_int(dev_index)))
    _cu_check_error(_cuda_lib.cuDeviceGetName(name_buf, len(name_buf), device))
    _cu_check_error(_cuda_lib.cuDeviceGetUuid(byref(uuid_t), device))
    _cu_check_error(
        _cuda_lib.cuDeviceComputeCapability(byref(cc_major), byref(cc_minor), device)
    )
    _cu_check_error(
        _cuda_lib.cuDeviceGetAttribute(
            byref(cores), CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device
        )
    )
    _cu_check_error(
        _cuda_lib.cuDeviceGetAttribute(
            byref(threads_per_core),
            CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
            device,
        )
    )

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        real_dev_index = [
            int(s) for s in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        ][dev_index]
    else:
        real_dev_index = dev_index

    info = _device_infos[dev_index] = _cu_device_info(
        index=real_dev_index,
        uuid=uuid.UUID(bytes=uuid_t.bytes),
        name=name_buf.value.decode(),
        multiprocessors=cores.value,
        cuda_cores=cores.value
        * _cu_get_processor_cores(cc_major.value, cc_minor.value),
        threads=cores.value * threads_per_core.value,
    )
    return info


def get_device_status(dev_index: int) -> _nvml_device_status:
    _init()
    if _init_pid is None:
        return None

    c_device = _nvmlDevice_t()
    c_utils = _nvmlUtilization_t()
    c_temperature = c_uint()
    c_memory_info = _nvmlBAR1Memory_t()

    dev_uuid = get_device_info(dev_index).uuid

    uuid_str = ("GPU-" + str(dev_uuid)).encode()

    if not _is_wsl:
        _nvml_check_error(
            _nvml_lib.nvmlDeviceGetHandleByUUID(uuid_str, byref(c_device))
        )

        _nvml_check_error(
            _nvml_lib.nvmlDeviceGetUtilizationRates(c_device, byref(c_utils))
        )
        gpu_util = c_utils.gpu
        mem_util = c_utils.memory

        _nvml_check_error(
            _nvml_lib.nvmlDeviceGetTemperature(
                c_device, NVML_TEMPERATURE_GPU, byref(c_temperature)
            )
        )
        temperature = c_temperature.value

        _nvml_check_error(
            _nvml_lib.nvmlDeviceGetMemoryInfo(c_device, byref(c_memory_info))
        )
        fb_total_mem = c_memory_info.total
        fb_free_mem = c_memory_info.free
        fb_used_mem = c_memory_info.used
    else:
        import defusedxml

        proc = subprocess.Popen(
            ["/usr/lib/wsl/lib/nvidia-smi", "-q", f"--id={dev_index}", "-x"],
            stdout=subprocess.PIPE,
        )
        proc.wait()
        xml_result = defusedxml.ElementTree.fromstring(proc.stdout.read())
        gpu_node = xml_result.find("gpu")

        fb_node = gpu_node.find("fb_memory_usage")
        fb_total_mem = int(parse_readable_size(fb_node.find("total").text)[0])
        fb_free_mem = int(parse_readable_size(fb_node.find("free").text)[0])
        fb_used_mem = int(parse_readable_size(fb_node.find("used").text)[0])

        util_node = gpu_node.find("utilization")
        if util_node.find("gpu_util").text == "N/A":
            gpu_util = 0
        else:
            gpu_util = int(util_node.find("gpu_util"))
        if util_node.find("memory_util").text == "N/A":
            mem_util = 0
        else:
            mem_util = int(util_node.find("memory_util"))

        temperature = int(gpu_node.find("temperature").find("gpu_temp").text[:-1])

    return _nvml_device_status(
        gpu_util=gpu_util,
        mem_util=mem_util,
        temperature=temperature,
        fb_total_mem=fb_total_mem,
        fb_free_mem=fb_free_mem,
        fb_used_mem=fb_used_mem,
    )


def get_handle_by_index(index: int) -> _nvmlDevice_t:
    _init_nvml()
    if _nvml_lib is None:
        return None

    c_index = c_int(index)
    device = _nvmlDevice_t()
    _nvml_check_error(_nvml_lib.nvmlDeviceGetHandleByIndex_v2(c_index, byref(device)))
    return device


def get_handle_by_uuid(uuid: bytes) -> _nvmlDevice_t:
    _init_nvml()
    if _nvml_lib is None:
        return None

    c_uuid = c_char_p(uuid)
    device = _nvmlDevice_t()
    _nvml_check_error(_nvml_lib.nvmlDeviceGetHandleByUUID(c_uuid, byref(device)))
    return device


def get_mig_mode(device: _nvmlDevice_t) -> Tuple[int, int]:
    _init_nvml()
    if _nvml_lib is None:
        return None

    c_current_mode, c_pending_mode = c_uint(), c_uint()
    _nvml_check_error(
        _nvml_lib.nvmlDeviceGetMigMode(
            device, byref(c_current_mode), byref(c_pending_mode)
        )
    )
    return c_current_mode.value, c_pending_mode.value


def get_max_mig_device_count(device: _nvmlDevice_t) -> int:
    _init_nvml()
    if _nvml_lib is None:
        return None

    c_count = c_uint()
    _nvml_check_error(_nvml_lib.nvmlDeviceGetMaxMigDeviceCount(device, byref(c_count)))
    return c_count.value


def get_mig_device_handle_by_index(device: _nvmlDevice_t, index: int) -> _nvmlDevice_t:
    _init_nvml()
    if _nvml_lib is None:
        return None

    c_index = c_uint(index)
    mig_device = _nvmlDevice_t()
    _nvml_check_error(
        _nvml_lib.nvmlDeviceGetMigDeviceHandleByIndex(
            device, c_index, byref(mig_device)
        )
    )
    return mig_device


def get_index(handle: _nvmlDevice_t) -> int:
    _init_nvml()
    if _nvml_lib is None:
        return None

    c_index = c_uint()
    _nvml_check_error(_nvml_lib.nvmlDeviceGetIndex(handle, byref(c_index)))
    return c_index.value


def get_uuid(handle: _nvmlDevice_t) -> bytes:
    _init_nvml()
    if _nvml_lib is None:
        return None

    c_uuid = create_string_buffer(NVML_DEVICE_UUID_V2_BUFFER_SIZE)
    _nvml_check_error(
        _nvml_lib.nvmlDeviceGetUUID(
            handle, c_uuid, c_uint(NVML_DEVICE_UUID_V2_BUFFER_SIZE)
        )
    )
    return c_uuid.value


def get_index_and_uuid(device: Union[int, bytes, str]) -> CudaDeviceInfo:
    _init_nvml()
    if _nvml_lib is None:
        return None

    try:
        device_index = int(device)
        device_handle = get_handle_by_index(device_index)
        uuid = get_uuid(device_handle)
    except ValueError:
        uuid = device if isinstance(device, bytes) else device.encode()
        uuid_handle = get_handle_by_uuid(uuid)
        device_index = get_index(uuid_handle)
        uuid = get_uuid(uuid_handle)

    return CudaDeviceInfo(uuid=uuid, device_index=device_index)


def get_compute_running_processes(handle: _nvmlDevice_t) -> List[nvmlFriendlyObject]:
    _init_nvml()
    if _nvml_lib is None:
        return None

    c_count = c_uint(0)
    func = getattr(_nvml_lib, "nvmlDeviceGetComputeRunningProcesses_v3", None)
    if func is None:
        func = getattr(_nvml_lib, "nvmlDeviceGetComputeRunningProcesses_v2")
    ret = func(handle, byref(c_count), None)

    if ret == NVML_SUCCESS:
        # special case, no running processes
        return []
    elif ret == NVML_ERROR_INSUFFICIENT_SIZE:
        # typical case
        # oversize the array incase more processes are created
        c_count.value = c_count.value * 2 + 5
        proc_array = _nvmlProcessInfo_t * c_count.value
        c_procs = proc_array()

        _nvml_check_error(func(handle, byref(c_count), c_procs))

        procs = []
        for i in range(c_count.value):
            # use an alternative struct for this object
            obj = nvmlStructToFriendlyObject(c_procs[i])
            if obj.usedGpuMemory == NVML_VALUE_NOT_AVAILABLE_ulonglong.value:
                # special case for WDDM on Windows, see comment above
                obj.usedGpuMemory = None
            procs.append(obj)

        return procs
    else:
        # error case
        _nvml_check_error(ret)


def _running_process_matches(handle: _nvmlDevice_t) -> bool:
    """Check whether the current process is same as that of handle
    Parameters
    ----------
    handle : _nvmlDevice_t
        NVML handle to CUDA device
    Returns
    -------
    out : bool
        Whether the device handle has a CUDA context on the running process.
    """
    return any(os.getpid() == o.pid for o in get_compute_running_processes(handle))


def get_cuda_context() -> CudaContext:
    """Check whether the current process already has a CUDA context created."""

    _init()
    if _init_pid is None:
        return CudaContext(has_context=False)

    for index in range(_get_all_device_count()):
        handle = get_handle_by_index(index)
        try:
            mig_current_mode, mig_pending_mode = get_mig_mode(handle)
        except NVMLAPIError as e:
            if e.errno == NVML_ERROR_NOT_SUPPORTED:
                mig_current_mode = NVML_DEVICE_MIG_DISABLE
            else:
                raise
        if mig_current_mode == NVML_DEVICE_MIG_ENABLE:
            for mig_index in range(get_max_mig_device_count(handle)):
                try:
                    mig_handle = get_mig_device_handle_by_index(handle, mig_index)
                except NVMLAPIError as e:
                    if e.errno == NVML_ERROR_NOT_FOUND:
                        # No MIG device with that index
                        continue
                    else:
                        raise
                if _running_process_matches(mig_handle):
                    return CudaContext(
                        has_context=True,
                        device_info=CudaDeviceInfo(
                            uuid=get_uuid(handle),
                            device_index=index,
                            mig_index=mig_index,
                        ),
                    )
        else:
            if _running_process_matches(handle):
                return CudaContext(
                    has_context=True,
                    device_info=CudaDeviceInfo(
                        uuid=get_uuid(handle), device_index=index
                    ),
                )

    return CudaContext(has_context=False)
