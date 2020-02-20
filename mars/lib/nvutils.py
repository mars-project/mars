# -*- coding: utf-8 -*-
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

import logging
import os
import uuid
from collections import namedtuple
from ctypes import c_char, c_char_p, c_int, c_uint, c_ulonglong, byref,\
    create_string_buffer, Structure, POINTER, CDLL

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
NVML_TEMPERATURE_GPU = 0

NVML_DRIVER_NOT_LOADED = 9


class _CUuuid_t(Structure):
    _fields_ = [
        ('bytes', c_char * 16)
    ]
class _nvmlUtilization_t(Structure):
    _fields_ = [
        ('gpu', c_uint),
        ('memory', c_uint),
    ]

class _struct_nvmlDevice_t(Structure):
    pass  # opaque handle
_nvmlDevice_t = POINTER(_struct_nvmlDevice_t)

class _nvmlBAR1Memory_t(Structure):
    _fields_ = [
        ('total', c_ulonglong),
        ('free', c_ulonglong),
        ('used', c_ulonglong),
    ]


def _load_nv_library(*libnames):
    for lib in libnames:
        try:
            return CDLL(lib)
        except OSError:
            continue


_cuda_lib = _nvml_lib = None

_cu_device_info = namedtuple('_cu_device_info', 'index uuid name multiprocessors cuda_cores threads')
_nvml_driver_info = namedtuple('_nvml_driver_info', 'driver_version cuda_version')
_nvml_device_status = namedtuple(
    '_nvml_device_status', 'gpu_util mem_util temperature fb_total_mem fb_used_mem fb_free_mem')

_init_pid = None
_gpu_count = None
_driver_info = None
_device_infos = dict()

_no_device_warned = False


class NVError(Exception):
    def __init__(self, msg, *args, errno=None):
        self._errno = errno
        super().__init__(msg or 'Unknown error', *args)

    def __str__(self):
        return '(%s) %s' % (self._errno, super().__str__())

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
        raise NVDeviceAPIError(_error_str.value.decode(), errno=result)


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

    _cuda_lib = _load_nv_library('libcuda.so', 'libcuda.dylib', 'cuda.dll')

    if _cuda_lib is None:
        return
    try:
        _cu_check_error(_cuda_lib.cuInit(0))
    except NVDeviceAPIError as ex:
        if ex.errno == CU_NO_CUDA_CAPABLE_DEVICE_DETECTED:
            _cuda_lib = None
            if not _no_device_warned:
                logger.warning('No CUDA device detected')
                _no_device_warned = True
        else:
            logger.exception('Failed to initialize libcuda.')
        return


def _init_nvml():
    global _nvml_lib, _no_device_warned
    if _init_pid == os.getpid():
        return

    _nvml_lib = _load_nv_library('libnvidia-ml.so', 'libnvidia-ml.dylib', 'nvml.dll')

    if _nvml_lib is None:
        return
    try:
        _nvml_check_error(_nvml_lib.nvmlInit_v2())
    except NVMLAPIError as ex:
        if ex.errno == NVML_DRIVER_NOT_LOADED:
            _nvml_lib = None
            if not _no_device_warned:
                logger.warning('Failed to load libnvidia-ml: %s, no CUDA device will be enabled', ex.message)
                _no_device_warned = True
        else:
            logger.exception('Failed to initialize libnvidia-ml.')
        return


def _init():
    global _init_pid

    _init_cp()
    _init_nvml()

    if _nvml_lib is not None and _cuda_lib is not None:
        _init_pid = os.getpid()


def get_device_count():
    global _gpu_count

    if _gpu_count is not None:
        return _gpu_count

    _init_nvml()
    if _nvml_lib is None:
        return None

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        devices = os.environ['CUDA_VISIBLE_DEVICES'].strip()
        if not devices:
            _gpu_count = 0
        else:
            _gpu_count = len(devices.split(','))
    else:
        n_gpus = c_uint()
        _cu_check_error(_nvml_lib.nvmlDeviceGetCount(byref(n_gpus)))
        _gpu_count = n_gpus.value
    return _gpu_count


def get_driver_info():
    global _driver_info

    _init_nvml()
    if _nvml_lib is None:
        return None
    if _driver_info is not None:
        return _driver_info

    version_buf = create_string_buffer(100)
    cuda_version = c_uint()

    _nvml_check_error(_nvml_lib.nvmlSystemGetDriverVersion(version_buf, len(version_buf)))
    _nvml_check_error(_nvml_lib.nvmlSystemGetCudaDriverVersion(byref(cuda_version)))

    _driver_info = _nvml_driver_info(
        driver_version=version_buf.value.decode(),
        cuda_version='%d.%d' % (cuda_version.value // 1000, cuda_version.value % 1000)
    )
    return _driver_info


def get_device_info(dev_index):
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
    _cu_check_error(_cuda_lib.cuDeviceComputeCapability(
        byref(cc_major), byref(cc_minor), device))
    _cu_check_error(_cuda_lib.cuDeviceGetAttribute(
        byref(cores), CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device))
    _cu_check_error(_cuda_lib.cuDeviceGetAttribute(
        byref(threads_per_core), CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device))

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        real_dev_index = [int(s) for s in os.environ['CUDA_VISIBLE_DEVICES'].split(',')][dev_index]
    else:
        real_dev_index = dev_index

    info = _device_infos[dev_index] = _cu_device_info(
        index=real_dev_index,
        uuid=uuid.UUID(bytes=uuid_t.bytes),
        name=name_buf.value.decode(),
        multiprocessors=cores.value,
        cuda_cores=cores.value * _cu_get_processor_cores(cc_major.value, cc_minor.value),
        threads=cores.value * threads_per_core.value,
    )
    return info


def get_device_status(dev_index):
    _init()
    if _init_pid is None:
        return None

    device = _nvmlDevice_t()
    utils = _nvmlUtilization_t()
    temperature = c_uint()
    memory_info = _nvmlBAR1Memory_t()

    dev_uuid = get_device_info(dev_index).uuid

    uuid_str = ('GPU-' + str(dev_uuid)).encode()

    _nvml_check_error(_nvml_lib.nvmlDeviceGetHandleByUUID(uuid_str, byref(device)))
    _nvml_check_error(_nvml_lib.nvmlDeviceGetUtilizationRates(device, byref(utils)))
    _nvml_check_error(_nvml_lib.nvmlDeviceGetTemperature(
        device, NVML_TEMPERATURE_GPU, byref(temperature)))
    _nvml_check_error(_nvml_lib.nvmlDeviceGetBAR1MemoryInfo(device, byref(memory_info)))

    return _nvml_device_status(
        gpu_util=utils.gpu,
        mem_util=utils.memory,
        temperature=temperature.value,
        fb_total_mem=memory_info.total,
        fb_free_mem=memory_info.free,
        fb_used_mem=memory_info.used,
    )
