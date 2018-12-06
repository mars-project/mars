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

import ctypes
import os
import sys


def _load_mkl_rt(lib_name):
    """
    Load certain MKL library
    """
    if sys.platform.startswith('win'):
        lib_path = os.path.join(sys.prefix, 'Library', 'bin', lib_name + '.dll')
    elif sys.platform == 'darwin':
        lib_path = os.path.join(sys.prefix, 'lib', 'lib' + lib_name + '.dylib')
    else:
        lib_path = os.path.join(sys.prefix, 'lib', 'lib' + lib_name + '.so')
    if not os.path.exists(lib_path):
        lib_path = None

    if lib_path:
        return ctypes.cdll.LoadLibrary(lib_path)


class MKLVersion(ctypes.Structure):
    _fields_ = [
        ('major', ctypes.c_int),
        ('minor', ctypes.c_int),
        ('update', ctypes.c_int),
        ('product_status', ctypes.c_char_p),
        ('build', ctypes.c_char_p),
        ('processor', ctypes.c_char_p),
        ('platform', ctypes.c_char_p),
    ]


mkl_free_buffers = None
mkl_get_version = None

mkl_rt = _load_mkl_rt('mkl_rt')
if mkl_rt:
    try:
        mkl_free_buffers = mkl_rt.mkl_free_buffers
        mkl_free_buffers.argtypes = []
        mkl_free_buffers.restype = None
    except AttributeError:
        pass

    try:
        _mkl_get_version = mkl_rt.mkl_get_version
        _mkl_get_version.argtypes = [ctypes.POINTER(MKLVersion)]
        _mkl_get_version.restype = None

        def mkl_get_version():
            version = MKLVersion()
            _mkl_get_version(version)
            return version
    except AttributeError:
        pass
