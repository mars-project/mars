#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import hashlib


cdef class Distributor(object):
    def __init__(self, int n_process):
        self.n_process = n_process

    cpdef int distribute(self, object uid):
        if not isinstance(uid, bytes):
            uid = str(uid).encode('utf-8')

        return int(hashlib.md5(uid).hexdigest(), 16) % self.n_process
