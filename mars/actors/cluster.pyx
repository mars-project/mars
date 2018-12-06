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


cdef class ClusterInfo:

    def __init__(self, bint standalone, int n_process, str address=None,
                 str serialization='pickle'):
        if address and ':' not in address:
            raise ValueError('address must contain port')

        self.standalone = standalone
        self.n_process = n_process
        self.address = address
        self.serialization = serialization

        if self.address:
            self.location, port = self.address.split(':', 1)
            self.port = int(port)
