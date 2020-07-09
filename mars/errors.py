#!/usr/bin/env python
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


class MarsError(RuntimeError):
    """
    """
    def __init__(self, msg=None):
        self._err_data = msg
        msg = repr(msg) if msg is not None else None
        super().__init__(msg)

    def __str__(self):
        return self.args[0] or ''


class StartArgumentError(MarsError):
    pass


class WorkerDead(MarsError):
    pass


class DependencyMissing(MarsError):
    pass


class StorageFull(MarsError):
    def __init__(self, msg=None, **kwargs):
        self._request_size = kwargs.pop('request_size', 0)
        self._capacity = kwargs.pop('total_size', 0)
        self._affected_keys = kwargs.pop('affected_keys', [])

        if self._request_size and self._capacity:
            msg = (msg or '') + ' request_size=%s, total_size=%s' \
                  % (self._request_size, self._capacity)
            msg = msg.strip()
        super().__init__(msg)

    @property
    def request_size(self):
        return self._request_size

    @property
    def capacity(self):
        return self._capacity

    @property
    def affected_keys(self):
        return self._affected_keys


class StorageDataExists(MarsError):
    pass


class ExecutionInterrupted(MarsError):
    pass


class ExecutionFailed(MarsError):
    pass


class ExecutionNotStopped(MarsError):
    pass


class ExecutionStateUnknown(MarsError):
    pass


class ResponseMalformed(MarsError):
    pass


class SpillNotConfigured(MarsError):
    pass


class GraphNotExists(MarsError):
    pass


class PromiseTimeout(MarsError):
    pass


class SpillSizeExceeded(MarsError):
    pass


class SerializationFailed(MarsError):
    def __init__(self, msg=None, **kwargs):
        obj = kwargs.pop('obj', None)
        if obj:
            msg = (msg or '') + ' type=%s repr=%s' % (type(obj), repr(obj))
        super().__init__(msg)


class NoDataToSpill(MarsError):
    pass


class PinDataKeyFailed(MarsError):
    pass


class ObjectNotInPlasma(MarsError):
    pass


class WorkerProcessStopped(MarsError):
    pass
