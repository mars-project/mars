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


class MarsError(RuntimeError):
    """
    """
    def __init__(self, msg=None):
        self._err_data = msg
        msg = repr(msg) if msg is not None else None
        super(MarsError, self).__init__(msg)

    def __str__(self):
        if hasattr(self, 'message'):
            message = self.message
        else:
            message = self.args[0]  # py3
        return message or ''


class StartArgumentError(MarsError):
    pass


class StorageExhausted(MarsError):
    pass


class WorkerDead(MarsError):
    pass


class DependencyMissing(MarsError):
    pass


class StoreFull(MarsError):
    pass


class StoreKeyExists(MarsError):
    pass


class ChecksumMismatch(MarsError):
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


class NoDataToSpill(MarsError):
    pass


class PinChunkFailed(MarsError):
    pass


class ObjectNotInPlasma(MarsError):
    pass


class WorkerProcessStopped(MarsError):
    pass
