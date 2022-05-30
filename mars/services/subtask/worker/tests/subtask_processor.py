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
from collections import defaultdict
from typing import Any, Dict

from .....core import OperandType
from .....tests.core import _check_args, ObjectCheckMixin
from ...utils import iter_output_data
from ...worker.processor import SubtaskProcessor


class CheckStorageAPI:
    def __init__(self, storage_api):
        self._storage_api = storage_api
        self._put_data_keys = set()

    def __getattr__(self, item):
        return getattr(self._storage_api, item)

    @property
    def put(self):
        owner = self
        put = self._storage_api.put

        class _PutWrapper:
            def delay(self, data_key: str, obj: object, level=None):
                if data_key in owner._put_data_keys:
                    raise Exception(f"Duplicate data put: {data_key}, obj: {obj}")
                else:
                    owner._put_data_keys.add(data_key)
                    return put.delay(data_key, obj, level)

            def __getattr__(self, item):
                return getattr(put, item)

        return _PutWrapper()


class CheckedSubtaskProcessor(ObjectCheckMixin, SubtaskProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        check_options = dict()
        if self.subtask.extra_config:
            kwargs = self.subtask.extra_config.copy()
        else:
            kwargs = dict()
        self._operand_executors = operand_executors = kwargs.pop(
            "operand_executors", dict()
        )
        for op, executor in operand_executors.items():
            op.register_executor(executor)
        for key in _check_args:
            check_options[key] = kwargs.get(key, True)
        self._check_options = check_options
        self._check_keys = kwargs.get("check_keys")
        self._storage_api = CheckStorageAPI(self._storage_api)

    def _execute_operand(self, ctx: Dict[str, Any], op: OperandType):
        super()._execute_operand(ctx, op)
        if self._check_options.get("check_all", True):
            for out in op.outputs:
                if out not in self._chunk_graph.result_chunks:
                    continue
                if self._check_keys and out.key not in self._check_keys:
                    continue
                if out.key not in ctx and any(
                    k[0] == out.key for k in ctx if isinstance(k, tuple)
                ):
                    # both shuffle mapper and reducer
                    continue
                self.assert_object_consistent(out, ctx[out.key])

    async def _store_data(self, *args, **kwargs):
        # `iter_output_data` must ensure values order since we only return values.
        shuffle_output = {
            key: data
            for key, data, is_shuffle in iter_output_data(
                self.subtask.chunk_graph, self._datastore
            )
            if is_shuffle
        }
        # assert output keys order consistent
        if shuffle_output:
            mapper_reducer_indices = defaultdict(list)
            for chunk_key, reducer_index in shuffle_output.keys():
                mapper_reducer_indices[chunk_key].append(reducer_index)
            for reducer_indices in mapper_reducer_indices.values():
                assert sorted(reducer_indices) == list(reducer_indices), (
                    reducer_indices,
                    sorted(reducer_indices),
                )
        return await super()._store_data(*args, **kwargs)

    async def done(self):
        await super().done()
        for op in self._operand_executors:
            try:
                op.unregister_executor()
            except KeyError:
                pass
