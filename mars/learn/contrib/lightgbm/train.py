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
import pickle
import random
from collections import defaultdict

import numpy as np

from .... import opcodes
from ....context import get_context, RunningMode
from ....serialize import DictField, Int32Field, KeyField, ListField, StringField, ValueType
from ...operands import LearnMergeDictOperand, OutputType
from ...utils import concat_chunks
from .align import align_inputs
from .core import LGBMModelType, get_model_cls_from_type

logger = logging.getLogger(__name__)


class LGBMTrain(LearnMergeDictOperand):
    _op_type_ = opcodes.LGBM_TRAIN

    _model_type = Int32Field('model_type', on_serialize=lambda x: x.value,
                             on_deserialize=LGBMModelType)
    _data = KeyField('data')
    _label = KeyField('label')
    _sample_weight = KeyField('sample_weight')
    _params = DictField('params', key_type=ValueType.string)
    _kwds = DictField('kwds', key_type=ValueType.string)

    _lgbm_endpoints = ListField('lgbm_endpoints', ValueType.string)
    _lgbm_port = Int32Field('lgbm_port')
    _tree_learner = StringField('tree_learner')
    _timeout = Int32Field('timeout')

    def __init__(self, model_type=None, data=None, label=None, sample_weight=None,
                 params=None, kwds=None, lgbm_endpoints=None, lgbm_port=None,
                 tree_learner=None, timeout=None, merge=False, output_types=None, **kw):
        super().__init__(_model_type=model_type, _data=data, _label=label,
                         _sample_weight=sample_weight, _params=params, _kwds=kwds,
                         _lgbm_endpoints=lgbm_endpoints, _lgbm_port=lgbm_port,
                         _tree_learner=tree_learner, _timeout=timeout, _merge=merge,
                         _output_types=output_types, **kw)
        if self._output_types is None:
            self._output_types = [OutputType.object]

    @property
    def model_type(self) -> LGBMModelType:
        return self._model_type

    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label

    @property
    def sample_weight(self):
        return self._sample_weight

    @property
    def params(self) -> dict:
        return self._params

    @property
    def kwds(self) -> dict:
        return self._kwds

    @property
    def lgbm_endpoints(self) -> list:
        return self._lgbm_endpoints

    @property
    def lgbm_port(self) -> int:
        return self._lgbm_port

    @property
    def timeout(self) -> int:
        return self._timeout

    @property
    def tree_learner(self) -> str:
        return self._tree_learner

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        it = iter(inputs)
        self._data = next(it)
        if self._label is not None:
            self._label = next(it)
        if self._sample_weight is not None:
            self._sample_weight = next(it)

    def __call__(self):
        inputs = [self._data]
        if self._label is not None:
            inputs.append(self._label)
        if self._sample_weight is not None:
            inputs.append(self._sample_weight)
        return self.new_tileable(inputs)

    @staticmethod
    def _get_data_chunks_workers(ctx, data):
        # data_chunk.inputs is concat, and concat's input is the co-allocated chunks
        metas = ctx.get_chunk_metas([c.key for c in data.chunks])
        return [m.workers[0] for m in metas]

    @staticmethod
    def _concat_chunks_by_worker(chunks, chunk_workers):
        worker_to_chunks = defaultdict(list)
        for chunk, worker in zip(chunks, chunk_workers):
            worker_to_chunks[worker].append(chunk)
        worker_to_concat = dict()
        for worker, chunks in worker_to_chunks.items():
            worker_to_concat[worker] = concat_chunks(chunks)
        return worker_to_concat

    @staticmethod
    def _build_lgbm_endpoints(workers, base_port):
        worker_to_endpoint = dict()
        workers = set(workers)
        base_port = base_port or random.randint(10000, 65535 - len(workers))
        for idx, worker in enumerate(workers):
            worker_to_endpoint[worker] = '%s:%d' % (worker.split(':', 1)[0], base_port + idx)
        return worker_to_endpoint

    @classmethod
    def tile(cls, op: "LGBMTrain"):
        ctx = get_context()
        if ctx.running_mode != RunningMode.distributed:
            assert all(len(inp.chunks) == 1 for inp in op.inputs)

            chunk_op = op.copy().reset_key()
            out_chunk = chunk_op.new_chunk([inp.chunks[0] for inp in op.inputs],
                                           shape=(1,), index=(0,))
            new_op = op.copy()
            return new_op.new_tileables(op.inputs, chunks=[out_chunk], nsplits=((1,),))
        else:
            data = op.data
            label = op.label
            weight = op.sample_weight

            workers = cls._get_data_chunks_workers(ctx, data)
            worker_to_endpoint = cls._build_lgbm_endpoints(workers, op.lgbm_port)
            worker_endpoints = list(worker_to_endpoint.values())
            worker_to_data = cls._concat_chunks_by_worker(data.chunks, workers)
            if label is not None:
                worker_to_label = cls._concat_chunks_by_worker(label.chunks, workers)
            else:
                worker_to_label = dict()
            if weight is not None:
                worker_to_weight = cls._concat_chunks_by_worker(weight.chunks, workers)
            else:
                worker_to_weight = dict()

            out_chunks = []
            for worker in workers:
                chunk_op = op.copy().reset_key()
                chunk_op._data = data_chunk = worker_to_data[worker]
                chunk_op._label = label_chunk = worker_to_label.get(worker)
                chunk_op._sample_weight = weight_chunk = worker_to_weight.get(worker)

                chunk_op._expect_worker = worker
                chunk_op._lgbm_endpoints = worker_endpoints
                chunk_op._lgbm_port = int(worker_to_endpoint[worker].rsplit(':', 1)[-1])

                input_chunks = [data_chunk]
                if label_chunk is not None:
                    input_chunks.append(label_chunk)
                if weight_chunk is not None:
                    input_chunks.append(weight_chunk)
                out_chunk = chunk_op.new_chunk(input_chunks, shape=(np.nan,),
                                               index=data_chunk.index[:1])
                out_chunks.append(out_chunk)

            new_op = op.copy()
            return new_op.new_tileables(op.inputs, chunks=out_chunks,
                                        nsplits=((np.nan for _ in out_chunks),))

    @classmethod
    def execute(cls, ctx, op: "LGBMTrain"):
        if op.merge:
            return super().execute(ctx, op)

        from lightgbm.basic import _safe_call, _LIB

        data_val = ctx[op.data.key]
        label_val = ctx[op.label.key]
        if op.sample_weight is not None:
            sample_weight_val = ctx[op.sample_weight.key]
        else:
            sample_weight_val = None

        params = op.params.copy()
        if ctx.running_mode == RunningMode.distributed:
            params['machines'] = op.lgbm_endpoints
            params['time_out'] = op.timeout
            params['num_machines'] = len(op.lgbm_endpoints)
            params['local_listen_port'] = op.lgbm_port

            if (op.tree_learner or '').lower() not in {'data', 'feature', 'voting'}:
                logger.warning('Parameter tree_learner not set or set to incorrect value %s, '
                               'using "data" as default' % op.tree_learner)
                params['tree_learner'] = 'data'
            else:
                params['tree_learner'] = op.tree_learner

        try:
            model_cls = get_model_cls_from_type(op.model_type)
            model = model_cls(**params)
            model.fit(data_val, label_val, sample_weight=sample_weight_val, **op.kwds)

            if hasattr(label_val, 'dtype'):
                model.set_params(out_dtype_=label_val.dtype)
            else:
                model.set_params(out_dtype_=label_val.dtypes[0])

            ctx[op.outputs[0].key] = pickle.dumps(model)
        finally:
            _safe_call(_LIB.LGBM_NetworkFree())


def train(params, data, label=None, sample_weight=None, model_type=None, **kwargs):
    model_type = model_type or LGBMModelType.CLASSIFIER

    evals_result = kwargs.pop('evals_result', dict())
    session = kwargs.pop('session', None)
    run_kwargs = kwargs.pop('run_kwargs', dict())
    timeout = kwargs.pop('timeout', 120)
    base_port = kwargs.pop('base_port', None)

    aligned = align_inputs(data, label, sample_weight).execute(session=session)
    if len(aligned) == 2:
        data, label = aligned
    else:
        data, label, sample_weight = aligned

    op = LGBMTrain(params=params, data=data, label=label, sample_weight=sample_weight,
                   model_type=model_type, timeout=timeout, lgbm_port=base_port, kwds=kwargs)
    ret = op().execute(session=session, **run_kwargs).fetch(session=session)

    bst = pickle.loads(ret)
    evals_result.update(bst.evals_result_ or {})
    return bst
