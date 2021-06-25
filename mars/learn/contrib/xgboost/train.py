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

import pickle
from collections import OrderedDict, defaultdict

import numpy as np

from .... import opcodes as OperandDef
from ....core import OutputType
from ....core.context import get_context
from ....core.operand import MergeDictOperand
from ....serialization.serializables import FieldTypes, DictField, KeyField, ListField
from ....utils import ensure_own_data
from .start_tracker import StartTracker
from .dmatrix import ToDMatrix


def _on_serialize_evals(evals_val):
    if evals_val is None:
        return None
    return [list(x) for x in evals_val]


class XGBTrain(MergeDictOperand):
    _op_type_ = OperandDef.XGBOOST_TRAIN

    _params = DictField('params', key_type=FieldTypes.string)
    _dtrain = KeyField('dtrain')
    _evals = ListField('evals', on_serialize=_on_serialize_evals)
    _kwargs = DictField('kwargs', key_type=FieldTypes.string)
    _tracker = KeyField('tracker')

    def __init__(self, params=None, dtrain=None, evals=None, kwargs=None,
                 tracker=None, gpu=None, **kw):
        super().__init__(_params=params, _dtrain=dtrain, _evals=evals, _kwargs=kwargs,
                         _tracker=tracker, _gpu=gpu, **kw)
        if self.output_types is None:
            self.output_types = [OutputType.object]

    @property
    def params(self):
        return self._params

    @property
    def dtrain(self):
        return self._dtrain

    @property
    def evals(self):
        return self._evals

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def tracker(self):
        return self._tracker

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._dtrain = self._inputs[0]
        rest = self._inputs[1:]
        if self._tracker is not None:
            self._tracker = self._inputs[-1]
            rest = rest[:-1]
        if self._evals is not None:
            evals_dict = OrderedDict(self._evals)
            new_evals_dict = OrderedDict()
            for new_key, val in zip(rest, evals_dict.values()):
                new_evals_dict[new_key] = val
            self._evals = list(new_evals_dict.items())

    def __call__(self):
        inputs = [self._dtrain]
        if self._evals is not None:
            inputs.extend(e[0] for e in self._evals)
        return self.new_tileable(inputs)

    @staticmethod
    def _get_dmatrix_chunks_workers(ctx, dmatrix):
        # dmatrix_chunk.inputs is concat, and concat's input is the coallocated chunks
        metas = ctx.get_chunks_meta(
            [c.inputs[0].inputs[0].key for c in dmatrix.chunks], fields=['bands'])
        return [m['bands'][0][0] for m in metas]

    @staticmethod
    def _get_dmatrix_worker_to_chunk(dmatrix, workers, ctx):
        worker_to_chunk = dict()
        expect_workers = set(workers)
        workers = XGBTrain._get_dmatrix_chunks_workers(ctx, dmatrix)
        for w, c in zip(workers, dmatrix.chunks):
            if w in expect_workers:
                worker_to_chunk[w] = c
        return worker_to_chunk

    @classmethod
    def tile(cls, op):
        ctx = get_context()

        inp = op.inputs[0]
        in_chunks = inp.chunks
        workers = cls._get_dmatrix_chunks_workers(ctx, inp)
        n_chunk = len(in_chunks)
        tracker_chunk = StartTracker(n_workers=n_chunk, pure_depends=[True] * n_chunk)\
            .new_chunk(in_chunks, shape=())
        out_chunks = []
        worker_to_evals = defaultdict(list)
        if op.evals is not None:
            for dm, ev in op.evals:
                worker_to_chunk = cls._get_dmatrix_worker_to_chunk(dm, workers, ctx)
                for worker, chunk in worker_to_chunk.items():
                    worker_to_evals[worker].append((chunk, ev))
        for in_chunk, worker in zip(in_chunks, workers):
            chunk_op = op.copy().reset_key()
            chunk_op.expect_worker = worker
            chunk_op._tracker = tracker_chunk
            chunk_evals = list(worker_to_evals.get(worker, list()))
            chunk_op._evals = chunk_evals
            input_chunks = [in_chunk] + [pair[0] for pair in chunk_evals] + [tracker_chunk]
            out_chunk = chunk_op.new_chunk(input_chunks, shape=(np.nan,),
                                           index=in_chunk.index[:1])
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tileables(op.inputs, chunks=out_chunks,
                                    nsplits=((np.nan for _ in out_chunks),))

    @classmethod
    def execute(cls, ctx, op):
        if op.merge:
            return super().execute(ctx, op)

        from xgboost import train, rabit

        dtrain = ToDMatrix.get_xgb_dmatrix(
            ensure_own_data(ctx[op.dtrain.key]))
        evals = tuple()
        if op.evals is not None:
            eval_dmatrices = [ToDMatrix.get_xgb_dmatrix(
                ensure_own_data(ctx[t[0].key])) for t in op.evals]
            evals = tuple((m, ev[1]) for m, ev in zip(eval_dmatrices, op.evals))
        params = op.params

        if op.tracker is None:
            # non distributed
            local_history = dict()
            kwargs = dict() if op.kwargs is None else op.kwargs
            bst = train(params, dtrain, evals=evals,
                        evals_result=local_history, **kwargs)
            ctx[op.outputs[0].key] = {'booster': pickle.dumps(bst), 'history': local_history}
        else:
            # distributed
            rabit_args = ctx[op.tracker.key]
            rabit.init([arg.tobytes() if isinstance(arg, memoryview) else arg
                        for arg in rabit_args])
            try:
                local_history = dict()
                bst = train(params, dtrain, evals=evals, evals_result=local_history,
                            **op.kwargs)
                ret = {'booster': pickle.dumps(bst), 'history': local_history}
                if rabit.get_rank() != 0:
                    ret = {}
                ctx[op.outputs[0].key] = ret
            finally:
                rabit.finalize()


def train(params, dtrain, evals=(), **kwargs):
    """
    Train XGBoost model in Mars manner.

    Parameters
    ----------
    Parameters are the same as `xgboost.train`.

    Returns
    -------
    results: Booster
    """

    evals_result = kwargs.pop('evals_result', dict())
    session = kwargs.pop('session', None)
    run_kwargs = kwargs.pop('run_kwargs', dict())
    op = XGBTrain(params=params, dtrain=dtrain, evals=evals, kwargs=kwargs)
    t = op()
    ret = t.execute(session=session, **run_kwargs).fetch(session=session)
    evals_result.update(ret['history'])
    bst = pickle.loads(ret['booster'])
    num_class = params.get('num_class')
    if num_class:
        bst.set_attr(num_class=str(num_class))
    return bst
