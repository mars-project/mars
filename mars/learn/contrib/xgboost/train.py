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

import itertools
import logging
import pickle
from collections import OrderedDict, defaultdict

import numpy as np

from .... import opcodes as OperandDef
from ....core import OutputType
from ....core.context import get_context
from ....core.operand import MergeDictOperand
from ....serialization.serializables import FieldTypes, DictField, KeyField, ListField
from ....utils import ensure_own_data
from .dmatrix import ToDMatrix, to_dmatrix
from .start_tracker import StartTracker

logger = logging.getLogger(__name__)


def _on_serialize_evals(evals_val):
    if evals_val is None:
        return None
    return [list(x) for x in evals_val]


class XGBTrain(MergeDictOperand):
    _op_type_ = OperandDef.XGBOOST_TRAIN

    params = DictField("params", key_type=FieldTypes.string, default=None)
    dtrain = KeyField("dtrain", default=None)
    evals = ListField("evals", on_serialize=_on_serialize_evals, default=None)
    kwargs = DictField("kwargs", key_type=FieldTypes.string, default=None)
    tracker = KeyField("tracker", default=None)

    def __init__(self, gpu=None, **kw):
        super().__init__(gpu=gpu, **kw)
        if self.output_types is None:
            self.output_types = [OutputType.object]

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self.dtrain = self._inputs[0]
        rest = self._inputs[1:]
        if self.tracker is not None:
            self.tracker = self._inputs[-1]
            rest = rest[:-1]
        if self.evals is not None:
            evals_dict = OrderedDict(self.evals)
            new_evals_dict = OrderedDict()
            for new_key, val in zip(rest, evals_dict.values()):
                new_evals_dict[new_key] = val
            self.evals = list(new_evals_dict.items())

    def __call__(self):
        inputs = [self.dtrain]
        if self.evals is not None:
            inputs.extend(e[0] for e in self.evals)
        return self.new_tileable(inputs)

    @staticmethod
    def _get_dmatrix_chunks_workers(ctx, dmatrix):
        # dmatrix_chunk.inputs is concat, and concat's input is the coallocated chunks
        metas = ctx.get_chunks_meta(
            [c.inputs[0].inputs[0].key for c in dmatrix.chunks], fields=["bands"]
        )
        return [m["bands"][0][0] for m in metas]

    @classmethod
    def tile(cls, op: "XGBTrain"):
        ctx = get_context()

        inp = op.inputs[0]
        in_chunks = inp.chunks
        workers = cls._get_dmatrix_chunks_workers(ctx, inp)
        worker_to_in_chunks = dict(zip(workers, in_chunks))
        n_chunk = len(in_chunks)
        out_chunks = []
        worker_to_evals = defaultdict(list)
        if op.evals is not None:
            for dm, ev in op.evals:
                ev_workers = cls._get_dmatrix_chunks_workers(ctx, dm)
                for ev_worker, ev_chunk in zip(ev_workers, dm.chunks):
                    worker_to_evals[ev_worker].append((ev_chunk, ev))

        all_workers = set(workers)
        all_workers.update(worker_to_evals)

        i = itertools.count(n_chunk)
        tracker_chunk = StartTracker(
            n_workers=len(all_workers), pure_depends=[True] * n_chunk
        ).new_chunk(in_chunks, shape=())
        for worker in all_workers:
            chunk_op = op.copy().reset_key()
            chunk_op.expect_worker = worker
            chunk_op.tracker = tracker_chunk
            if worker in worker_to_in_chunks:
                in_chunk = worker_to_in_chunks[worker]
            else:
                in_chunk_op = ToDMatrix(
                    data=None,
                    label=None,
                    weight=None,
                    base_margin=None,
                    missing=inp.op.missing,
                    feature_names=inp.op.feature_names,
                    feature_types=inp.op.feature_types,
                    _output_types=inp.op.output_types,
                )
                params = inp.params.copy()
                params["index"] = (next(i),)
                params["shape"] = (0, inp.shape[1])
                in_chunk = in_chunk_op.new_chunk(None, kws=[params])
            chunk_evals = list(worker_to_evals.get(worker, list()))
            chunk_op.evals = chunk_evals
            input_chunks = (
                [in_chunk] + [pair[0] for pair in chunk_evals] + [tracker_chunk]
            )
            out_chunk = chunk_op.new_chunk(
                input_chunks, shape=(np.nan,), index=in_chunk.index[:1]
            )
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tileables(
            op.inputs, chunks=out_chunks, nsplits=((np.nan for _ in out_chunks),)
        )

    @classmethod
    def execute(cls, ctx, op: "XGBTrain"):
        if op.merge:
            return super().execute(ctx, op)

        from xgboost import train, rabit

        params = op.params.copy()

        n_threads = 0
        if op.tracker is None:
            # non distributed
            ctx_n_threads = -1
        else:
            # distributed
            ctx_n_threads = ctx.get_slots()

        # fix parallelism on nodes
        for p in ["nthread", "n_jobs"]:
            if (
                params.get(p, None) is not None
                and params.get(p, ctx_n_threads) != ctx_n_threads
            ):  # pragma: no cover
                logger.info("Overriding `nthreads` defined in Mars worker.")
                n_threads = params[p]
                break
        if n_threads == 0 or n_threads is None:  # pragma: no branch
            n_threads = ctx_n_threads
        params.update({"nthread": n_threads, "n_jobs": n_threads})

        dtrain = ToDMatrix.get_xgb_dmatrix(
            ensure_own_data(ctx[op.dtrain.key]), nthread=n_threads
        )
        evals = tuple()
        if op.evals is not None:
            eval_dmatrices = [
                ToDMatrix.get_xgb_dmatrix(
                    ensure_own_data(ctx[t[0].key]), nthread=n_threads
                )
                for t in op.evals
            ]
            evals = tuple((m, ev[1]) for m, ev in zip(eval_dmatrices, op.evals))

        if op.tracker is None:
            # non distributed
            local_history = dict()
            kwargs = dict() if op.kwargs is None else op.kwargs
            bst = train(
                params, dtrain, evals=evals, evals_result=local_history, **kwargs
            )
            ctx[op.outputs[0].key] = {
                "booster": pickle.dumps(bst),
                "history": local_history,
            }
        else:
            # distributed
            logger.debug("Distributed train params: %r", params)

            rabit_args = ctx[op.tracker.key]
            rabit.init(
                [
                    arg.tobytes() if isinstance(arg, memoryview) else arg
                    for arg in rabit_args
                ]
            )
            try:
                local_history = dict()
                bst = train(
                    params, dtrain, evals=evals, evals_result=local_history, **op.kwargs
                )
                ret = {"booster": pickle.dumps(bst), "history": local_history}
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

    evals_result = kwargs.pop("evals_result", dict())
    session = kwargs.pop("session", None)
    run_kwargs = kwargs.pop("run_kwargs", dict())

    processed_evals = []
    if evals:
        for eval_dmatrix, name in evals:
            if not isinstance(name, str):
                raise TypeError("evals must a list of pairs (DMatrix, string)")
            if hasattr(eval_dmatrix, "op") and isinstance(eval_dmatrix.op, ToDMatrix):
                processed_evals.append((eval_dmatrix, name))
            else:
                processed_evals.append((to_dmatrix(eval_dmatrix), name))

    op = XGBTrain(params=params, dtrain=dtrain, evals=processed_evals, kwargs=kwargs)
    t = op()
    ret = t.execute(session=session, **run_kwargs).fetch(session=session)
    evals_result.update(ret["history"])
    bst = pickle.loads(ret["booster"])
    num_class = params.get("num_class")
    if num_class:
        bst.set_attr(num_class=str(num_class))
    return bst
