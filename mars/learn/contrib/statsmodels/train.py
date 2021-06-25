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

import pickle  # nosec  # pylint: disable=import_pickle

import cloudpickle

from .... import opcodes
from ....core import OutputType, recursive_tile
from ....core.context import get_context
from ....core.operand import OperandStage, MergeDictOperand
from ....serialization.serializables import KeyField, BytesField, DictField, Int32Field, \
    Float32Field, FunctionField
from ....utils import has_unknown_shape


class StatsModelsTrain(MergeDictOperand):
    _op_type_ = opcodes.STATSMODELS_TRAIN

    _exog = KeyField('exog')  # exogenous
    _endog = KeyField('endog')  # endogenous

    _num_partitions = Int32Field('num_partitions')
    _partition_id = Int32Field('partition_id')
    _factor = Float32Field('factor')
    _model_class = BytesField('model_class', on_serialize=cloudpickle.dumps,
                              on_deserialize=cloudpickle.loads)
    _init_kwds = DictField('init_kwds')
    _fit_kwds = DictField('fit_kwds')
    _estimation_method = FunctionField('estimation_method')
    _estimation_kwds = DictField('estimation_kwds')
    _join_method = FunctionField('join_method')
    _join_kwds = DictField('join_kwds')
    _results_class = BytesField('results_class', on_serialize=cloudpickle.dumps,
                                on_deserialize=cloudpickle.loads)
    _results_kwds = DictField('results_kwds')

    def __init__(self, exog=None, endog=None, num_partitions=None, partition_id=None,
                 factor=None, model_class=None, init_kwds=None, fit_kwds=None,
                 estimation_method=None, estimation_kwds=None, join_method=None,
                 join_kwds=None, results_class=None, results_kwds=None, **kw):
        super().__init__(_exog=exog, _endog=endog, _num_partitions=num_partitions,
                         _partition_id=partition_id, _factor=factor,
                         _model_class=model_class, _init_kwds=init_kwds,
                         _fit_kwds=fit_kwds, _estimation_method=estimation_method,
                         _estimation_kwds=estimation_kwds, _join_method=join_method,
                         _join_kwds=join_kwds, _results_class=results_class,
                         _results_kwds=results_kwds, **kw)

    @property
    def exog(self):
        return self._exog

    @property
    def endog(self):
        return self._endog

    @property
    def num_partitions(self):
        return self._num_partitions

    @property
    def partition_id(self):
        return self._partition_id

    @property
    def factor(self):
        return self._factor

    @property
    def model_class(self):
        return self._model_class

    @property
    def init_kwds(self) -> dict:
        return self._init_kwds

    @property
    def fit_kwds(self) -> dict:
        return self._fit_kwds

    @property
    def estimation_method(self):
        return self._estimation_method

    @property
    def estimation_kwds(self) -> dict:
        return self._estimation_kwds

    @property
    def join_method(self):
        return self._join_method

    @property
    def join_kwds(self) -> dict:
        return self._join_kwds

    @property
    def results_class(self):
        return self._results_class

    @property
    def results_kwds(self) -> dict:
        return self._results_kwds

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(inputs)
        self._exog = next(inputs_iter)
        self._endog = next(inputs_iter)

    def __call__(self, exog, endog):
        self._output_types = [OutputType.object]
        return self.new_tileable([exog, endog])

    @classmethod
    def tile(cls, op: "StatsModelsTrain"):
        if op.factor is not None:
            ctx = get_context()
            cluster_cpu_count = ctx.get_total_n_cpu()
            assert cluster_cpu_count > 0
            num_partitions = int(cluster_cpu_count * op.factor)
        else:
            num_partitions = op.num_partitions

        if has_unknown_shape(op.exog, op.endog):
            yield

        exog = op.exog
        if exog.ndim > 1 and exog.chunk_shape[1] > 1:
            exog = exog.rechunk({1: exog.shape[1]})
        exog = yield from recursive_tile(
            exog.rebalance(num_partitions=num_partitions))
        endog = yield from recursive_tile(
            op.endog.rebalance(num_partitions=num_partitions))

        assert len(exog.chunks) == len(endog.chunks)

        # generate map stage
        map_chunks = []
        for part_id, (exog_chunk, endog_chunk) in enumerate(zip(exog.chunks, endog.chunks)):
            new_op = op.copy().reset_key()
            new_op._factor = None
            new_op._partition_id = part_id
            new_op._num_partitions = num_partitions
            new_op.stage = OperandStage.map

            map_chunks.append(new_op.new_chunk(
                [exog_chunk, endog_chunk], index=exog_chunk.index))

        # generate combine (join) stage
        new_op = op.copy().reset_key()
        new_op._factor = None
        new_op._num_partitions = num_partitions
        new_op.stage = OperandStage.combine

        combine_chunk = new_op.new_chunk(map_chunks, index=(0,))

        # generate tileable
        new_op = op.copy().reset_key()
        return new_op.new_tileables(op.inputs, chunks=[combine_chunk])

    @classmethod
    def _execute_map(cls, ctx, op: "StatsModelsTrain"):
        endog = ctx[op.endog.key]
        exog = ctx[op.exog.key]

        # code from statsmodels.base.distributed_estimation::_helper_fit_partition
        model = op.model_class(endog, exog, **op.init_kwds)
        results = op.estimation_method(model, op.partition_id, op.num_partitions,
                                       fit_kwds=op.fit_kwds, **op.estimation_kwds)
        ctx[op.outputs[0].key] = pickle.dumps(results)

    @classmethod
    def _execute_combine(cls, ctx, op: "StatsModelsTrain"):
        # code from statsmodels.base.distributed_estimation::DistributedModel.fit
        results_list = [pickle.loads(ctx[inp.key]) for inp in op.inputs]  # nosec
        params = op.join_method(results_list, **op.join_kwds)
        res_mod = op.model_class([0], [0], **op.init_kwds)
        result = op.results_class(res_mod, params, **op.results_kwds)

        ctx[op.outputs[0].key] = pickle.dumps(result)

    @classmethod
    def execute(cls, ctx, op: "StatsModelsTrain"):
        if op.merge:  # pragma: no cover
            super().execute(ctx, op)
        elif op.stage == OperandStage.combine:
            cls._execute_combine(ctx, op)
        else:
            cls._execute_map(ctx, op)
