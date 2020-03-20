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

import cloudpickle
import numpy as np

from ...context import RunningMode
from ...core import Object, OBJECT_TYPE
from ...serialize import KeyField, Int32Field, DictField, AnyField, BoolField
from ...tiles import TilesError
from ...tensor.core import TensorOrder
from ...utils import check_chunks_unknown_shape, tokenize
from ..operands import LearnOperand, LearnOperandMixin, OutputType


class TreeObject(Object):
    def fetch(self, session=None, **kw):
        result = self._data.fetch(session=session, **kw)
        return cloudpickle.loads(result) \
            if isinstance(result, bytes) else result


class TreeBase(LearnOperand, LearnOperandMixin):
    _input = KeyField('input')
    _leaf_size = Int32Field('leaf_size')
    _metric = AnyField('metric')

    _metric_params = DictField('metric_params')

    def __init__(self, leaf_size=None, metric=None,
                 metric_params=None, output_types=None, **kw):
        super().__init__(_leaf_size=leaf_size, _metric=metric,
                         _metric_params=metric_params,
                         _output_types=output_types, **kw)
        if self._output_types is None:
            self._output_types = [OutputType.object]

    @property
    def input(self):
        return self._input

    @property
    def leaf_size(self):
        return self._leaf_size

    @property
    def metric(self):
        return self._metric

    @property
    def metric_params(self):
        return self._metric_params

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, a):
        return self.new_tileable([a])

    @classmethod
    def tile(cls, op):
        check_chunks_unknown_shape(op.inputs, TilesError)

        # ball tree and kd tree requires the full data,
        # thus rechunk input tensor into 1 chunk
        inp = op.input.rechunk({ax: s for ax, s in enumerate(op.input.shape)})
        inp = inp._inplace_tile()
        out = op.outputs[0]

        chunk_op = op.copy().reset_key()
        kw = out.params
        kw['index'] = inp.chunks[0].index
        chunk = chunk_op.new_chunk([inp.chunks[0]], kws=[kw])

        new_op = op.copy()
        tileable_kw = out.params
        tileable_kw['nsplits'] = ((1,),)
        tileable_kw['chunks'] = [chunk]
        return new_op.new_tileables(op.inputs, kws=[tileable_kw])

    @classmethod
    def execute(cls, ctx, op):
        if op.gpu:  # pragma: no cover
            raise NotImplementedError('Does not support tree-based '
                                      'nearest neighbors on GPU')

        a = ctx[op.input.key]
        tree = cls._tree_type(
            a, op.leaf_size, metric=op.metric,
            **(op.metric_params or dict()))
        if ctx.running_mode in [RunningMode.local_cluster, RunningMode.distributed]:
            # for local cluster and distributed, pickle always
            ctx[op.outputs[0].key] = cloudpickle.dumps(tree)
        else:
            # otherwise, to be clear for local, just put into storage directly
            ctx[op.outputs[0].key] = tree


def _on_serialize_tree(tree):
    return cloudpickle.dumps(tree) if not hasattr(tree, 'key') else tree


def _on_deserialize_tree(ser):
    return cloudpickle.loads(ser) if isinstance(ser, bytes) else ser


class TreeQueryBase(LearnOperand, LearnOperandMixin):
    _input = KeyField('input')
    _tree = AnyField('tree', on_serialize=_on_serialize_tree,
                     on_deserialize=_on_deserialize_tree)
    _n_neighbors = Int32Field('n_neighbors')
    _return_distance = BoolField('return_distance')

    def __init__(self, tree=None, n_neighbors=None, return_distance=None,
                 output_types=None, **kw):
        super().__init__(_tree=tree, _n_neighbors=n_neighbors,
                         _return_distance=return_distance,
                         _output_types=output_types, **kw)
        if self._output_types is None:
            self._output_types = [OutputType.tensor] * self.output_limit

    @property
    def input(self):
        return self._input

    @property
    def tree(self):
        return self._tree

    @property
    def n_neighbors(self):
        return self._n_neighbors

    @property
    def return_distance(self):
        return self._return_distance

    @property
    def output_limit(self):
        return 2 if self._return_distance else 1

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]
        if isinstance(self._tree, OBJECT_TYPE):
            self._tree = self._inputs[1]

    def _update_key(self):
        values = []
        for value in self._values_:
            if isinstance(value, self._tree_type):
                values.append(cloudpickle.dumps(value))
            else:
                values.append(value)
        self._obj_set('_key', tokenize(type(self).__name__, *values))
        return self

    def __call__(self, x):
        kws = []
        if self._return_distance:
            kws.append({'shape': (x.shape[0], self._n_neighbors),
                        'dtype': np.dtype(np.float64),
                        'order': x.order,
                        'type': 'distance'})
        kws.append({
            'shape': (x.shape[0], self._n_neighbors),
            'dtype': np.dtype(np.int64),
            'order': TensorOrder.C_ORDER,
            'type': 'indices'
        })
        inputs = [x]
        if isinstance(self._tree, OBJECT_TYPE):
            inputs.append(self._tree)
        return self.new_tileables(inputs, kws=kws, output_limit=len(kws))

    @classmethod
    def tile(cls, op):
        inp = op.input

        if inp.chunk_shape[1] != 1:
            check_chunks_unknown_shape([inp], TilesError)
            inp = inp.rechunk({1: inp.shape[1]})._inplace_tile()

        tree_chunk = None
        if isinstance(op.tree, OBJECT_TYPE):
            tree_chunk = op.tree.chunks[0]
        out_chunks = [[] for _ in range(len(op.outputs))]
        for chunk in inp.chunks:
            chunk_op = op.copy().reset_key()
            if tree_chunk is not None:
                chunk_op._tree = tree_chunk
            chunk_kws = []
            if op.return_distance:
                chunk_kws.append({
                    'shape': (chunk.shape[0], op.n_neighbors),
                    'dtype': np.dtype(np.float64),
                    'order': chunk.order,
                    'index': chunk.index,
                    'type': 'distance'
                })
            chunk_kws.append({
                'shape': (chunk.shape[0], op.n_neighbors),
                'dtype': np.dtype(np.int64),
                'order': TensorOrder.C_ORDER,
                'index': chunk.index,
                'type': 'indices'
            })
            chunk_inputs = [chunk]
            if tree_chunk is not None:
                chunk_inputs.append(tree_chunk)
            chunks = chunk_op.new_chunks(chunk_inputs, kws=chunk_kws,
                                         output_limit=len(chunk_kws))
            for cs, c in zip(out_chunks, chunks):
                cs.append(c)

        kws = [o.params for o in op.outputs]
        nsplits = list(inp.nsplits)
        nsplits[1] = (op.n_neighbors,)
        if op.return_distance:
            kws[0]['chunks'] = out_chunks[0]
            kws[0]['nsplits'] = tuple(nsplits)
        kws[-1]['chunks'] = out_chunks[-1]
        kws[-1]['nsplits'] = tuple(nsplits)
        new_op = op.copy()
        return new_op.new_tileables(op.inputs, kws=kws, output_limit=len(kws))

    @classmethod
    def execute(cls, ctx, op):
        if op.gpu:  # pragma: no cover
            raise NotImplementedError('Does not support tree-based '
                                      'nearest neighbors on GPU')

        x = ctx[op.input.key]
        if len(op.inputs) == 2:
            tree = ctx[op.tree.key]
        else:
            tree = op.tree
        tree = cloudpickle.loads(tree) if isinstance(tree, bytes) else tree
        ret = tree.query(x, op.n_neighbors, op.return_distance)
        if op.return_distance:
            ctx[op.outputs[0].key] = ret[0]
            ctx[op.outputs[1].key] = ret[1]
        else:
            ctx[op.outputs[0].key] = ret
