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

import numpy as np

from ... import opcodes as OperandDef
from ... import tensor as mt
from ...core import ENTITY_TYPE, ExecutableTuple, recursive_tile
from ...core.context import get_context
from ...serialization.serializables import AnyField, KeyField
from ...tensor.core import TENSOR_TYPE, TensorOrder
from ..operands import LearnOperand, LearnOperandMixin, OutputType
from ..utils.multiclass import type_of_target
from ..utils import check_consistent_length, column_or_1d


class CheckTargets(LearnOperand, LearnOperandMixin):
    _op_type_ = OperandDef.CHECK_TARGETS

    _y_true = AnyField('y_true')
    _y_pred = AnyField('y_pred')
    _type_true = KeyField('type_true')
    _type_pred = KeyField('type_pred')

    def __init__(self, y_true=None, y_pred=None, type_true=None, type_pred=None, **kw):
        super().__init__(_y_true=y_true, _y_pred=y_pred,
                         _type_true=type_true, _type_pred=type_pred, **kw)
        # scalar(y_type), y_true, y_pred
        self.output_types = [OutputType.tensor] * 3

    @property
    def output_limit(self):
        return 3

    @property
    def y_true(self):
        return self._y_true

    @property
    def y_pred(self):
        return self._y_pred

    @property
    def type_true(self):
        return self._type_true

    @property
    def type_pred(self):
        return self._type_pred

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs)
        if isinstance(self._y_true, ENTITY_TYPE):
            self._y_true = next(inputs_iter)
        if isinstance(self._y_pred, ENTITY_TYPE):
            self._y_pred = next(inputs_iter)
        if self._type_true is not None:
            self._type_true = next(inputs_iter)
        if self._type_pred is not None:
            self._type_pred = next(inputs_iter)

    def __call__(self, y_true, y_pred):
        inputs = []
        if isinstance(y_true, ENTITY_TYPE):
            inputs.append(y_true)
        if isinstance(y_pred, ENTITY_TYPE):
            inputs.append(y_pred)
        self._type_true = type_of_target(y_true)
        self._type_pred = type_of_target(y_pred)
        inputs.extend([self._type_true, self._type_pred])

        kws = list()
        kws.append({
            'shape': (),
            'dtype': np.dtype(object),
            'order': TensorOrder.C_ORDER
        })
        kws.extend([y.params for y in (mt.tensor(y_true),
                                       mt.tensor(y_pred))])
        kws[1]['shape'] = kws[2]['shape'] = (np.nan,)
        return ExecutableTuple(self.new_tileables(inputs, kws=kws))

    @classmethod
    def tile(cls, op):
        y_true, y_pred = op.y_true, op.y_pred
        for y in (op.y_true, op.y_pred):
            if isinstance(y, ENTITY_TYPE):
                if np.isnan(y.size):  # pragma: no cover
                    yield

        check_consistent_length(y_true, y_pred)

        # make sure type_true and type_pred executed first
        chunks = [op.type_true.chunks[0], op.type_pred.chunks[0]]
        yield chunks

        ctx = get_context()
        type_true, type_pred = [d.item() if hasattr(d, 'item') else d
                                for d in ctx.get_chunks_result([c.key for c in chunks])]

        y_type = {type_true, type_pred}
        if y_type == {"binary", "multiclass"}:
            y_type = {"multiclass"}

        if len(y_type) > 1:
            raise ValueError(f"Classification metrics can't handle a mix of {type_true} "
                             f"and {type_pred} targets")

        # We can't have more than one value on y_type => The set is no more needed
        y_type = y_type.pop()

        # No metrics support "multiclass-multioutput" format
        if (y_type not in ["binary", "multiclass", "multilabel-indicator"]):
            raise ValueError(f"{y_type} is not supported")

        if y_type in ["binary", "multiclass"]:
            y_true = column_or_1d(y_true)
            y_pred = column_or_1d(y_pred)
            if y_type == "binary":
                unique_values = mt.union1d(y_true, y_pred)
                y_type = mt.where(mt.count_nonzero(unique_values) > 2,
                                  'multiclass', y_type)
        elif y_type.startswith('multilabel'):
            y_true = mt.tensor(y_true).tosparse()
            y_pred = mt.tensor(y_pred).tosparse()
            y_type = 'multilabel-indicator'

        if not isinstance(y_true, ENTITY_TYPE):
            y_true = mt.tensor(y_true)
        if not isinstance(y_pred, ENTITY_TYPE):
            y_pred = mt.tensor(y_pred)

        if not isinstance(y_type, TENSOR_TYPE):
            y_type = mt.tensor(y_type, dtype=object)

        y_type = yield from recursive_tile(y_type)
        y_true = yield from recursive_tile(y_true)
        y_pred = yield from recursive_tile(y_pred)

        kws = [out.params for out in op.outputs]
        kws[0].update(dict(nsplits=(), chunks=[y_type.chunks[0]]))
        kws[1].update(dict(nsplits=y_true.nsplits, chunks=y_true.chunks,
                           shape=tuple(sum(sp) for sp in y_true.nsplits)))
        kws[2].update(dict(nsplits=y_pred.nsplits, chunks=y_pred.chunks,
                           shape=tuple(sum(sp) for sp in y_pred.nsplits)))
        new_op = op.copy()
        return new_op.new_tileables(op.inputs, kws=kws)


def _check_targets(y_true, y_pred):
    """Check that y_true and y_pred belong to the same classification task

    This converts multiclass or binary types to a common shape, and raises a
    ValueError for a mix of multilabel and multiclass targets, a mix of
    multilabel formats, for the presence of continuous-valued or multioutput
    targets, or for targets of different lengths.

    Column vectors are squeezed to 1d, while multilabel formats are returned
    as CSR sparse label indicators.

    Parameters
    ----------
    y_true : array-like

    y_pred : array-like

    Returns
    -------
    type_true : one of {'multilabel-indicator', 'multiclass', 'binary'}
        The type of the true target data, as output by
        ``utils.multiclass.type_of_target``

    y_true : Tensor

    y_pred : Tensor
    """
    op = CheckTargets(y_true=y_true, y_pred=y_pred)
    return op(y_true, y_pred)
