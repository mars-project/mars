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

from collections.abc import Iterable
from functools import partial

from .. import opcodes
from ..context import ContextBase
from ..core import Entity, Base
from ..serialize import FunctionField, ListField, DictField, BoolField, Int32Field
from ..operands import ObjectOperand, ObjectOperandMixin
from ..tensor.core import TENSOR_TYPE
from ..dataframe.core import DATAFRAME_TYPE, SERIES_TYPE, INDEX_TYPE
from .utils import replace_inputs, find_objects


class RemoteFunction(ObjectOperand, ObjectOperandMixin):
    _op_type_ = opcodes.REMOTE_FUNCATION
    _op_module_ = 'remote'

    _function = FunctionField('function')
    _function_args = ListField('function_args')
    _function_kwargs = DictField('function_kwargs')
    _retry_when_fail = BoolField('retry_when_fail')
    _n_output = Int32Field('n_output')

    def __init__(self, function=None, function_args=None,
                 function_kwargs=None, retry_when_fail=None,
                 n_output=None, **kw):
        super().__init__(_function=function, _function_args=function_args,
                         _function_kwargs=function_kwargs,
                         _retry_when_fail=retry_when_fail,
                         _n_output=n_output, **kw)

    @property
    def function(self):
        return self._function

    @property
    def function_args(self):
        return self._function_args

    @property
    def function_kwargs(self):
        return self._function_kwargs

    @property
    def retry_when_fail(self):
        return self._retry_when_fail

    @property
    def n_output(self):
        return self._n_output

    @property
    def output_limit(self):
        return self._n_output or 1

    @property
    def retryable(self) -> bool:
        return self._retry_when_fail

    @classmethod
    def _no_prepare(cls, tileable):
        return isinstance(tileable, (TENSOR_TYPE, DATAFRAME_TYPE,
                                     SERIES_TYPE, INDEX_TYPE))

    def _set_inputs(self, inputs):
        raw_inputs = getattr(self, '_inputs', None)
        super()._set_inputs(inputs)

        function_inputs = iter(inp for inp in self._inputs
                               if isinstance(inp.op, RemoteFunction))
        mapping = {inp: new_inp for inp, new_inp in zip(inputs, self._inputs)}
        if raw_inputs is not None:
            for raw_inp in raw_inputs:
                if self._no_prepare(raw_inp):  # pragma: no cover
                    raise NotImplementedError
                else:
                    mapping[raw_inp] = next(function_inputs)
        self._function_args = replace_inputs(self._function_args, mapping)
        self._function_kwargs = replace_inputs(self._function_kwargs, mapping)

    def __call__(self):
        find_inputs = partial(find_objects, types=(Entity, Base))
        inputs = find_inputs(self._function_args) + find_inputs(self._function_kwargs)
        if any(self._no_prepare(inp) for inp in inputs):  # pragma: no cover
            raise NotImplementedError('For now DataFrame, Tensor etc '
                                      'cannot be passed to arguments')
        if self.n_output is None:
            return self.new_tileable(inputs)
        else:
            return self.new_tileables(
                inputs, kws=[dict(i=i) for i in range(self.n_output)])

    @classmethod
    def tile(cls, op):
        outs = op.outputs
        chunk_op = op.copy().reset_key()

        chunk_inputs = []
        prepare_inputs = []
        for inp in op.inputs:
            if cls._no_prepare(inp):  # pragma: no cover
                # if input is tensor, DataFrame etc,
                # do not prepare data, because the data mey be to huge,
                # and users can choose to fetch slice of the data themselves
                prepare_inputs.extend([False] * len(inp.chunks))
            else:
                prepare_inputs.extend([True] * len(inp.chunks))
            chunk_inputs.extend(inp.chunks)
        chunk_op._prepare_inputs = prepare_inputs

        out_chunks = [list() for _ in range(len(outs))]
        chunk_kws = []
        for i, out in enumerate(outs):
            chunk_params = out.params
            chunk_params['index'] = ()
            chunk_params['i'] = i
            chunk_kws.append(chunk_params)
        chunks = chunk_op.new_chunks(chunk_inputs, kws=chunk_kws)
        for i, c in enumerate(chunks):
            out_chunks[i].append(c)

        kws = []
        for i, out in enumerate(outs):
            params = out.params
            params['chunks'] = out_chunks[i]
            params['nsplits'] = ()
            kws.append(params)
        new_op = op.copy()
        return new_op.new_tileables(op.inputs, kws=kws)

    @classmethod
    def execute(cls, ctx, op: "RemoteFunction"):
        from ..session import Session

        session = ctx.get_current_session()
        prev_default_session = Session.default

        inputs_to_data = {inp: ctx[inp.key] for inp, prepare_inp
                          in zip(op.inputs, op.prepare_inputs) if prepare_inp}

        function = op.function
        function_args = replace_inputs(op.function_args, inputs_to_data)
        function_kwargs = replace_inputs(op.function_kwargs, inputs_to_data)

        # set session created from context as default one
        session.as_default()
        try:
            if isinstance(ctx, ContextBase):
                with ctx:
                    result = function(*function_args, **function_kwargs)
            else:
                result = function(*function_args, **function_kwargs)
        finally:
            # set back default session
            Session._set_default_session(prev_default_session)

        if op.n_output is None:
            ctx[op.outputs[0].key] = result
        else:
            if not isinstance(result, Iterable):
                raise TypeError('Specifying n_output={}, '
                                'but result is not iterable, got {}'.format(
                    op.n_output, result))
            result = list(result)
            if len(result) != op.n_output:
                raise ValueError('Length of return value should be {}, '
                                 'got {}'.format(op.n_output, len(result)))
            for out, r in zip(op.outputs, result):
                ctx[out.key] = r


def spawn(func, args=(), kwargs=None, retry_when_fail=True, n_output=None):
    if not isinstance(args, tuple):
        args = [args]
    else:
        args = list(args)
    if kwargs is None:
        kwargs = dict()
    if not isinstance(kwargs, dict):
        raise TypeError('kwargs has to be a dict')

    op = RemoteFunction(function=func, function_args=args,
                        function_kwargs=kwargs,
                        retry_when_fail=retry_when_fail,
                        n_output=n_output)
    return op()
