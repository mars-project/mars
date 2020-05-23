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

from functools import partial

from .. import opcodes
from ..core import Entity, Base
from ..serialize import FunctionField, ListField, DictField
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

    def __init__(self, function=None, function_args=None,
                 function_kwargs=None, **kw):
        super().__init__(_function=function, _function_args=function_args,
                         _function_kwargs=function_kwargs, **kw)

    @property
    def function(self):
        return self._function

    @property
    def function_args(self):
        return self._function_args

    @property
    def function_kwargs(self):
        return self._function_kwargs

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
        return self.new_tileable(inputs)

    @classmethod
    def tile(cls, op):
        out = op.outputs[0]

        chunk_op = op.copy().reset_key()
        chunk_params = out.params
        chunk_params['index'] = ()

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
        chunk = chunk_op.new_chunk(chunk_inputs, kws=[chunk_params])

        new_op = op.copy()
        params = out.params
        params['chunks'] = [chunk]
        params['nsplits'] = ()
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    def execute(cls, ctx, op):
        from ..session import Session

        session = ctx.get_current_session()
        prev_default_session = Session.default

        inputs_to_data = {inp: ctx[inp.key] for inp, prepare_inp
                          in zip(op.inputs, op.prepare_inputs) if prepare_inp}

        try:
            # set session created from context as default one
            session.as_default()

            function = op.function
            function_args = replace_inputs(op.function_args, inputs_to_data)
            function_kwargs = replace_inputs(op.function_kwargs, inputs_to_data)

            result = function(*function_args, **function_kwargs)
            ctx[op.outputs[0].key] = result
        finally:
            # set back default session
            Session._set_default_session(prev_default_session)


def spawn(func, args=(), kwargs=None):
    if not isinstance(args, tuple):
        args = [args]
    else:
        args = list(args)
    if kwargs is None:
        kwargs = dict()
    if not isinstance(kwargs, dict):
        raise TypeError('kwargs has to be a dict')

    op = RemoteFunction(function=func, function_args=args,
                        function_kwargs=kwargs)
    return op()
