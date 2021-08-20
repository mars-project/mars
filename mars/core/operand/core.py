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

import sys
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import numpy as np
try:
    from numpy.core._exceptions import UFuncTypeError
except ImportError:  # pragma: no cover
    UFuncTypeError = None

from ...typing import TileableType, ChunkType, OperandType
from ...utils import calc_data_size
from ..context import Context
from ..mode import is_eager_mode
from ..entity import OutputType, TILEABLE_TYPE, ExecutableTuple, \
    get_chunk_types, get_tileable_types, \
    get_output_types, get_fetch_class

_op_type_to_executor: Dict[Type[OperandType], Callable] = dict()
_op_type_to_size_estimator: Dict[Type[OperandType], Callable] = dict()


class TileableOperandMixin:
    __slots__ = ()

    def check_inputs(self, inputs: List[TileableType]):
        if not inputs:
            return
        for inp in inputs:
            if isinstance(inp, TILEABLE_TYPE):
                i = inp.extra_params['_i']
                if not inp.op.output_types:
                    continue
                if inp.op.output_types[i] != OutputType.dataframe:
                    continue
                dtypes = getattr(inp, 'dtypes', None)
                if dtypes is None:
                    raise ValueError(
                        f'{inp} has unknown dtypes, '
                        f'it must be executed first before {str(type(self))}')

    @classmethod
    def _check_if_gpu(cls, inputs: List[TileableType]):
        if inputs is not None and \
                len([inp for inp in inputs
                     if inp is not None and getattr(inp, 'op', None) is not None]) > 0:
            if all(inp.op.gpu is True for inp in inputs):
                return True
            elif all(inp.op.gpu is False for inp in inputs):
                return False

    def _create_chunk(self,
                      output_idx: int,
                      index: Tuple[int],
                      **kw) -> ChunkType:
        output_type = kw.pop('output_type', self._get_output_type(output_idx))
        if not output_type:
            raise ValueError('output_type should be specified')

        if isinstance(output_type, (list, tuple)):
            output_type = output_type[output_idx]
        chunk_type, chunk_data_type = get_chunk_types(output_type)
        kw['_i'] = output_idx
        kw['op'] = self
        kw['index'] = index
        if output_type == OutputType.scalar:
            # tensor
            kw['order'] = 'C_ORDER'
        data = chunk_data_type(**kw)
        return chunk_type(data)

    def _new_chunks(self,
                    inputs: List[ChunkType],
                    kws: dict = None,
                    **kw) -> List[ChunkType]:
        output_limit = kw.pop('output_limit', None)
        if output_limit is None:
            output_limit = getattr(self, 'output_limit')

        self.check_inputs(inputs)
        getattr(self, '_set_inputs')(inputs)
        if getattr(self, 'gpu', None) is None:
            self.gpu = self._check_if_gpu(self._inputs)
        if getattr(self, '_key', None) is None:
            getattr(self, '_update_key')()

        chunks = []
        if isinstance(output_limit, float) and kws:
            output_limit = len(kws)
        for j in range(output_limit):
            create_chunk_kw = kw.copy()
            if kws:
                create_chunk_kw.update(kws[j])
            index = create_chunk_kw.pop('index', None)
            chunk = self._create_chunk(j, index, **create_chunk_kw)
            chunks.append(chunk)

        setattr(self, 'outputs', chunks)
        if len(chunks) > 1:
            # for each output chunk, hold the reference to the other outputs
            # so that either no one or everyone are gc collected
            for j, t in enumerate(chunks):
                t.data._siblings = [c.data for c in chunks[:j] + chunks[j + 1:]]
        return chunks

    def new_chunks(self,
                   inputs: List[ChunkType],
                   kws: dict = None,
                   **kwargs) -> List[ChunkType]:
        """
        Create chunks.

        A chunk is a node in a fine grained graph, all the chunk objects are created by
        calling this function, it happens mostly in tiles.
        The generated chunks will be set as this operand's outputs and each chunk will
        hold this operand as it's op.

        Parameters
        ----------
        inputs : list
            Input chunks.
        kws : dict
            Kwargs for each output.
        kwargs : dict
            common kwargs for all outputs

        Returns
        -------
        chunks : list
            Output chunks.

        .. note::
            It's a final method, do not override.
            Override the method `_new_chunks` if needed.
        """
        return self._new_chunks(inputs, kws=kws, **kwargs)

    def new_chunk(self,
                  inputs: List[ChunkType],
                  kws: dict = None,
                  **kw) -> ChunkType:
        if getattr(self, 'output_limit') != 1:
            raise TypeError('cannot new chunk with more than 1 outputs')

        return self.new_chunks(inputs, kws=kws, **kw)[0]

    @staticmethod
    def _fill_nan_shape(kw: dict):
        nsplits = kw.get('nsplits')
        shape = kw.get('shape')
        if nsplits is not None and shape is not None:
            nsplits = tuple(nsplits)
            shape = list(shape)
            for idx, (s, sp) in enumerate(zip(shape, nsplits)):
                if not np.isnan(s):
                    continue
                s = sum(sp)
                if not np.isnan(s):
                    shape[idx] = s
            kw['shape'] = tuple(shape)
            kw['nsplits'] = nsplits
        return kw

    def _create_tileable(self,
                         output_idx: int,
                         **kw) -> TileableType:
        output_type = kw.pop('output_type', self._get_output_type(output_idx))
        if output_type is None:
            raise ValueError('output_type should be specified')

        if isinstance(output_type, (list, tuple)):
            output_type = output_type[output_idx]
        tileable_type, tileable_data_type = get_tileable_types(output_type)
        kw['_i'] = output_idx
        kw['op'] = self
        if output_type == OutputType.scalar:
            # tensor
            kw['order'] = 'C_ORDER'

        kw = self._fill_nan_shape(kw)
        data = tileable_data_type(**kw)
        return tileable_type(data)

    def _new_tileables(self,
                       inputs: List[TileableType],
                       kws: dict = None,
                       **kw) -> List[TileableType]:
        output_limit = kw.pop('output_limit', None)
        if output_limit is None:
            output_limit = getattr(self, 'output_limit')

        self.check_inputs(inputs)
        getattr(self, '_set_inputs')(inputs)
        if getattr(self, 'gpu', None) is None:
            self.gpu = self._check_if_gpu(self._inputs)
        if getattr(self, '_key', None) is None:
            getattr(self, '_update_key')()  # update key when inputs are set

        tileables = []
        for j in range(output_limit):
            create_tensor_kw = kw.copy()
            if kws:
                create_tensor_kw.update(kws[j])
            tileable = self._create_tileable(j, **create_tensor_kw)
            tileables.append(tileable)

        setattr(self, 'outputs', tileables)
        if len(tileables) > 1:
            # for each output tileable, hold the reference to the other outputs
            # so that either no one or everyone are gc collected
            for j, t in enumerate(tileables):
                t.data._siblings = [tileable.data for tileable in tileables[:j] + tileables[j + 1:]]
        return tileables

    def new_tileables(self,
                      inputs: List[TileableType],
                      kws=None,
                      **kw) -> List[TileableType]:
        """
        Create tileable objects(Tensors or DataFrames).

        This is a base function for create tileable objects like tensors or dataframes,
        it will be called inside the `new_tensors` and `new_dataframes`.
        If eager mode is on, it will trigger the execution after tileable objects are created.

        Parameters
        ----------
        inputs : list
            Input tileables
        kws : dict
            Kwargs for each output.
        kw : dict
            Common kwargs for all outputs.

        Returns
        -------
        tileables : list
            Output tileables.

        .. note::
            It's a final method, do not override.
            Override the method `_new_tileables` if needed.
        """
        tileables = self._new_tileables(inputs, kws=kws, **kw)
        if is_eager_mode():
            ExecutableTuple(tileables).execute()
        return tileables

    def new_tileable(self,
                     inputs: List[TileableType],
                     kws: dict = None,
                     **kw) -> TileableType:
        if getattr(self, 'output_limit') != 1:
            raise TypeError('cannot new chunk with more than 1 outputs')

        return self.new_tileables(inputs, kws=kws, **kw)[0]

    @classmethod
    def pre_tile(cls, op: OperandType):
        """
        Operation before tile.

        Parameters
        ----------
        op : OperandType
          Operand to tile
        """

    @classmethod
    def tile(cls, op: OperandType):
        raise NotImplementedError

    @classmethod
    def post_tile(cls,
                  op: OperandType,
                  results: List[TileableType]):
        """
        Operation after tile.

        Parameters
        ----------
        op : OperandType
          Operand to tile.
        results: list
          List of tiled results.
        """

    @classmethod
    def pre_execute(cls,
                    ctx: Union[dict, Context],
                    op: OperandType):
        """
        Operation before execute.

        Parameters
        ----------
        ctx : dict
            Data store.
        op : OperandType
            Operand to execute.
        """

    @classmethod
    def execute(cls,
                ctx: Union[dict, Context],
                op: OperandType):
        raise NotImplementedError

    @classmethod
    def post_execute(cls,
                     ctx: Union[dict, Context],
                     op: OperandType):
        """
        Operand before execute.

        Parameters
        ----------
        ctx : dict
            Data store
        op : OperandType
            Operand to execute.
        """

    @classmethod
    def estimate_size(cls,
                      ctx: dict,
                      op: OperandType):
        from .fetch import FetchShuffle

        exec_size = 0
        outputs = op.outputs
        pure_dep_keys = \
            set(inp.key for inp, is_dep in zip(op.inputs or (), op.pure_depends or ()) if is_dep)
        if all(not c.is_sparse() and hasattr(c, 'nbytes') and
               not np.isnan(c.nbytes) for c in outputs):
            for out in outputs:
                ctx[out.key] = (out.nbytes, out.nbytes)

        for inp in op.inputs or ():
            if inp.key in pure_dep_keys:
                continue
            try:
                if isinstance(inp.op, FetchShuffle):
                    keys_and_shapes = inp.extra_params.get('_shapes', dict()).items()
                else:
                    keys_and_shapes = [(inp.key, getattr(inp, 'shape', None))]

                # execution size of a specific data chunk may be
                # larger than stored type due to objects
                for key, shape in keys_and_shapes:
                    exec_size += ctx[key][0]
            except KeyError:
                if not op.sparse:
                    inp_size = calc_data_size(inp)
                    if not np.isnan(inp_size):
                        exec_size += inp_size
        exec_size = int(exec_size)

        total_out_size = 0
        chunk_sizes = dict()
        for out in outputs:
            try:
                if not out.is_sparse():
                    chunk_size = calc_data_size(out)
                else:
                    chunk_size = exec_size
                if np.isnan(chunk_size):
                    raise TypeError
                chunk_sizes[out.key] = chunk_size
                total_out_size += chunk_size
            except (AttributeError, TypeError, ValueError):
                pass

        exec_size = max(exec_size, total_out_size)
        memory_scale = op.memory_scale or 1.0
        for out in outputs:
            if out.key in ctx:
                continue
            if out.key in chunk_sizes:
                result_size = chunk_sizes[out.key]
            else:
                result_size = max(exec_size // len(outputs),
                                  total_out_size // max(len(chunk_sizes), 1))
            try:
                if getattr(out, 'dtype', None) is not None and out.is_sparse():
                    max_sparse_size = out.nbytes + np.dtype(np.int64).itemsize * np.prod(out.shape) * out.ndim
                else:
                    max_sparse_size = np.nan
            except TypeError:  # pragma: no cover
                max_sparse_size = np.nan
            if not np.isnan(max_sparse_size):
                result_size = min(result_size, max_sparse_size)
            ctx[out.key] = (result_size, exec_size * memory_scale // len(outputs))

    @classmethod
    def concat_tileable_chunks(cls,
                               tileable: TileableType):
        raise NotImplementedError

    @classmethod
    def create_tileable_from_chunks(cls,
                                    chunks: List[ChunkType],
                                    inputs: List[TileableType] = None,
                                    **kw) -> TileableType:
        raise NotImplementedError

    def get_fetch_op_cls(self,
                         obj: ChunkType):
        from .shuffle import ShuffleProxy

        output_types = get_output_types(obj, unknown_as=OutputType.object)
        fetch_cls, fetch_shuffle_cls = get_fetch_class(output_types[0])
        if isinstance(self, ShuffleProxy):
            cls = fetch_shuffle_cls
        else:
            cls = fetch_cls

        def _inner(**kw):
            return cls(output_types=output_types, **kw)

        return _inner

    def get_fuse_op_cls(self,
                        obj: ChunkType):
        raise NotImplementedError

    @classmethod
    def register_executor(cls, executor: Callable):
        _op_type_to_executor[cls] = executor

    @classmethod
    def unregister_executor(cls):
        del _op_type_to_executor[cls]

    @classmethod
    def register_size_estimator(cls, size_estimator: Callable):
        _op_type_to_size_estimator[cls] = size_estimator

    @classmethod
    def unregister_size_estimator(cls):
        del _op_type_to_size_estimator[cls]


def execute(results: Dict[str, Any], op: OperandType):
    try:
        executor = _op_type_to_executor[type(op)]
    except KeyError:
        executor = type(op).execute

    # pre execute
    op.pre_execute(results, op)
    try:
        if UFuncTypeError is None:  # pragma: no cover
            return executor(results, op)
        else:
            # Cast `UFuncTypeError` to `TypeError` since subclasses of the former is unpickleable.
            # The `UFuncTypeError` was introduced by numpy#12593 since v1.17.0.
            try:
                return executor(results, op)
            except UFuncTypeError as e:  # pragma: no cover
                raise TypeError(str(e)).with_traceback(sys.exc_info()[2]) from None
    except NotImplementedError:
        for op_cls in type(op).__mro__:
            if op_cls in _op_type_to_executor:
                executor = _op_type_to_executor[op_cls]
                _op_type_to_executor[type(op)] = executor
                return executor(results, op)
        raise KeyError(f'No handler found for op: {op}')
    finally:
        op.post_execute(results, op)


def estimate_size(results: Dict[str, Any], op: OperandType):
    try:
        size_estimator = _op_type_to_size_estimator[type(op)]
    except KeyError:
        size_estimator = type(op).estimate_size

    try:
        return size_estimator(results, op)
    except NotImplementedError:
        for op_cls in type(op).__mro__:
            if op_cls in _op_type_to_size_estimator:
                size_estimator = _op_type_to_size_estimator[op_cls]
                _op_type_to_size_estimator[type(op)] = size_estimator
                return size_estimator(results, op)
        raise KeyError(f'No handler found for op: '
                       f'{op} to estimate size')
