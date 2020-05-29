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

import numpy as np
import pandas as pd

from .... import opcodes
from ....serialize import ValueType, AnyField, Int64Field, BoolField, \
    StringField, Int32Field, KeyField, TupleField, DictField, ListField
from ....tiles import TilesError
from ....utils import lazy_import, check_chunks_unknown_shape
from ...operands import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ...core import DATAFRAME_TYPE
from ...utils import build_empty_df, build_empty_series, parse_index

cudf = lazy_import('cudf', globals=globals())


class DataFrameRollingAgg(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.ROLLING_AGG

    _input = KeyField('input')
    _window = AnyField('window')
    _min_periods = Int64Field('min_periods')
    _center = BoolField('center')
    _win_type = StringField('win_type')
    _on = StringField('on')
    _axis = Int32Field('axis')
    _closed = StringField('closed')
    _func = AnyField('func')
    _func_args = TupleField('func_args')
    _func_kwargs = DictField('func_kwargs')
    # for chunks
    _preds = ListField('preds', ValueType.key)
    _succs = ListField('succs', ValueType.key)

    def __init__(self, input=None, window=None, min_periods=None, center=None,  # pylint: disable=redefined-builtin
                 win_type=None, on=None, axis=None, closed=None, func=None,
                 func_args=None, func_kwargs=None, object_type=None,
                 preds=None, succs=None, **kw):
        super().__init__(_input=input, _window=window, _min_periods=min_periods,
                         _center=center, _win_type=win_type, _on=on,
                         _axis=axis, _closed=closed, _func=func,
                         _func_args=func_args, _func_kwargs=func_kwargs,
                         _object_type=object_type,
                         _preds=preds, _succs=succs, **kw)

    @property
    def input(self):
        return self._input

    @property
    def window(self):
        return self._window

    @property
    def min_periods(self):
        return self._min_periods

    @property
    def center(self):
        return self._center

    @property
    def win_type(self):
        return self._win_type

    @property
    def on(self):
        return self._on

    @property
    def axis(self):
        return self._axis

    @property
    def closed(self):
        return self._closed

    @property
    def func(self):
        return self._func

    @property
    def func_args(self):
        return self._func_args

    @property
    def func_kwargs(self):
        return self._func_kwargs

    @property
    def preds(self):
        return self._preds if self._preds is not None else []

    @property
    def succs(self):
        return self._succs if self._succs is not None else []

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        input_iter = iter(self._inputs)
        self._input = next(input_iter)
        if self._preds is not None:
            self._preds = [next(input_iter) for _ in self._preds]
        if self._succs is not None:
            self._succs = [next(input_iter) for _ in self._succs]

    def __call__(self, rolling):
        inp = rolling.input

        if isinstance(inp, DATAFRAME_TYPE):
            pd_index = inp.index_value.to_pandas()
            empty_df = build_empty_df(inp.dtypes, index=pd_index[:0])
            params = rolling.params.copy()
            if params['win_type'] == 'freq':
                params['win_type'] = None
            if self._func != 'count':
                empty_df = empty_df._get_numeric_data()
            test_df = empty_df.rolling(**params).agg(self._func)
            if self._axis == 0:
                index_value = inp.index_value
            else:
                index_value = parse_index(test_df.index,
                                          rolling.params, inp,
                                          store_data=False)
            self._object_type = ObjectType.dataframe
            return self.new_dataframe(
                [inp], shape=(inp.shape[0], test_df.shape[1]),
                dtypes=test_df.dtypes, index_value=index_value,
                columns_value=parse_index(test_df.columns, store_data=True))
        else:
            pd_index = inp.index_value.to_pandas()
            empty_series = build_empty_series(inp.dtype, index=pd_index[:0],
                                              name=inp.name)
            test_obj = empty_series.rolling(**rolling.params).agg(self._func)
            if isinstance(test_obj, pd.DataFrame):
                self._object_type = ObjectType.dataframe
                return self.new_dataframe(
                    [inp], shape=(inp.shape[0], test_obj.shape[1]),
                    dtypes=test_obj.dtypes, index_value=inp.index_value,
                    columns_value=parse_index(test_obj.dtypes.index,
                                              store_data=True))
            else:
                self._object_type = ObjectType.series
                return self.new_series(
                    [inp], shape=inp.shape, dtype=test_obj.dtype,
                    index_value=inp.index_value, name=test_obj.name)

    @classmethod
    def _check_can_be_tiled(cls, op, is_window_int):
        inp = op.input
        axis = op.axis

        if axis == 0 and inp.ndim == 2:
            check_chunks_unknown_shape([inp], TilesError)
            inp = inp.rechunk({1: inp.shape[1]})._inplace_tile()

        if is_window_int:
            # if window is integer
            if any(np.isnan(ns) for ns in inp.nsplits[op.axis]):
                raise TilesError('input DataFrame or Series '
                                 'has unknown chunk shape on axis {}'.format(op.axis))
        else:
            # if window is offset
            # must be aware of index's meta including min and max
            for i in range(inp.chunk_shape[axis]):
                chunk_index = [0, 0]
                chunk_index[axis] = i
                chunk = inp.cix[tuple(chunk_index)]

                if axis == 0:
                    index_value = chunk.index_value
                else:
                    index_value = chunk.columns_value
                if pd.isnull(index_value.min_val) or pd.isnull(index_value.max_val):
                    raise TilesError('input DataFrame or Series '
                                     'has unknown index meta {}'.format(op.axis))

        return inp

    @classmethod
    def _find_extra_chunks_for_int_window(cls, op, inp, cur_chunk_index):
        from ...indexing.iloc import DataFrameIlocGetItem, SeriesIlocGetItem

        axis = op.axis
        window = op.window
        center = op.center

        # find prev chunks
        i = cur_chunk_index[axis]
        rest = window if not center else window // 2
        prev_chunks = []
        while i > 0 and rest > 0:
            prev_chunk_index = list(cur_chunk_index)
            prev_chunk_index[axis] = i - 1
            prev_chunk_index = tuple(prev_chunk_index)

            prev_chunk = inp.cix[prev_chunk_index]
            size = prev_chunk.shape[axis]
            if size <= rest:
                prev_chunks.insert(0, prev_chunk)
                rest -= size
            else:
                if prev_chunk.ndim == 1:
                    slice_prev_chunk_op = SeriesIlocGetItem(
                        indexes=[slice(-rest, None)])
                else:
                    slices = [slice(None)] * 2
                    slices[axis] = slice(-rest, None)
                    slice_prev_chunk_op = DataFrameIlocGetItem(indexes=slices)
                slice_prev_chunk = slice_prev_chunk_op.new_chunk([prev_chunk])
                prev_chunks.insert(0, slice_prev_chunk)
                rest = 0

            i -= 1

        # find succ chunks
        j = cur_chunk_index[axis]
        rest = 0 if not center else window - window // 2 - 1
        chunk_size = inp.chunk_shape[axis]
        succ_chunks = []
        while j < chunk_size - 1 and rest > 0:
            succ_chunk_index = list(cur_chunk_index)
            succ_chunk_index[axis] = j + 1
            succ_chunk_index = tuple(succ_chunk_index)

            succ_chunk = inp.cix[succ_chunk_index]
            size = succ_chunk.shape[axis]
            if size <= rest:
                succ_chunks.append(succ_chunk)
                rest -= size
            else:
                if succ_chunk.ndim == 1:
                    slice_succ_chunk_op = SeriesIlocGetItem(
                        indexes=[slice(rest)])
                else:
                    slices = [slice(None)] * 2
                    slices[axis] = slice(rest)
                    slice_succ_chunk_op = DataFrameIlocGetItem(indexes=slices)
                slice_succ_chunk = slice_succ_chunk_op.new_chunk([succ_chunk])
                succ_chunks.append(slice_succ_chunk)
                rest = 0

            j += 1

        return prev_chunks, succ_chunks

    @classmethod
    def _find_extra_chunks_for_offset_window(cls, op, inp, cur_chunk_index):
        from ...indexing.loc import DataFrameLocGetItem

        # when window is offset, center=True is not supported
        assert not op.center

        axis = op.axis
        window = pd.Timedelta(op.window)
        ndim = inp.ndim

        # find prev chunks
        i = cur_chunk_index[axis]
        prev_chunks = []
        cur_index_min = inp.cix[cur_chunk_index].index_value.min_val
        start = cur_index_min - window
        assert cur_chunk_index is not None
        while i > 0:
            prev_chunk_index = list(cur_chunk_index)
            prev_chunk_index[axis] = i - 1
            prev_chunk_index = tuple(prev_chunk_index)

            prev_chunk = inp.cix[prev_chunk_index]
            prev_index_max = prev_chunk.index_value.max_val
            if prev_index_max >= start:
                slices = [slice(None)] * ndim
                slices[axis] = slice(start, None)
                prev_chunk_op = DataFrameLocGetItem(indexes=slices,
                                                    object_type=prev_chunk.op.object_type)
                slice_prev_chunk = prev_chunk_op.new_chunk([prev_chunk])
                prev_chunks.insert(0, slice_prev_chunk)
            else:
                # index max < start, break
                break

            i -= 1

        return prev_chunks, []

    @classmethod
    def tile(cls, op):
        inp = op.input
        out = op.outputs[0]
        is_window_int = op.win_type != 'freq'
        axis = op.axis
        input_ndim = inp.ndim
        output_ndim = out.ndim

        # check if can be tiled
        inp = cls._check_can_be_tiled(op, is_window_int)

        if inp.ndim == 1 and out.ndim == 1:
            # input series, output series
            other_iter = [None]
        elif inp.ndim == 1:
            # input series, output dataframe
            other_iter = [0]
        else:
            other_iter = range(inp.chunk_shape[1 - axis])

        out_chunks = []
        for i in other_iter:
            for j in range(inp.chunk_shape[axis]):
                chunk_op = op.copy().reset_key()

                if inp.ndim == 1:
                    chunk_index = (j,)
                else:
                    chunk_index = [None, None]
                    chunk_index[1 - axis] = i
                    chunk_index[axis] = j
                    chunk_index = tuple(chunk_index)

                inp_chunk = inp.cix[chunk_index]
                if is_window_int:
                    pred_chunks, succ_chunks = \
                        cls._find_extra_chunks_for_int_window(op, inp, chunk_index)
                else:
                    pred_chunks, succ_chunks = \
                        cls._find_extra_chunks_for_offset_window(op, inp, chunk_index)

                out_chunk_index = [None] * output_ndim
                out_chunk_index[axis] = j
                if output_ndim == 2:
                    out_chunk_index[1 - axis] = i
                out_chunk_index = tuple(out_chunk_index)

                chunk_params = {'index': out_chunk_index}
                if input_ndim == 1 and output_ndim == 1:
                    chunk_params['shape'] = inp_chunk.shape
                    chunk_params['dtype'] = out.dtype
                    chunk_params['index_value'] = inp_chunk.index_value
                    chunk_params['name'] = inp_chunk.name
                elif input_ndim == 1 and output_ndim == 2:
                    chunk_params['shape'] = (inp_chunk.shape[0], out.shape[1])
                    chunk_params['dtypes'] = out.dtypes
                    chunk_params['index_value'] = inp_chunk.index_value
                    chunk_params['columns_value'] = out.columns_value
                else:
                    out_shape = list(out.shape)
                    out_shape[axis] = inp_chunk.shape[axis]
                    chunk_params['shape'] = tuple(out_shape)
                    chunk_params['index_value'] = \
                        inp_chunk.index_value if axis == 0 else out.index_value
                    chunk_params['dtypes'] = out.dtypes if axis == 0 else inp_chunk.dtypes
                    chunk_params['columns_value'] = \
                        out.columns_value if axis == 0 else inp_chunk.columns_value

                if len(pred_chunks) > 0:
                    chunk_op._preds = pred_chunks
                if len(succ_chunks) > 0:
                    chunk_op._succs = succ_chunks
                out_chunk = chunk_op.new_chunk([inp_chunk] + pred_chunks + succ_chunks,
                                               kws=[chunk_params])
                out_chunks.append(out_chunk)

        params = out.params
        params['chunks'] = out_chunks
        if out.ndim == 1:
            params['shape'] = (inp.shape[0],)
        else:
            params['shape'] = (inp.shape[0], params['shape'][1])
        nsplits = list(inp.nsplits)
        if input_ndim == 1 and output_ndim == 2:
            nsplits.append((out.shape[1],))
        elif input_ndim == 2 and output_ndim == 2:
            nsplits[1 - op.axis] = (out.shape[1 - op.axis],)
        params['nsplits'] = tuple(nsplits)
        new_op = op.copy()
        return new_op.new_tileables([inp], kws=[params])

    @classmethod
    def execute(cls, ctx, op):
        inp = ctx[op.input.key]
        axis = op.axis
        win_type = op.win_type
        window = op.window
        if win_type == 'freq':
            win_type = None
            window = pd.Timedelta(window)

        preds = [ctx[pred.key] for pred in op.preds]
        pred_size = sum(pred.shape[axis] for pred in preds)
        succs = [ctx[succ.key] for succ in op.succs]
        succ_size = sum(succ.shape[axis] for succ in succs)

        xdf = pd if isinstance(inp, (pd.DataFrame, pd.Series)) else cudf

        if pred_size > 0 or succ_size > 0:
            data = xdf.concat(preds + [inp] + succs, axis=axis)
        else:
            data = inp

        r = data.rolling(window=window, min_periods=op.min_periods,
                         center=op.center, win_type=win_type,
                         on=op.on, axis=axis, closed=op.closed)
        result = r.aggregate(op.func, *op.func_args, **op.func_kwargs)

        if pred_size > 0 or succ_size > 0:
            slc = [slice(None)] * result.ndim
            slc[axis] = slice(pred_size, result.shape[axis] - succ_size)
            result = result.iloc[tuple(slc)]

        ctx[op.outputs[0].key] = result
