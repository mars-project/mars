from ..core import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ..utils import parse_index
from .... import opcodes as OperandDef
from ....serialize import KeyField, SeriesField
import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pass


class DataFrameFromTensor(DataFrameOperand, DataFrameOperandMixin):
    """
    Represents data from mars tensor
    """
    _op_type_ = OperandDef.DATAFRAME_FROM_TENSOR
    _dtypes = SeriesField('dtypes')
    _input = KeyField('input')

    def __init__(self, dtypes=None, gpu=None, sparse=None, **kw):
        super(DataFrameFromTensor, self).__init__(_dtypes=dtypes,
                                                  _gpu=gpu, _sparse=sparse,
                                                  _object_type=ObjectType.dataframe, **kw)

    @property
    def dtypes(self):
        return self._dtypes

    @property
    def input(self):
        return self._input

    def _set_inputs(self, inputs):
        super(DataFrameFromTensor, self)._set_inputs(inputs)
        self._input = inputs

    def __call__(self, input_tensor):
        if input_tensor.ndim == 0:
            raise IndexError('Not support converted from 0-dim tensor')
        elif input_tensor.ndim == 1:
            # convert to Series
            index_value = pd.RangeIndex(start=0, stop=input_tensor.shape[0])
            columns_value = pd.RangeIndex(start=0, stop=1)

        else:
            # convert to DataFrame
            index_value = pd.RangeIndex(start=0, stop=input_tensor.shape[0])
            columns_value = pd.RangeIndex(start=0, stop=input_tensor.shape[1])

        return self.new_dataframe(input_tensor, input_tensor.shape, dtypes=self.dtypes,
                                  index_value=parse_index(index_value),
                                  columns_value=parse_index(columns_value))

    @classmethod
    def tile(cls, op):
        out_df = op.outputs[0]
        in_tensor = op.input
        out_chunks = []
        nsplits = in_tensor.nsplits
        if any(any(np.isnan(ns)) for ns in nsplits):
            raise NotImplementedError('NAN shape is not supported in DataFrame')

        cum_size = [np.cumsum(s) for s in nsplits]
        for in_chunk in in_tensor.chunks:
            out_op = op.copy().reset_key()
            if in_chunk.ndim == 1:
                i = in_chunk.index
                columns_value = pd.RangeIndex(0, 1)
            else:
                i, j = in_chunk.index
                column_stop = cum_size[1][j]
                columns_value = pd.RangeIndex(start=column_stop - in_chunk.shape[1], stop=column_stop)

            index_stop = cum_size[0][i]
            index_value = pd.RangeIndex(start=index_stop - in_chunk.shape[0], stop=index_stop)
            out_chunk = out_op.new_chunk([in_chunk], shape=in_chunk.shape, index=in_chunk.index,
                                         index_value=index_value,
                                         columns_value=columns_value)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_dataframes(None, out_df.shape, dtypes=out_df.dtypes,
                                     index_value=out_df.index_value,
                                     columns_value=out_df.columns,
                                     chunks=out_chunks, nsplits=in_tensor.nsplits)


def from_tensor(tensor, gpu=None, sparse=False):
    if tensor.ndim > 2:
        raise TypeError('Not support create DataFrame from {0} dims tensor', format(tensor.ndim))

    op = DataFrameFromTensor(dtypes=tensor.dtype, gpu=gpu, sparse=sparse)
    return op(tensor)
