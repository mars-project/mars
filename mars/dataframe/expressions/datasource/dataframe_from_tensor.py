from ..core import DataFrameOperand, DataFrameOperandMixin
from ..utils import parse_index
from .... import opcodes as OperandDef
from ....serialize import DataFrameField, SeriesField


class TensorToDataFrame(DataFrameOperand, DataFrameOperandMixin):
    """
    Represents data from mars tensor
    """
    _op_type_ = OperandDef.DATAFRAME_FROM_TENSOR

    _data = DataFrameField('data')
    _dtypes = SeriesField('dtypes')

    def __init__(self, data=None, dtypes=None, gpu=None, sparse=None, **kw):
        if dtypes is None and data is not None:
            dtypes = data.dtypes
        super(TensorToDataFrame, self).__init__(_data=data, _dtypes=dtypes,
                                                _gpu=gpu, _sparse=sparse, **kw)

    @property
    def data(self):
        return self._data

    @property
    def dtypes(self):
        return self._dtypes

    def __call__(self, inputs, shape, index_value=None, columns_value=None, chunk_size=None):
        if index_value is None and columns_value is None:
            index_value = parse_index(self._data.index)
            columns_value = parse_index(self._data.columns, store_data=True)

        return self.new_dataframe(None, shape, dtypes=self.dtypes,
                                  index_value=index_value,
                                  columns_value=columns_value,
                                  raw_chunk_size=chunk_size)

    @classmethod
    def tile(cls, op):
        df = op.outputs[0]
        in_df = op.data
        out_chunks = []
        index_value = list(range(in_df.shape[0]))
        columns_value = list(range(in_df.shape[1]))
        for in_chunk in in_df.chunks:
            out_op = op.copy().reset_key()
            out_chunk = out_op.new_chunk([in_chunk], shape=in_chunk.shape, index=in_chunk.index,
                                         index_value=index_value, columns_value=columns_value)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_dataframes(op.inputs, df.shape, dtypes=op.dtypes,
                                     index_value=df.index_value,
                                     columns_value=df.columns,
                                     chunks=out_chunks, nsplits=in_df.nsplits)


def from_tensor(tensor, chunk_size=None, gpu=None, sparse=False):
    if tensor.ndim > 2:
        raise TypeError('Not support create DataFrame from {0} dims tensor', format(tensor.ndim))

    op = TensorToDataFrame(data=tensor, dtypes=tensor.dtype, gpu=gpu, sparse=sparse)

    # make index/column value if create DataFrame from tensor
    return op(tensor, tensor.shape, index_value=list(range(tensor.shape[0])),
              columns_value=list(range(tensor.shape[1])), chunk_size=chunk_size)
