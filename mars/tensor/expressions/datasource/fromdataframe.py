from .core import TensorHasInput
from .... import opcodes as OperandDef
from ....serialize import KeyField


class TensorDataFrameDataSource(TensorHasInput):
    """ represent tensor from DataFrame """

    _op_type_ = OperandDef.TENSOR_FROM_DATAFRAME
    _input = KeyField('_input')

    def __init__(self, dtype=None, gpu=None, **kw):
        super(TensorDataFrameDataSource, self).__init__(_dtype=dtype, _gpu=gpu,
                                                        _sparse=True, **kw)


def from_dataframe(in_df):
    if not all(in_df.dtypes[0] == elem for elem in in_df.dtypes):
        raise TypeError('Not support heterogeneous DataFrame to tensor')

    op = TensorDataFrameDataSource(dtype=in_df.dtypes[0], gpu=in_df.op.gpu)
    return op(in_df)
