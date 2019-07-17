from .core import TensorHasInput
from .... import opcodes as OperandDef
from ....serialize import KeyField
from ....dataframe.expressions.utils import build_empty_df


class TensorDataFrameDataSource(TensorHasInput):
    """ represent tensor from DataFrame """

    _op_type_ = OperandDef.TENSOR_FROM_DATAFRAME
    _input = KeyField('_input')

    def __init__(self, dtype=None, gpu=None, **kw):
        super(TensorDataFrameDataSource, self).__init__(_dtype=dtype, _gpu=gpu,
                                                        _sparse=True, **kw)


def from_dataframe(in_df):
    empty_pdf = build_empty_df(in_df.dtypes)
    op = TensorDataFrameDataSource(dtype=empty_pdf.dtypes[0], gpu=in_df.op.gpu)
    return op(in_df)
