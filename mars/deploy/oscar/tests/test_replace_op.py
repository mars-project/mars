from mars.tensor.arithmetic.add import TensorAdd


def _replace_op(ctx, op):
    # change the op from TensorAdd to TensorSubtract.
    type(op)._func_name = 'subtract'
    executor = type(op).execute
    return executor(ctx, op)


TensorAdd.register_executor(_replace_op)
