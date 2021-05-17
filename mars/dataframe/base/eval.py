#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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

import ast
import binascii
import operator
import sys
import tokenize
import textwrap
from collections import OrderedDict
from functools import reduce
from io import StringIO

import numpy as np
import pandas as pd

from ... import opcodes
from ...core import ENTITY_TYPE, OutputType, get_output_types, recursive_tile
from ...serialization.serializables import BoolField, DictField, StringField
from ..core import DATAFRAME_TYPE
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import parse_index

LOCAL_TAG = '_local_var_'
BACKTICK_TAG = '_backtick_var_'


def _tokenize_str(reader):
    token_generator = tokenize.generate_tokens(reader)

    def _iter_backtick_string(gen, line, back_start):
        for _, tokval, start, _, _ in gen:
            if tokval == '`':
                return BACKTICK_TAG + binascii.b2a_hex(
                    line[back_start[1] + 1:start[1]].encode()).decode()
        else:
            raise SyntaxError(f'backtick quote at {back_start} does not match')

    for toknum, tokval, start, _, line in token_generator:
        if toknum == tokenize.OP:
            if tokval == '@':
                tokval = LOCAL_TAG
            if tokval == '&':
                toknum = tokenize.NAME
                tokval = 'and'
            elif tokval == '|':
                toknum = tokenize.NAME
                tokval = 'or'
        elif tokval == '`':
            yield tokenize.NAME, _iter_backtick_string(token_generator, line, start)
            continue
        yield toknum, tokval


class CollectionVisitor(ast.NodeVisitor):
    _op_handlers = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.mod: operator.mod,
        ast.Pow: operator.pow,

        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.In: lambda x, y: y.isin(x),
        ast.NotIn: lambda x, y: ~y.isin(x),

        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
        ast.Invert: operator.invert,

        ast.And: operator.and_,
        ast.Or: operator.or_
    }

    def __init__(self, resolvers, target, env):
        self.env = env
        self.target = target
        self.resolvers = resolvers

        self.referenced_vars = set()
        self.assigned = False
        self.entity_subscribe = False

    def _preparse(self, expr):
        reader = StringIO(expr).readline
        return tokenize.untokenize(list(_tokenize_str(reader)))

    def eval(self, expr, rewrite=True):
        if rewrite:
            expr = self._preparse(expr)
        node = ast.fix_missing_locations(ast.parse(expr))
        return self.visit(node)

    def get_named_object(self, obj_name):
        for resolver in self.resolvers:
            try:
                return resolver[obj_name]
            except (IndexError, KeyError):
                continue
        if obj_name in self.env:
            self.referenced_vars.add(obj_name)
            return self.env[obj_name]
        raise KeyError(f'name {obj_name} is not defined')

    def visit(self, node):
        if isinstance(node, DATAFRAME_TYPE):
            return node
        node_name = node.__class__.__name__
        method = 'visit_' + node_name
        try:
            visitor = getattr(self, method)
        except AttributeError:
            raise SyntaxError('Query string contains unsupported syntax: {}'.format(node_name))
        return visitor(node)

    def visit_Module(self, node):
        if self.target is None and len(node.body) != 1:
            raise SyntaxError('Only a single expression is allowed')
        result = None
        for expr in node.body:
            result = self.visit(expr)
        return result

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        return self._op_handlers[type(node.op)](left, right)

    def visit_Call(self, node):
        func = self.visit(node.func)
        args = [self.visit(n) for n in node.args]
        kwargs = OrderedDict([(kw.arg, self.visit(kw.value)) for kw in node.keywords])
        return func(*args, **kwargs)

    def visit_Compare(self, node):
        ops = node.ops
        comps = node.comparators

        if len(comps) == 1:
            binop = ast.BinOp(op=ops[0], left=node.left, right=comps[0])
            return self.visit(binop)

        left = node.left
        values = []
        for op, comp in zip(ops, comps):
            new_node = ast.Compare(comparators=[comp], left=left, ops=[op])
            left = comp
            values.append(new_node)
        return self.visit(ast.BoolOp(op=ast.And(), values=values))

    def visit_BoolOp(self, node):
        def func(lhs, rhs):
            binop = ast.BinOp(op=node.op, left=lhs, right=rhs)
            return self.visit(binop)
        return reduce(func, node.values)

    def visit_UnaryOp(self, node):
        op = self.visit(node.operand)
        return self._op_handlers[type(node.op)](op)

    def visit_Name(self, node):
        if node.id.startswith(LOCAL_TAG):
            local_name = node.id.replace(LOCAL_TAG, '')
            self.referenced_vars.add(local_name)
            return self.env[local_name]
        if node.id.startswith(BACKTICK_TAG):
            local_name = binascii.a2b_hex(node.id.replace(BACKTICK_TAG, '').encode()).decode()
            return self.get_named_object(local_name)
        return self.get_named_object(node.id)

    def visit_NameConstant(self, node):  # pragma: no cover
        return node.value

    def visit_Num(self, node):  # pragma: no cover
        return node.n

    def visit_Str(self, node):  # pragma: no cover
        return node.s

    def visit_Constant(self, node):
        return node.value

    def visit_List(self, node):
        return [self.visit(e) for e in node.elts]

    def visit_Assign(self, node):
        if self.target is None:
            raise ValueError('Target not specified for assignment')
        if isinstance(node.targets[0], ast.Tuple):
            raise ValueError('Does not support assigning to multiple objects')

        target = node.targets[0].id
        value = self.visit(node.value)
        self.target[target] = value
        self.assigned = True

    visit_Tuple = visit_List

    def visit_Attribute(self, node):
        attr = node.attr
        value = node.value

        ctx = node.ctx
        if isinstance(ctx, ast.Load):
            resolved = self.visit(value)
            return getattr(resolved, attr)

        raise ValueError("Invalid Attribute context {0}".format(ctx.__name__))

    def visit_Subscript(self, node):
        value = self.visit(node.value)
        sub = self.visit(node.slice)
        if isinstance(value, ENTITY_TYPE):
            self.entity_subscribe = True
        return value[sub]

    def visit_Index(self, node):
        return self.visit(node.value)

    def visit_Slice(self, node):
        lower = node.lower
        if lower is not None:
            lower = self.visit(lower)
        upper = node.upper
        if upper is not None:
            upper = self.visit(upper)
        step = node.step
        if step is not None:
            step = self.visit(step)

        return slice(lower, upper, step)


class DataFrameEval(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.DATAFRAME_EVAL

    _expr = StringField('expr')
    _parser = StringField('parser')
    _engine = StringField('engine')
    _variables = DictField('variables')
    _self_target = BoolField('self_target')
    _is_query = BoolField('is_query')

    def __init__(self, expr=None, parser=None, engine=None, variables=None,
                 self_target=None, is_query=None, **kw):
        super().__init__(_expr=expr, _parser=parser, _engine=engine, _variables=variables,
                         _self_target=self_target, _is_query=is_query, **kw)

    @property
    def expr(self):
        return self._expr

    @property
    def parser(self):
        return self._parser

    @property
    def engine(self):
        return self._engine

    @property
    def variables(self):
        return self._variables

    @property
    def self_target(self):
        return self._self_target

    @property
    def is_query(self):
        return self._is_query

    def __call__(self, df, output_type, shape, dtypes):
        self._output_types = [output_type]
        params = df.params
        new_index_value = df.index_value if not np.isnan(shape[0]) else parse_index(pd.RangeIndex(-1))
        if output_type == OutputType.dataframe:
            params.update(dict(
                dtypes=dtypes, shape=shape,
                columns_value=parse_index(dtypes.index, store_data=True),
                index_value=new_index_value,
            ))
        else:
            name, dtype = dtypes
            params = dict(
                name=name, dtype=dtype, shape=shape,
                index_value=new_index_value,
            )
        return self.new_tileable([df], **params)

    def convert_to_query(self, df, output_type, shape, dtypes):
        new_op = self.copy().reset_key()
        new_op._is_query = True
        new_op._self_target = False
        return new_op(df, output_type, shape, dtypes)

    @classmethod
    def tile(cls, op: 'DataFrameEval'):
        in_df = op.inputs[0]
        out_df = op.outputs[0]

        if in_df.ndim == 2:
            if in_df.chunk_shape[1] > 1:
                in_df = yield from recursive_tile(
                    in_df.rechunk({1: in_df.shape[1]}))

        chunks = []
        for c in in_df.chunks:
            if out_df.ndim == 2:
                new_shape = (np.nan if np.isnan(out_df.shape[0]) else c.shape[0], out_df.shape[1])
                params = dict(
                    dtypes=out_df.dtypes, shape=new_shape,
                    columns_value=parse_index(out_df.dtypes.index, store_data=True),
                    index_value=c.index_value,
                    index=c.index,
                )
            else:
                new_shape = (np.nan if np.isnan(out_df.shape[0]) else c.shape[0],)
                params = dict(
                    name=out_df.name, dtype=out_df.dtype, shape=new_shape,
                    index_value=c.index_value,
                    index=(c.index[0],)
                )
            new_op = op.copy().reset_key()
            chunks.append(new_op.new_chunk([c], **params))

        new_op = op.copy().reset_key()
        params = out_df.params

        new_nsplits = [in_df.nsplits[0], (out_df.shape[-1],)]
        if np.isnan(out_df.shape[0]):
            new_nsplits[0] = (np.nan,) * len(in_df.nsplits[0])
        if out_df.ndim == 1:
            new_nsplits = new_nsplits[:1]

        params.update(dict(
            chunks=chunks,
            nsplits=tuple(new_nsplits),
        ))
        return new_op.new_tileables([in_df], **params)

    @classmethod
    def execute(cls, ctx, op: 'DataFrameEval'):
        in_data = ctx[op.inputs[0].key]

        if op.self_target:
            in_data = in_data.copy()

        if op.is_query:
            val = in_data.query(op.expr, parser=op.parser, engine=op.engine, local_dict=op.variables)
        else:
            val = in_data.eval(op.expr, parser=op.parser, engine=op.engine, local_dict=op.variables)
        ctx[op.outputs[0].key] = val


def mars_eval(expr, parser='mars', engine=None, local_dict=None, global_dict=None,
              resolvers=(), level=0, target=None, inplace=False):
    """

    Evaluate a Python expression as a string using various backends.

    The following arithmetic operations are supported: ``+``, ``-``, ``*``,
    ``/``, ``**``, ``%``, ``//`` (python engine only) along with the following
    boolean operations: ``|`` (or), ``&`` (and), and ``~`` (not).
    Additionally, the ``'pandas'`` parser allows the use of :keyword:`and`,
    :keyword:`or`, and :keyword:`not` with the same semantics as the
    corresponding bitwise operators.  :class:`~pandas.Series` and
    :class:`~pandas.DataFrame` objects are supported and behave as they would
    with plain ol' Python evaluation.

    Parameters
    ----------
    expr : str
        The expression to evaluate. This string cannot contain any Python
        `statements
        <https://docs.python.org/3/reference/simple_stmts.html#simple-statements>`__,
        only Python `expressions
        <https://docs.python.org/3/reference/simple_stmts.html#expression-statements>`__.
    local_dict : dict or None, optional
        A dictionary of local variables, taken from locals() by default.
    global_dict : dict or None, optional
        A dictionary of global variables, taken from globals() by default.
    resolvers : list of dict-like or None, optional
        A list of objects implementing the ``__getitem__`` special method that
        you can use to inject an additional collection of namespaces to use for
        variable lookup. For example, this is used in the
        :meth:`~DataFrame.query` method to inject the
        ``DataFrame.index`` and ``DataFrame.columns``
        variables that refer to their respective :class:`~pandas.DataFrame`
        instance attributes.
    level : int, optional
        The number of prior stack frames to traverse and add to the current
        scope. Most users will **not** need to change this parameter.
    target : object, optional, default None
        This is the target object for assignment. It is used when there is
        variable assignment in the expression. If so, then `target` must
        support item assignment with string keys, and if a copy is being
        returned, it must also support `.copy()`.
    inplace : bool, default False
        If `target` is provided, and the expression mutates `target`, whether
        to modify `target` inplace. Otherwise, return a copy of `target` with
        the mutation.

    Returns
    -------
    ndarray, numeric scalar, DataFrame, Series

    Raises
    ------
    ValueError
        There are many instances where such an error can be raised:

        - `target=None`, but the expression is multiline.
        - The expression is multiline, but not all them have item assignment.
          An example of such an arrangement is this:

          a = b + 1
          a + 2

          Here, there are expressions on different lines, making it multiline,
          but the last line has no variable assigned to the output of `a + 2`.
        - `inplace=True`, but the expression is missing item assignment.
        - Item assignment is provided, but the `target` does not support
          string item assignment.
        - Item assignment is provided and `inplace=False`, but the `target`
          does not support the `.copy()` method

    See Also
    --------
    DataFrame.query : Evaluates a boolean expression to query the columns
            of a frame.
    DataFrame.eval : Evaluate a string describing operations on
            DataFrame columns.

    Notes
    -----
    The ``dtype`` of any objects involved in an arithmetic ``%`` operation are
    recursively cast to ``float64``.

    See the :ref:`enhancing performance <enhancingperf.eval>` documentation for
    more details.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> df = md.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
    >>> df.execute()
      animal  age
    0    dog   10
    1    pig   20

    We can add a new column using ``pd.eval``:

    >>> md.eval("double_age = df.age * 2", target=df).execute()
      animal  age  double_age
    0    dog   10          20
    1    pig   20          40
    """
    if not isinstance(expr, str):
        raise TypeError('expr must be a string')

    expr = textwrap.dedent(expr)

    try:
        frame = sys._getframe(level + 1)
        local_dict = local_dict or dict()
        local_dict.update(frame.f_locals)
        global_dict = global_dict or dict()
        global_dict.update(frame.f_globals)
    finally:
        del frame

    env = dict()
    env.update(global_dict)
    env.update(local_dict)

    ref_frames = set(resolvers) | set([target] if target is not None else [])
    self_target = len(resolvers) > 0 and resolvers[0] is target

    if target is not None and not inplace:
        target = target.copy()

    visitor = CollectionVisitor(resolvers, target, env)
    result = visitor.eval(expr)
    result = result if result is not None else target
    has_var_frame = any(isinstance(env[k], ENTITY_TYPE) for k in visitor.referenced_vars)
    if len(ref_frames) != 1 or visitor.entity_subscribe or has_var_frame:
        if parser != 'mars':
            raise NotImplementedError('Does not support parser names other than mars')
        if engine is not None:
            raise NotImplementedError('Does not support specifying engine names')
        return result
    else:
        parser = 'pandas' if parser == 'mars' else parser
        referenced_env = {k: env[k] for k in visitor.referenced_vars}
        op = DataFrameEval(expr, parser=parser, engine=engine, variables=referenced_env,
                           self_target=visitor.assigned and self_target, is_query=False)
        output_type = get_output_types(result)[0]
        dtypes = result.dtypes if result.ndim == 2 else (result.name, result.dtype)
        return op(resolvers[0], output_type, result.shape, dtypes)


def df_eval(df, expr, inplace=False, **kwargs):
    """
    Evaluate a string describing operations on DataFrame columns.

    Operates on columns only, not specific rows or elements.  This allows
    `eval` to run arbitrary code, which can make you vulnerable to code
    injection if you pass user input to this function.

    Parameters
    ----------
    expr : str
        The expression string to evaluate.
    inplace : bool, default False
        If the expression contains an assignment, whether to perform the
        operation inplace and mutate the existing DataFrame. Otherwise,
        a new DataFrame is returned.
    **kwargs
        See the documentation for :func:`eval` for complete details
        on the keyword arguments accepted by
        :meth:`~pandas.DataFrame.query`.

    Returns
    -------
    ndarray, scalar, or pandas object
        The result of the evaluation.

    See Also
    --------
    DataFrame.query : Evaluates a boolean expression to query the columns
        of a frame.
    DataFrame.assign : Can evaluate an expression or function to create new
        values for a column.
    eval : Evaluate a Python expression as a string using various
        backends.

    Notes
    -----
    For more details see the API documentation for :func:`~eval`.
    For detailed examples see :ref:`enhancing performance with eval
    <enhancingperf.eval>`.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> df = md.DataFrame({'A': range(1, 6), 'B': range(10, 0, -2)})
    >>> df.execute()
       A   B
    0  1  10
    1  2   8
    2  3   6
    3  4   4
    4  5   2
    >>> df.eval('A + B').execute()
    0    11
    1    10
    2     9
    3     8
    4     7
    dtype: int64

    Assignment is allowed though by default the original DataFrame is not
    modified.

    >>> df.eval('C = A + B').execute()
       A   B   C
    0  1  10  11
    1  2   8  10
    2  3   6   9
    3  4   4   8
    4  5   2   7
    >>> df.execute()
       A   B
    0  1  10
    1  2   8
    2  3   6
    3  4   4
    4  5   2

    Use ``inplace=True`` to modify the original DataFrame.

    >>> df.eval('C = A + B', inplace=True)
    >>> df.execute()
       A   B   C
    0  1  10  11
    1  2   8  10
    2  3   6   9
    3  4   4   8
    4  5   2   7

    Multiple columns can be assigned to using multi-line expressions:

    >>> df.eval('''
    ... C = A + B
    ... D = A - B
    ... ''').execute()
       A   B   C  D
    0  1  10  11 -9
    1  2   8  10 -6
    2  3   6   9 -3
    3  4   4   8  0
    4  5   2   7  3
    """
    level = kwargs.pop('level', None) or 0
    kwargs['inplace'] = inplace
    val = mars_eval(expr, resolvers=(df,), target=df, level=level + 1, **kwargs)
    if not inplace:
        return val


def df_query(df, expr, inplace=False, **kwargs):
    """
    Query the columns of a DataFrame with a boolean expression.

    Parameters
    ----------
    expr : str
        The query string to evaluate.

        You can refer to variables
        in the environment by prefixing them with an '@' character like
        ``@a + b``.

        You can refer to column names that contain spaces or operators by
        surrounding them in backticks. This way you can also escape
        names that start with a digit, or those that  are a Python keyword.
        Basically when it is not valid Python identifier. See notes down
        for more details.

        For example, if one of your columns is called ``a a`` and you want
        to sum it with ``b``, your query should be ```a a` + b``.

    inplace : bool
        Whether the query should modify the data in place or return
        a modified copy.
    **kwargs
        See the documentation for :func:`eval` for complete details
        on the keyword arguments accepted by :meth:`DataFrame.query`.

    Returns
    -------
    DataFrame
        DataFrame resulting from the provided query expression.

    See Also
    --------
    eval : Evaluate a string describing operations on
        DataFrame columns.
    DataFrame.eval : Evaluate a string describing operations on
        DataFrame columns.

    Notes
    -----
    The result of the evaluation of this expression is first passed to
    :attr:`DataFrame.loc` and if that fails because of a
    multidimensional key (e.g., a DataFrame) then the result will be passed
    to :meth:`DataFrame.__getitem__`.

    This method uses the top-level :func:`eval` function to
    evaluate the passed query.

    The :meth:`~pandas.DataFrame.query` method uses a slightly
    modified Python syntax by default. For example, the ``&`` and ``|``
    (bitwise) operators have the precedence of their boolean cousins,
    :keyword:`and` and :keyword:`or`. This *is* syntactically valid Python,
    however the semantics are different.

    You can change the semantics of the expression by passing the keyword
    argument ``parser='python'``. This enforces the same semantics as
    evaluation in Python space. Likewise, you can pass ``engine='python'``
    to evaluate an expression using Python itself as a backend. This is not
    recommended as it is inefficient compared to using ``numexpr`` as the
    engine.

    The :attr:`DataFrame.index` and
    :attr:`DataFrame.columns` attributes of the
    :class:`~pandas.DataFrame` instance are placed in the query namespace
    by default, which allows you to treat both the index and columns of the
    frame as a column in the frame.
    The identifier ``index`` is used for the frame index; you can also
    use the name of the index to identify it in a query. Please note that
    Python keywords may not be used as identifiers.

    For further details and examples see the ``query`` documentation in
    :ref:`indexing <indexing.query>`.

    *Backtick quoted variables*

    Backtick quoted variables are parsed as literal Python code and
    are converted internally to a Python valid identifier.
    This can lead to the following problems.

    During parsing a number of disallowed characters inside the backtick
    quoted string are replaced by strings that are allowed as a Python identifier.
    These characters include all operators in Python, the space character, the
    question mark, the exclamation mark, the dollar sign, and the euro sign.
    For other characters that fall outside the ASCII range (U+0001..U+007F)
    and those that are not further specified in PEP 3131,
    the query parser will raise an error.
    This excludes whitespace different than the space character,
    but also the hashtag (as it is used for comments) and the backtick
    itself (backtick can also not be escaped).

    In a special case, quotes that make a pair around a backtick can
    confuse the parser.
    For example, ```it's` > `that's``` will raise an error,
    as it forms a quoted string (``'s > `that'``) with a backtick inside.

    See also the Python documentation about lexical analysis
    (https://docs.python.org/3/reference/lexical_analysis.html)
    in combination with the source code in :mod:`pandas.core.computation.parsing`.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> df = md.DataFrame({'A': range(1, 6),
    ...                    'B': range(10, 0, -2),
    ...                    'C C': range(10, 5, -1)})
    >>> df.execute()
       A   B  C C
    0  1  10   10
    1  2   8    9
    2  3   6    8
    3  4   4    7
    4  5   2    6
    >>> df.query('A > B').execute()
       A  B  C C
    4  5  2    6

    The previous expression is equivalent to

    >>> df[df.A > df.B].execute()
       A  B  C C
    4  5  2    6

    For columns with spaces in their name, you can use backtick quoting.

    >>> df.query('B == `C C`').execute()
       A   B  C C
    0  1  10   10

    The previous expression is equivalent to

    >>> df[df.B == df['C C']].execute()
       A   B  C C
    0  1  10   10
    """
    level = kwargs.pop('level', None) or 0
    predicate = mars_eval(expr, resolvers=(df,), level=level + 1, **kwargs)
    result = df[predicate]

    if isinstance(predicate.op, DataFrameEval):
        output_type = get_output_types(result)[0]
        dtypes = result.dtypes if result.ndim == 2 else (result.name, result.dtype)
        result = predicate.op.convert_to_query(df, output_type, result.shape, dtypes)

    if inplace:
        df.data = result.data
    else:
        return result
