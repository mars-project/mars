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

import numpy as np
import pandas as pd

from ... import opcodes
from ...core import OutputType, get_output_types, recursive_tile
from ...serialization.serializables import DictField, Int64Field
from ..core import IndexValue
from ..operands import DataFrameOperandMixin, DataFrameOperand
from ..utils import build_concatenated_rows_frame, parse_index


class GroupByHead(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.GROUPBY_HEAD
    _op_module_ = 'dataframe.groupby'

    _row_count = Int64Field('row_count')
    _groupby_params = DictField('groupby_params')

    def __init__(self, row_count=None, groupby_params=None, **kw):
        super().__init__(_row_count=row_count, _groupby_params=groupby_params, **kw)

    @property
    def row_count(self) -> int:
        return self._row_count

    @property
    def groupby_params(self) -> dict:
        return self._groupby_params

    def __call__(self, groupby):
        df = groupby
        while df.op.output_types[0] not in (OutputType.dataframe, OutputType.series):
            df = df.inputs[0]

        selection = groupby.op.groupby_params.pop('selection', None)
        if df.ndim > 1 and selection:
            if isinstance(selection, tuple) and selection not in df.dtypes:
                selection = list(selection)

            result_df = df[selection]
        else:
            result_df = df

        self._output_types = [OutputType.dataframe] if result_df.ndim == 2 else [OutputType.series]

        params = result_df.params
        params['shape'] = (np.nan,) + result_df.shape[1:]
        if isinstance(df.index_value.value, IndexValue.RangeIndex):
            params['index_value'] = parse_index(pd.RangeIndex(-1), df.key)

        return self.new_tileable([df], **params)

    @classmethod
    def tile(cls, op: 'GroupByHead'):
        in_df = op.inputs[0]
        groupby_params = op.groupby_params.copy()
        selection = groupby_params.pop('selection', None)

        if len(in_df.shape) > 1:
            in_df = build_concatenated_rows_frame(in_df)
        out_df = op.outputs[0]

        # when row_count is not positive or there is only one chunk,
        # tile with a single chunk
        if op.row_count <= 0 or len(in_df.chunks) == 0:
            c = in_df.chunks[0]
            chunk_op = op.copy().reset_key()
            params = c.params
            row_num = 0 if op.row_count <= 0 else np.nan
            params['shape'] = (row_num,) + c.shape[1:]
            params['index_value'] = out_df.index_value
            out_chunk = chunk_op.new_chunk([c], **params)

            tileable_op = op.copy().reset_key()
            params = out_df.params
            params['shape'] = (row_num,) + c.shape[1:]
            params['index_value'] = out_df.index_value
            return tileable_op.new_tileables([in_df], nsplits=((row_num,), ) + in_df.nsplits[1:],
                                             chunks=[out_chunk], **params)

        if in_df.ndim > 1 and selection:
            if isinstance(selection, tuple) and selection not in in_df.dtypes:
                selection = list(selection)

            if not isinstance(selection, list):
                pre_selection = [selection]
            else:
                pre_selection = list(selection)

            if isinstance(groupby_params.get('by'), list):
                pre_selection += [el for el in groupby_params['by'] if el not in pre_selection]

            if len(pre_selection) != in_df.shape[1]:
                in_df = yield from recursive_tile(in_df[pre_selection])

        # generate pre chunks
        pre_chunks = []
        for c in in_df.chunks:
            pre_op = op.copy().reset_key()
            pre_op._output_types = get_output_types(c)
            pre_op._groupby_params = op.groupby_params.copy()
            pre_op._groupby_params.pop('selection', None)
            params = c.params
            params['shape'] = (np.nan,) + c.shape[1:]
            pre_chunks.append(pre_op.new_chunk([c], **params))

        new_op = op.copy().reset_key()
        new_op._output_types = get_output_types(in_df)
        new_nsplits = ((np.nan,) * len(in_df.nsplits[0]),) + in_df.nsplits[1:]
        pre_tiled = new_op.new_tileable([in_df], chunks=pre_chunks,
                                        nsplits=new_nsplits, **in_df.params)

        # generate groupby
        grouped = yield from recursive_tile(pre_tiled.groupby(**groupby_params))
        if selection:
            grouped = yield from recursive_tile(grouped[selection])

        # generate post chunks
        post_chunks = []
        for c in grouped.chunks:
            post_op = op.copy().reset_key()
            post_op._groupby_params = op.groupby_params.copy()
            post_op._groupby_params.pop('selection', None)
            if op.output_types[0] == OutputType.dataframe:
                index = c.index
            else:
                index = (c.index[0],)
            params = out_df.params
            params['index'] = index
            post_chunks.append(post_op.new_chunk([c], **params))

        new_op = op.copy().reset_key()
        new_nsplits = ((np.nan,) * len(in_df.nsplits[0]),)
        if out_df.ndim > 1:
            new_nsplits += ((out_df.shape[1],),)
        return new_op.new_tileables([in_df], chunks=post_chunks,
                                    nsplits=new_nsplits, **out_df.params)

    @classmethod
    def execute(cls, ctx, op: 'GroupByHead'):
        in_data = ctx[op.inputs[0].key]

        params = op.groupby_params.copy()
        selection = params.pop('selection', None)

        if hasattr(in_data, 'groupby'):
            grouped = in_data.groupby(**params)
        else:
            grouped = in_data

        if selection:
            grouped = grouped[selection]

        ctx[op.outputs[0].key] = grouped.head(op.row_count)


def head(groupby, n=5):
    """
    Return first n rows of each group.

    Similar to ``.apply(lambda x: x.head(n))``, but it returns a subset of rows
    from the original Series or DataFrame with original index and order preserved
    (``as_index`` flag is ignored).

    Does not work for negative values of `n`.

    Returns
    -------
    Series or DataFrame

    See Also
    --------
    Series.groupby
    DataFrame.groupby

    Examples
    --------

    >>> import mars.dataframe as md
    >>> df = md.DataFrame([[1, 2], [1, 4], [5, 6]],
    ...                   columns=['A', 'B'])
    >>> df.groupby('A').head(1).execute()
       A  B
    0  1  2
    2  5  6
    >>> df.groupby('A').head(-1).execute()
    Empty DataFrame
    Columns: [A, B]
    Index: []
    """
    groupby_params = groupby.op.groupby_params.copy()
    groupby_params.pop('as_index', None)

    op = GroupByHead(row_count=n, groupby_params=groupby_params)
    return op(groupby)
