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

from collections.abc import Iterable

from ... import opcodes
from ...core import OutputType
from ...serialization.serializables import AnyField
from ..operands import DataFrameOperandMixin, DataFrameOperand
from ..utils import parse_index


class GroupByIndex(DataFrameOperandMixin, DataFrameOperand):
    _op_type_ = opcodes.INDEX
    _op_module_ = 'dataframe.groupby'

    _selection = AnyField('selection')

    def __init__(self, selection=None, output_types=None, **kw):
        super().__init__(_selection=selection, _output_types=output_types, **kw)

    @property
    def selection(self):
        return self._selection

    @property
    def groupby_params(self):
        params = self.inputs[0].op.groupby_params
        params['selection'] = self.selection
        return params

    def build_mock_groupby(self, **kwargs):
        groupby_op = self.inputs[0].op
        return groupby_op.build_mock_groupby(**kwargs)[self.selection]

    def __call__(self, groupby):
        indexed = groupby.op.build_mock_groupby()[self.selection]

        if indexed.ndim == 1:
            self.output_types = [OutputType.series_groupby]
            params = dict(shape=(groupby.shape[0],), name=self.selection,
                          dtype=groupby.dtypes[self.selection],
                          index_value=groupby.index_value,
                          key_dtypes=groupby.key_dtypes)
        else:
            self.output_types = [OutputType.dataframe_groupby]

            if isinstance(self.selection, Iterable) and not isinstance(self.selection, str):
                item_list = list(self.selection)
            else:
                item_list = [self.selection]

            params = groupby.params.copy()
            params['dtypes'] = new_dtypes = groupby.dtypes[item_list]
            params['selection'] = self.selection
            params['shape'] = (groupby.shape[0], len(item_list))
            params['columns_value'] = parse_index(new_dtypes.index, store_data=True)

        return self.new_tileable([groupby], **params)

    @classmethod
    def tile(cls, op: 'GroupByIndex'):
        in_groupby = op.inputs[0]
        out_groupby = op.outputs[0]

        chunks = []
        for c in in_groupby.chunks:
            if op.output_types[0] == OutputType.series_groupby:
                params = dict(shape=(c.shape[0],), name=op.selection,
                              index=(c.index[0],), dtype=c.dtypes[op.selection],
                              index_value=c.index_value, key_dtypes=c.key_dtypes)
            else:
                params = c.params.copy()
                params['dtypes'] = out_groupby.dtypes
                params['selection'] = op.selection
                params['shape'] = (c.shape[0], len(op.selection))
                params['columns_value'] = out_groupby.columns_value

            new_op = op.copy().reset_key()
            chunks.append(new_op.new_chunk([c], **params))

        new_op = op.copy().reset_key()
        params = out_groupby.params.copy()
        new_nsplits = (in_groupby.nsplits[0], (len(op.selection),)) if out_groupby.ndim == 2 \
            else (in_groupby.nsplits[0],)
        params.update(dict(chunks=chunks, nsplits=new_nsplits))
        return new_op.new_tileables([in_groupby], **params)

    @classmethod
    def execute(cls, ctx, op: 'GroupByIndex'):
        in_data = ctx[op.inputs[0].key]
        ctx[op.outputs[0].key] = in_data[op.selection]


def df_groupby_getitem(df_groupby, item):
    try:
        hash(item)
        hashable = True
    except TypeError:
        hashable = False

    if hashable and item in df_groupby.dtypes:
        output_types = [OutputType.series_groupby]
    elif isinstance(item, Iterable) and all(it in df_groupby.dtypes for it in item):
        output_types = [OutputType.dataframe_groupby]
    else:
        raise NameError(f'Cannot slice groupby with {item!r}')

    if df_groupby.selection:
        raise IndexError(f'Column(s) {df_groupby.selection!r} already selected')

    op = GroupByIndex(selection=item, output_types=output_types)
    return op(df_groupby)
