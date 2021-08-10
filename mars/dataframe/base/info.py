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
import pandas as pd
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ...serialization.serializables import StringField, BoolField, Int64Field, KeyField, AnyField
from ...core import OutputType
from ..utils import parse_index
import io


class DataFrameInfo(DataFrameOperand, DataFrameOperandMixin):

    input = KeyField('input')
    verbose = BoolField('verbose')
    buf = AnyField('buf')
    max_cols = Int64Field('max_cols')
    memory_usage = StringField('memory_usage')
    show_counts = BoolField('show_counts')
    null_counts = BoolField('null_counts')

    def __init__(self, verbose=None, buf=None, max_cols=None, memory_usage=None, show_counts=None, null_counts=None, **kw):
        super().__init__(verbose=verbose,
                         buf=buf,
                         max_cols=max_cols,
                         memory_usage=memory_usage,
                         show_counts=show_counts,
                         null_counts=null_counts,
                         **kw)
        self.output_types = [OutputType.series]

    @classmethod
    def get_info_data(cls, op, in_data, chunk_shape):
        info_chunks = []
        chunk_num = chunk_shape[0] * chunk_shape[1]
        for i in range(chunk_num):
            in_chunk = in_data.chunks[i]
            chunk_op = op.copy().reset_key()

            test_df = pd.Series([''])
            index_value = parse_index(test_df.index)

            chunks = chunk_op.new_chunk([in_chunk],
                                        shape=(1,),
                                        output_type=op.output_types[0],
                                        dtype=op.outputs[0].dtype,
                                        index=in_chunk.index,
                                        index_value=index_value)
            info_chunks.append(chunks)
        return info_chunks

    @classmethod
    def tile_info(cls, op, in_data):
        out = op.outputs[0]
        chunk_shape = in_data.chunk_shape

        # stage 1: get info data
        info_chunks = cls.get_info_data(op, in_data, chunk_shape)

        # stage 2: merge data
        chunk_op = DataFrameMergeInfoData(output_types=[OutputType.series], output_type=OutputType.series)
        chk = chunk_op.new_chunk(info_chunks,
                                 shape=(1,),
                                 output_type=op.output_types[0],
                                 dtype=op.outputs[0].dtype,
                                 index=(0,),
                                 index_value=out.index_value)
        # stage 3: print info data
        print_chunk_op = DataFrameInfoPrinter(buf=op.buf,
                                              verbose=op.verbose,
                                              max_cols=op.max_cols,
                                              show_counts=op.show_counts,
                                              null_counts=op.null_counts)
        print_chunks = [print_chunk_op.new_chunk([chk],
                                                 shape=(1,),
                                                 index=(0,),
                                                 index_value=op.outputs[0].index_value,
                                                 dtype=op.outputs[0].dtype,
                                                 name=op.outputs[0].name)]

        new_op = op.copy()
        return new_op.new_seriess(op.inputs,
                                  shape=out.shape,
                                  chunks=print_chunks,
                                  nsplits=((1,),),
                                  index_value=out.index_value,
                                  dtype=out.dtype, name=out.name)

    @classmethod
    def tile(cls, op):
        series = op.inputs[0]
        if len(series.chunks) == 1:
            chunk_op = op.copy().reset_key()
            out_chunks = [chunk_op.new_chunk(series.chunks,
                                             shape=(1,),
                                             index=(0,),
                                             index_value=op.outputs[0].index_value,
                                             dtype=op.outputs[0].dtype,
                                             name=op.outputs[0].name)]

            print_chunk_op = DataFrameInfoPrinter(buf=op.buf,
                                                  verbose=op.verbose,
                                                  max_cols=op.max_cols,
                                                  show_counts=op.show_counts,
                                                  null_counts=op.null_counts)

            print_chunks = [print_chunk_op.new_chunk(out_chunks,
                                                     shape=(1,),
                                                     index=(0,),
                                                     index_value=op.outputs[0].index_value,
                                                     dtype=op.outputs[0].dtype,
                                                     name=op.outputs[0].name)]

            new_op = op.copy()
            kws = op.outputs[0].params.copy()
            kws['nsplits'] = ((1,),)
            kws['chunks'] = print_chunks
            return new_op.new_seriess(op.inputs, **kws)
        else:
            return cls.tile_info(op, series)

    @classmethod
    def execute(cls, ctx, op):
        df = ctx[op.inputs[0].key]
        buf = io.StringIO()
        df.info(buf=buf,
                memory_usage=op.memory_usage)
        info_data = buf.getvalue().split("\n")
        info_data[0] = "<class 'mars.dataframe.DataFrame'>"
        result = "\n".join(info_data)
        ctx[op.outputs[0].key] = pd.Series(result)

    def __call__(self, df):
        test_df = pd.Series([''])
        index_value = parse_index(test_df.index)
        return self.new_series([df], shape=test_df.shape, dtype=test_df.dtype, index_value=index_value)


class DataFrameInfoPrinter(DataFrameOperand, DataFrameOperandMixin):

    buf = AnyField('buf')
    verbose = BoolField('verbose')
    max_cols = Int64Field('max_cols')
    show_counts = BoolField('show_counts')
    null_counts = BoolField('null_counts')

    def __init__(self, buf=None, verbose=None, max_cols=None, show_counts=None, null_counts=None, **kw):
        super().__init__(buf=buf,
                         verbose=verbose,
                         max_cols=max_cols,
                         show_counts=show_counts,
                         null_counts=null_counts,
                         **kw)

        self.output_types = [OutputType.series]

    @classmethod
    def get_total_cols(cls, df_info):
        splited_info = df_info.strip().split("\n")
        return eval(splited_info[2].split()[3])

    @classmethod
    def convert_to_summary(cls, df_info):
        splited_info = df_info.strip().split("\n")
        columns_info = splited_info[5:-2]
        total_columns = len(columns_info)
        first_column = columns_info[0].split()[1]
        last_column = columns_info[-1].split()[1]
        summary_info = [f'Columns: {total_columns} entries, {first_column} to {last_column}']
        summary_info = "\n".join(splited_info[:2] + summary_info + splited_info[-2:]) + "\n"
        return summary_info

    @classmethod
    def not_show_counts(cls, df_info):
        splited_info = df_info.strip().split("\n")
        splited_info[3] = ' #   Column  Dtype'
        splited_info[4] = '---  ------  -----'
        for idx, col_info in enumerate(splited_info[5:-2]):
            index, col_name, _, _, dtype = col_info.strip().split()
            splited_info[idx + 5] = f' {index}   {col_name}       {dtype}'
        return "\n".join(splited_info) + "\n"

    @classmethod
    def execute(cls, ctx, op):
        df_info = ctx[op.inputs[0].key][0]
        total_columns = cls.get_total_cols(df_info)
        if (op.verbose is not None and not op.verbose) or (op.max_cols is not None and total_columns > op.max_cols):
            df_info = cls.convert_to_summary(df_info)
        elif (op.show_counts is not None and not op.show_counts) or (op.null_counts is not None and not op.null_counts):
            df_info = cls.not_show_counts(df_info)
        if op.buf is None:
            print(df_info)
        else:
            op.buf.write(df_info)
        ctx[op.outputs[0].key] = pd.Series([''])


class DataFrameMergeInfoData(DataFrameOperand, DataFrameOperandMixin):

    def __init__(self, **kws):
        super().__init__(**kws)
        self.output_types = [OutputType.series]

    @classmethod
    def execute(cls, ctx, op):
        inputs = [ctx[s.key] for s in op.inputs]
        inputs = [s[0].strip().split('\n') for s in inputs]

        first_row = "<class 'mars.dataframe.DataFrame'>\n"
        raw_columns = []
        for input in inputs:
            raw_columns.extend(input[5:-2])
        splited_columns = [s.split() for s in raw_columns]
        column_names = list(set([s[1] for s in splited_columns]))
        column_to_counts = dict(zip(column_names, [0] * len(column_names)))
        column_to_dtype = dict(zip(column_names, [0] * len(column_names)))
        index_column = []
        for one_col in splited_columns:
            column_to_counts[one_col[1]] += eval(one_col[2])
            column_to_dtype[one_col[1]] = one_col[4]
            if one_col[1] not in index_column:
                index_column.append(one_col[1])

        total_entries = 0
        test_col = column_names[0]
        for input in inputs:
            for one_col in input[5:-2]:
                if one_col.split()[1] == test_col:
                    total_entries += eval(input[1].split()[1])

        index_type = inputs[0][1].split(":")[0]
        index_to_index = dict()
        for input in inputs:
            first_index = input[1].split()[-3]
            last_index = input[1].split()[-1]
            index_to_index[first_index] = last_index
        first_index = list(index_to_index.keys())[0]
        last_index = list(index_to_index.values())[-1]
        total_columns = len(column_names)
        second_row = f'{index_type}: {total_entries}, {first_index} to {last_index}\n'
        third_row = f'Data columns (total {total_columns} columns):\n'
        fourth_row = ' #   Column  Non-Null Count  Dtype\n'
        fifth_row = '---  ------  --------------  -----\n'
        columns_info = []
        for index, column_name in enumerate(index_column):
            counts = column_to_counts[column_name]
            dtype = column_to_dtype[column_name]
            column_info = f' {index}   {column_name}       {counts} not-null      {dtype}\n'
            columns_info.append(column_info)

        dtypes_info = 'dtypes:'
        dtypes_to_count = pd.Series(column_to_dtype.values()).value_counts().to_dict()
        for dtype, count in dtypes_to_count.items():
            dtypes_info += f' {dtype}({count}),'
        dtypes_info = dtypes_info.strip(',')
        dtypes_info += '\n'

        total_memory_usage = 0
        for input in inputs:
            memory_info = input[-1]
            memory_usage = memory_info.split(":")[1]
            if "GB" in memory_usage:
                total_memory_usage += eval(memory_usage[:-2].strip().strip("+")) * 1024 * 1024 * 1024
            elif "MB" in memory_usage:
                total_memory_usage += eval(memory_usage[:-2].strip().strip("+")) * 1024 * 1024
            elif "KB" in memory_usage:
                total_memory_usage += eval(memory_usage[:-2].strip().strip("+")) * 1024
            else:
                total_memory_usage += eval(memory_usage[:-5].strip().strip("+"))
        contain_object = '+' if 'object' in column_to_dtype.values() else ''
        if total_memory_usage < 1024:
            last_row = f'memory usage: {total_memory_usage}{contain_object} bytes\n'
        elif total_memory_usage < 1024 * 1024:
            total_memory_usage = round(total_memory_usage / 1024, 1)
            last_row = f'memory usage: {total_memory_usage}{contain_object} KB\n'
        elif total_memory_usage < 1024 * 1024 * 1024:
            total_memory_usage = round(total_memory_usage / (1024 * 1024), 1)
            last_row = f'memory usage: {total_memory_usage}{contain_object} MB\n'
        else:
            total_memory_usage = round(total_memory_usage / (1024 * 1024 * 1024), 1)
            last_row = f'memory usage: {total_memory_usage}{contain_object} GB\n'

        result = first_row + second_row + third_row + fourth_row + fifth_row
        for column_info in columns_info:
            result += column_info
        result += dtypes_info + last_row
        ctx[op.outputs[0].key] = pd.Series(result)


def info(arg, verbose=None, buf=None, max_cols=None, memory_usage=None, show_counts=None, null_counts=None):

    op = DataFrameInfo(verbose=verbose,
                       buf=buf,
                       max_cols=max_cols,
                       memory_usage=memory_usage,
                       show_counts=show_counts,
                       null_counts=null_counts)
    op(arg).execute()
