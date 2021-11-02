.. _operand_implementation:

Implement a Mars Operand
========================

Use ``read_csv`` as an example to illustrate how to implement a Mars operand.

Define Operand Class
--------------------

All Mars operands inherit from the base class ``Operand``, it defines the
basic properties of operand, each module has it's own child class, such as
``DataFrameOperand``, ``TensorOperand``, etc. For tilebale operand, it also
needs to inherit from ``TileableOperandMixin`` to implement ``tile`` and ``execute``
functions. So we firstly define operand class and set output types in init method,
the types could be DataFrame or Tensor which depends on the type of operand's output data,
``__call__`` method is also needed for creating a Mars dataframe.

.. code-block:: python

    # NOTE: Use relative import if in Mars modules
    from mars.dataframe.operands import DataFrameOperand, DataFrameOperandMixin
    from mars.core import OutputType
    from mars.serialization.serializables import StringField


    class SimpleReadCSV(DataFrameOperand, DataFrameOperandMixin):
        path = StringField('path')

        def __init__(self, **kw):
            super().__init__(**kw)
            self._output_types=[OutputType.dataframe]

        def __call__(self, index_value=None, columns_value=None,
                     dtypes=None, chunk_bytes=None):
            # np.nan means its size is unknown on axis 0
            shape = (np.nan, len(dtypes))
            return self.new_dataframe(
                None,
                shape,
                dtypes=dtypes,
                index_value=index_value,
                columns_value=columns_value,
                chunk_bytes=chunk_bytes,
            )

For the ``SimpleReadCSV`` operand, the property ``path`` means the path of csv file,
we use a ``StringField`` to indicate the property's type which is useful for serialization.
If the type is uncertain, ``AnyField`` will work.

Implement Tile Method
---------------------

Tile method is the next goal, this method will split the computing task into
several sub tasks. Ideally, these tasks can be assigned on different executors
in parallel. In the specific case of ``read_csv``, each sub task read a block of bytes
from the file, so we need calculate the offset and length of each block in the
tile function. As we use the same class for both coarse-grained and fine-grained operand,
``offset``, ``length`` and other properties are added to record information for
fine-grained operand.

.. code-block:: python

    import os

    import numpy as np

    from mars.core import OutputType
    from mars.dataframe.operands import DataFrameOperand, DataFrameOperandMixin
    from mars.serialization.serializables import AnyField, StringField, Int64Field
    from mars.utils import parse_readable_size


    class SimpleReadCSV(DataFrameOperand, DataFrameOperandMixin):
        path = StringField("path")
        chunk_bytes = AnyFiled('chunk_bytes')
        offset = Int64Field("offset")
        length = Int64Field("length")

        def __init__(self, **kw):
            super().__init__(**kw)
            self._output_types=[OutputType.dataframe]

        @classmethod
        def tile(cls, op: "SimpleReadCSV"):
            # out is operand's output in coarse-grained graph
            out = op.outputs[0]

            file_path = op.path
            file_size = os.path.getsize(file_path)

            # split file into chunks
            chunk_bytes = int(parse_readable_size(op.chunk_bytes)[0])
            offset = 0
            index_num = 0
            out_chunks = []
            while offset < file_size:
                chunk_op = op.copy().reset_key()
                chunk_op.path = file_path
                # offset and length for current chunk
                chunk_op.offset = offset
                chunk_op.length = min(chunk_bytes, file_size - offset)
                # calculate chunk's meta, including shape, index_value, columns_value
                # here we use np.nan to represent unknown shape
                shape = (np.nan, len(out.dtypes))
                # use `op.new_chunk` to create a dataframe chunk
                new_chunk = chunk_op.new_chunk(
                    None,
                    shape=shape,
                    index=(index_num, 0),
                    index_value=out.index_value,
                    columns_value=out.columns_value,
                    dtypes=out.dtypes,
                )
                offset += chunk_bytes
                index_num += 1
                out_chunks.append(new_chunk)

            # create a new tileable which holds `chunks` for generating fine-grained graph
            new_op = op.copy()
            # `nsplits` records the split info for each axis. For read_csv,
            # the output dataframe is split into multiple parts on axis 0 and
            # keep one chunk on axis 1, so the nsplits will be
            # like ((np.nan, np.nan, ...), (out.shape[1],))
            nsplits = ((np.nan,) * len(out_chunks), (out.shape[1],))
            return new_op.new_dataframes(
                None,
                out.shape,
                dtypes=out.dtypes,
                index_value=out.index_value,
                columns_value=out.columns_value,
                chunks=out_chunks,
                nsplits=nsplits,
            )


Implement Execute Method
------------------------

When sub task is delivered to executor, Mars will call operand's execute method to
perform calculations. When it comes to ``read_csv``, we need read the block from the file
according to the ``offset`` and ``length``, however the ``offset`` is a rough position as
we can't read a csv file from the middle of a line, using ``readline`` to find the starts
and ends at delimiter boundaries.

.. code-block:: python

    from io import BytesIO

    import pandas as pd

    from mars.dataframe.utils import build_empty_df


    def _find_chunk_start_end(f, offset, size):
        f.seek(offset)
        if f.tell() == 0:
            start = 0
        else:
            f.readline()
            start = f.tell()
        f.seek(offset + size)
        f.readline()
        end = f.tell()
        return start, end


    class SimpleReadCSV(DataFrameOperand, DataFrameOperandMixin):
        @classmethod
        def execute(cls, ctx: Union[dict, Context], op: "SimpleReadCSV"):
            out = op.outputs[0]
            with open(op.path, 'rb') as f:
                start, end = _find_chunk_start_end(f, op.offset, op.length)
                if end == start:
                    # the last chunk may be empty
                    data = build_empty_df(out.dtypes)
                else:
                    f.seek(start)
                    if start == 0:
                        # The first chunk contains header, skip header rows
                        data = pd.read_csv(BytesIO(f.read(end - start)),
                                           skiprows=1,
                                           names=out.dtypes.index)
                    else:
                        data = pd.read_csv(BytesIO(f.read(end - start)),
                                           names=out.dtypes.index)

            ctx[out.key] = data

After reading the chunk data by ``pd.read_csv``, we store the results in ``ctx``.
``SimpleReadCSV`` only has one output here, for operand like ``SVD`` that has multiple
outputs, we can store them separately using output's keys.

Define User Interface
---------------------

Finally, we need define function ``read_csv`` exposed to users. In this function, besides
creating a ``SimpleReadCSV`` operand, a sample data is taken to infer some meta information
of Mars DataFrame, such as dtypes, columns, index, etc.

.. code-block:: python

    from mars.dataframe.utils import parse_index

    def read_csv(file_path, chunk_bytes='16M'):
        # use first 10 lines to infer
        with open(file_path, 'rb') as f:
            head_lines = b"".join([f.readline() for _ in range(10)])

        mini_df = pd.read_csv(BytesIO(head_lines))
        index_value = parse_index(mini_df.index)
        columns_value = parse_index(mini_df.columns, store_data=True)
        dtypes = mini_df.dtypes
        op = SimpleReadCSV(path=file_path, chunk_bytes=chunk_bytes)
        return op(
            index_value=index_value,
            columns_value=columns_value,
            dtypes=dtypes,
            chunk_bytes=chunk_bytes,
        )

Write Tests
-----------

Mars uses pytest for testing, we can add tests under the ``tests`` subdirectory
of the specific module and follow the current examples of tests. Define a test
function and use the shared fixture ``setup`` to run your tests under the test
environment.

.. code-block:: python

    def test_simple_read_csv_execution(setup):
        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, "test.csv")
            # write to a csv file
            raw = pd.DataFrame({
                'int': range(10),
                'float': np.random.rand(10),
                'str': [f'value_{i}' for i in range(10)]
            }).to_csv(file_path, index=False)
            mdf = read_csv(file_path).execute().fetch()
            pd.testing.assert_frame_equal(raw, mdf)

When tests pass locally, we can submit a pull requests on GitHub, the test suite
will run automatically on GitHub Actions and Azure Pipelines continuous integration
services, if all checks have passed, it means the pull request is up to the quality
of merging.

Documenting Your Code
---------------------

If the changes add APIs to Mars modules, we should document our code in ``docs``
directory, it can be done following the regarding :ref:`documentation <build_documentation>`.
