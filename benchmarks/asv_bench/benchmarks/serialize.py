# Copyright 1999-2022 Alibaba Group Holding Ltd.
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

import cloudpickle
import numpy as np
import pandas as pd

from mars.core.operand import MapReduceOperand
from mars.dataframe.operands import DataFrameOperandMixin
from mars.oscar.core import ActorRef

from mars.oscar.backends.message import SendMessage, new_message_id, ActorRefMessage
from mars.serialization import serialize, deserialize
from mars.serialization.serializables import (
    Serializable,
    Int64Field,
    Float64Field,
    ListField,
    DataTypeField,
    SeriesField,
    NDArrayField,
    StringField,
    FieldTypes,
    BoolField,
    Int32Field,
    Float32Field,
    SliceField,
    Datetime64Field,
    Timedelta64Field,
    TupleField,
    DictField,
    Complex64Field,
    Complex128Field,
)
from mars.services.subtask import Subtask, SubtaskResult, SubtaskStatus
from mars.services.task import new_task_id
from mars.utils import tokenize

# do warm up
serialize(None)


class SerializableChild(Serializable):
    str_field = StringField("str_field")
    int_field = Int64Field("int_field")
    float_field = Float64Field("float_field")
    dtype_field = DataTypeField("dtype_field")
    series_field = SeriesField("series_field")
    ndarray_field = NDArrayField("ndarray_field")
    int_list_field = ListField("int_list_field", field_type=FieldTypes.int64)
    float_list_field = ListField("float_list_field", field_type=FieldTypes.float64)
    str_list_field = ListField("str_list_field", field_type=FieldTypes.string)


class SerializableParent(Serializable):
    children = ListField("children", field_type=FieldTypes.reference)


class MySerializable(Serializable):
    _bool_val = BoolField("f1")
    _int32_val = Int32Field("f2")
    _int64_val = Int64Field("f3")
    _float32_val = Float32Field("f4")
    _float64_val = Float64Field("f5")
    _complex64_val = Complex64Field("f6")
    _complex128_val = Complex128Field("f7")
    _string_val = StringField("f8")
    _datetime64_val = Datetime64Field("f9")
    _timedelta64_val = Timedelta64Field("f10")
    _datatype_val = DataTypeField("f11")
    _slice_val = SliceField("f12")
    _list_val = ListField("list_val", FieldTypes.int64)
    _tuple_val = TupleField("tuple_val", FieldTypes.string)
    _dict_val = DictField("dict_val", FieldTypes.string, FieldTypes.bytes)


class SerializeSerializableSuite:
    def setup(self):
        children = []
        for idx in range(1000):
            child = SerializableChild(
                str_field="abcd" * 1024,
                int_field=idx,
                float_field=float(idx) * 1.42,
                dtype_field=np.dtype("<M8"),
                series_field=pd.Series([np.dtype(int)] * 1024, name="dtype"),
                ndarray_field=np.random.rand(1000),
                int_list_field=np.random.randint(0, 1000, size=(1000,)).tolist(),
                float_list_field=np.random.rand(1000).tolist(),
                str_list_field=[str(i * 2.8571) for i in range(100)],
            )
            children.append(child)
        self.test_data = SerializableParent(children=children)

    def time_serialize_deserialize(self):
        deserialize(*serialize(self.test_data))


class SerializeSubtaskSuite:
    def setup(self):
        self.subtasks = []
        for i in range(10):
            subtask = Subtask(
                subtask_id=new_task_id(),
                stage_id=new_task_id(),
                logic_key=new_task_id(),
                session_id=new_task_id(),
                task_id=new_task_id(),
                chunk_graph=None,
                expect_bands=[
                    ("ray://mars_cluster_1649927648/17/0", "numa-0"),
                ],
                bands_specified=False,
                virtual=False,
                priority=(1, 0),
                retryable=True,
                extra_config={},
            )
            self.subtasks.append(subtask)

    def time_pickle_serialize_deserialize_subtask(self):
        for i in range(20000):
            deserialize(*cloudpickle.loads(cloudpickle.dumps(serialize(self.subtasks))))


class SerializePrimitivesSuite:
    def setup(self):
        self.test_primitive_serializable = []
        for i in range(10000):
            my_serializable = MySerializable(
                _bool_val=True,
                _int32_val=-32,
                _int64_val=-64,
                _float32_val=np.float32(2.0),
                _float64_val=2.0,
                _complex64_val=np.complex64(1 + 2j),
                _complex128_val=1 + 2j,
                _string_val="string_value",
                _datetime64_val=pd.Timestamp(123),
                _timedelta64_val=pd.Timedelta(days=1),
                _datatype_val=np.dtype(np.int32),
                _slice_val=slice(1, 10, 2),
                _list_val=[1, 2],
                _tuple_val=("a", "b"),
                _dict_val={"a": b"bytes_value"},
            )
            self.test_primitive_serializable.append(my_serializable)

    def time_serialize_deserialize_primitive(self):
        deserialize(*serialize(self.test_primitive_serializable))

    def time_pickle_serialize_deserialize_basic(self):
        deserialize(
            *cloudpickle.loads(
                cloudpickle.dumps(serialize(self.test_primitive_serializable))
            )
        )


class SerializeContainersSuite:
    def setup(self):
        self.test_list = list(range(100000))
        self.test_tuple = tuple(range(100000))
        self.test_dict = {i: i for i in range(100000)}

    def time_pickle_serialize_deserialize_list(self):
        deserialize(*cloudpickle.loads(cloudpickle.dumps(serialize(self.test_list))))

    def time_pickle_serialize_deserialize_tuple(self):
        deserialize(*cloudpickle.loads(cloudpickle.dumps(serialize(self.test_tuple))))

    def time_pickle_serialize_deserialize_dict(self):
        deserialize(*cloudpickle.loads(cloudpickle.dumps(serialize(self.test_dict))))


class MockDFOperand(MapReduceOperand, DataFrameOperandMixin):
    _op_type_ = 14320


class SerializeFetchShuffleSuite:
    def setup(self):
        from mars.core import OutputType
        from mars.core.operand import OperandStage
        from mars.dataframe.operands import DataFrameShuffleProxy

        source_chunks = []
        for i in range(1000):
            op = MockDFOperand(
                _output_types=[OutputType.dataframe],
                _key=tokenize(i),
                stage=OperandStage.map,
            )
            source_chunks.append(op.new_chunk([], index=(i,)))

        shuffle_chunk = DataFrameShuffleProxy(
            output_types=[OutputType.dataframe]
        ).new_chunk(source_chunks)

        from mars.utils import build_fetch_shuffle

        fetch_chunk = build_fetch_shuffle(shuffle_chunk, n_reducers=10)

        self.test_fetch_chunks = []
        for i in range(1000):
            reduce_op = MockDFOperand(
                _output_types=[OutputType.dataframe],
                _key=tokenize((i, 1)),
                stage=OperandStage.reduce,
            )
            self.test_fetch_chunks.append(
                reduce_op.new_chunk([fetch_chunk], index=(i,))
            )

    def time_pickle_serialize_fetch_shuffle_chunks(self):
        for fetch_chunk in self.test_fetch_chunks:
            header, buffers = serialize(fetch_chunk)
            serialized = cloudpickle.dumps((header, buffers))
            deserialize(*cloudpickle.loads(serialized))


class SerializeMessageSuite:
    def setup(self):
        self.send_messages = []
        self.actor_ref_messages = []
        for i in range(10000):
            ref = ActorRef(
                "ray://mars_cluster_1649927648/17/0",
                b"F20Wyerq6EiqltB8jAVs7L3N_task_manager",
            )
            new_result = SubtaskResult(
                subtask_id=new_task_id(),
                session_id=new_task_id(),
                task_id=new_task_id(),
                stage_id=new_task_id(),
                status=SubtaskStatus.succeeded,
                progress=1.0,
                data_size=1000000.0,
                bands=[("ray://mars_cluster_1649927648/17/0", "numa-0")],
                execution_start_time=1646125099.622051,
                execution_end_time=1646125104.448726,
            )
            send_message = SendMessage(
                new_message_id(),
                ref,
                new_result,
                protocol=0,
            )
            self.send_messages.append(send_message)
            actor_ref_message = ActorRefMessage(
                message_id=new_message_id(),
                actor_ref=ref,
                protocol=0,
            )
            self.actor_ref_messages.append(actor_ref_message)

    def time_pickle_serialize_deserialize_send_messages(self):
        deserialize(
            *cloudpickle.loads(cloudpickle.dumps(serialize(self.send_messages)))
        )

    def time_pickle_serialize_deserialize_actor_ref_messages(self):
        deserialize(
            *cloudpickle.loads(cloudpickle.dumps(serialize(self.actor_ref_messages)))
        )


if __name__ == '__main__':
    subtask_suite = SerializeSubtaskSuite()
    subtask_suite.setup()
    import time
    start = time.time_ns()
    subtask_suite.time_pickle_serialize_deserialize_subtask()
    print(f"Serialize cost {(time.time_ns() - start) / 1000_000}")
