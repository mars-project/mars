import pytest
from ..profiling import ProfilingData, ProfilingDataOperator, DummyOperator
from ...tests.core import check_dict_structure_same


def test_profiling_data():
    ProfilingData.init("abc")
    try:
        for n in ["general", "serialization"]:
            assert isinstance(ProfilingData["abc", n], ProfilingDataOperator)
        assert ProfilingData["def"] is DummyOperator
        assert ProfilingData["abc", "def"] is DummyOperator
        assert ProfilingData["abc", "def", 1] is DummyOperator
        ProfilingData["def"].set("a", 1)
        ProfilingData["def"].inc("b", 1)
        assert sum(ProfilingData["def"].nest("a").values()) == 0
        ProfilingData["abc", "serialization"].set("a", 1)
        ProfilingData["abc", "serialization"].inc("b", 1)
        with pytest.raises(TypeError):
            assert ProfilingData["abc", "serialization"].nest("a")
        assert sum(ProfilingData["abc", "serialization"].nest("c").values()) == 0
    finally:
        v = ProfilingData.pop("abc")
        check_dict_structure_same(
            v, {"general": {}, "serialization": {"a": 1, "b": 1, "c": {}}}
        )
