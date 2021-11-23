from ..profiling import ProfilingData


def test_profiling_data():
    ProfilingData.init("abc")
    try:
        for n in ["general", "serialization"]:
            assert n in ProfilingData["abc"]
            assert type(ProfilingData["abc", n]) == dict
        assert ProfilingData["def"] is None
        assert ProfilingData["abc", "def"] is None
        assert ProfilingData["abc", "def", 1] is None
    finally:
        ProfilingData.pop("abc")
