from collections import OrderedDict


class ProfilingData:
    _data = {}

    @classmethod
    def init(cls, task_id: str):
        cls._data[task_id] = {
            "general": {},
            "serialization": {},
        }

    @classmethod
    def pop(cls, task_id: str):
        return cls._data.pop(task_id, None)

    @classmethod
    def general(cls, task_id: str):
        task_data = cls._data.get(task_id)
        if task_data is not None:
            return task_data["general"]
        return None

    @classmethod
    def serialization(cls, task_id: str):
        task_data = cls._data.get(task_id)
        if task_data is not None:
            return task_data["serialization"]
        return None
