class DummyOperator:
    @staticmethod
    def set(key, value):
        pass

    @staticmethod
    def inc(key, value):
        pass

    @staticmethod
    def setdefault(key, default=None):
        return DummyOperator

    @staticmethod
    def get(key, default=None):
        pass

    @staticmethod
    def values():
        return []


class ProfilingDataOperator:
    __slots__ = ("_target",)

    def __init__(self, target):
        self._target = target

    def set(self, key, value):
        self._target[key] = value

    def inc(self, key, value):
        old = getattr(self._target, key, 0)
        self._target[key] = old + value

    def setdefault(self, key, default=None):
        r = self._target.setdefault(key, default)
        return DummyOperator if r is None else ProfilingDataOperator(r)

    def get(self, key, default=None):
        r = self._target.get(key, default)
        return DummyOperator if r is None else ProfilingDataOperator(r)

    def values(self):
        return self._target.values()


class _ProfilingData:
    def __init__(self):
        self._data = {}

    def init(self, task_id: str):
        self._data[task_id] = {
            "general": {},
            "serialization": {},
        }

    def pop(self, task_id: str):
        return self._data.pop(task_id, None)

    def __getitem__(self, item):
        key = item if isinstance(item, tuple) else (item,)
        v = None
        d = self._data
        for k in key:
            v = d.get(k, None)
            if v is None:
                break
            else:
                d = v
        return DummyOperator if v is None else ProfilingDataOperator(v)


ProfilingData = _ProfilingData()
