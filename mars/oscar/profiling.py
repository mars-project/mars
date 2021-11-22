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
        return v


ProfilingData = _ProfilingData()
