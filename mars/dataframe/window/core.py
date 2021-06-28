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

from ...serialization.serializables import Serializable, KeyField


class Window(Serializable):
    _input = KeyField('input')

    def __init__(self, input=None, **kw):  # pylint: disable=redefined-builtin
        super().__init__(_input=input, **kw)

    @property
    def input(self):
        return self._input

    @property
    def params(self):
        raise NotImplementedError

    def _repr(self, params):
        kvs = [f'{k}={v}' for k, v in params.items() if v is not None]
        return '{} [{}]'.format(self._repr_name(), ','.join(kvs))

    def _repr_name(self):
        return type(self).__name__

    def __repr__(self):
        return self._repr(self.params)

    def __getitem__(self, item):
        columns = self.input.dtypes.index
        if isinstance(item, (list, tuple)):
            item = list(item)
            for col in item:
                if col not in columns:
                    raise KeyError(f'Column not found: {col}')
        else:
            if item not in columns:
                raise KeyError(f'Column not found: {item}')

        return type(self)(input=self.input[item], **self.params)

    def __getattr__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError:
            if self.input.ndim == 2 and item in self.input.dtypes:
                return self[item]
            else:
                raise

    def __dir__(self):
        result = list(super().__dir__())
        if self.input.ndim == 1:
            return result
        else:
            return sorted(result + [k for k in self.input.dtypes.index
                                    if isinstance(k, str) and k.isidentifier()])
