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

from collections import OrderedDict

import pandas as pd

from ...core import ENTITY_TYPE, ExecutableTuple
from ...utils import adapt_mars_docstring


class PlotAccessor:
    def __init__(self, obj):
        self._obj = obj

    def __call__(self, kind='line', session=None, **kwargs):
        to_executes = OrderedDict()
        to_executes['__object__'] = self._obj

        for k, v in kwargs.items():
            if isinstance(v, ENTITY_TYPE):
                to_executes[k] = v

        result = dict()
        executed = ExecutableTuple(
            to_executes.values()).execute().fetch()
        for p, v in zip(to_executes, executed):
            result[p] = v

        data = result.pop('__object__')
        pd_kwargs = kwargs.copy()
        pd_kwargs['kind'] = kind
        pd_kwargs.update(result)

        return data.plot(**pd_kwargs)

    @classmethod
    def _gen_func(cls, name, doc):
        def _inner(self, *args, **kwargs):
            return self(kind=name, *args, **kwargs)

        _inner.__name__ = name
        _inner.__doc__ = doc

        return _inner

    @classmethod
    def _register(cls, method):
        doc = getattr(pd.DataFrame.plot, method).__doc__
        new_doc = adapt_mars_docstring(doc)
        setattr(cls, method, cls._gen_func(method, new_doc))
