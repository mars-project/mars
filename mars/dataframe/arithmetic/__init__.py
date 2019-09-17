# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

from ..utils import wrap_notimplemented_exception
from .abs import abs, DataFrameAbs
from .add import add, radd, DataFrameAdd
from .div import div, rdiv, DataFrameDiv

def _install():
    from ..core import DataFrame, Series
    setattr(DataFrame, 'abs', abs)
    setattr(Series, 'abs', abs)

    setattr(DataFrame, '__add__', wrap_notimplemented_exception(add))
    setattr(DataFrame, '__radd__', wrap_notimplemented_exception(radd))
    setattr(DataFrame, 'add', add)
    setattr(DataFrame, 'radd', radd)
    setattr(Series, '__add__', wrap_notimplemented_exception(add))
    setattr(Series, '__radd__', wrap_notimplemented_exception(radd))
    setattr(Series, 'add', add)
    setattr(Series, 'radd', radd)

    setattr(DataFrame, '__div__', wrap_notimplemented_exception(div))
    setattr(DataFrame, '__rdiv__', wrap_notimplemented_exception(rdiv))
    setattr(DataFrame, 'div', div)
    setattr(DataFrame, 'rdiv', rdiv)
    setattr(Series, '__div__', wrap_notimplemented_exception(div))
    setattr(Series, '__rdiv__', wrap_notimplemented_exception(rdiv))
    setattr(Series, 'div', div)
    setattr(Series, 'rdiv', rdiv)


_install()
del _install
