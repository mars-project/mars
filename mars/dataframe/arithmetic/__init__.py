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
from .subtract import subtract, rsubtract, DataFrameSubtract
from .floordiv import floordiv, rfloordiv, DataFrameFloorDiv
from .truediv import truediv, rtruediv, DataFrameTrueDiv


def _install():
    from ..core import DataFrame, Series

    for entity in (DataFrame, Series):
        setattr(entity, 'abs', abs)

        setattr(entity, '__add__', wrap_notimplemented_exception(add))
        setattr(entity, '__radd__', wrap_notimplemented_exception(radd))
        setattr(entity, 'add', add)
        setattr(entity, 'radd', radd)

        setattr(entity, '__sub__', wrap_notimplemented_exception(subtract))
        setattr(entity, '__rsub__', wrap_notimplemented_exception(rsubtract))
        setattr(entity, 'sub', subtract)
        setattr(entity, 'rsub', rsubtract)

        setattr(entity, '__floordiv__', wrap_notimplemented_exception(floordiv))
        setattr(entity, '__rfloordiv__', wrap_notimplemented_exception(rfloordiv))
        setattr(entity, '__truediv__', wrap_notimplemented_exception(truediv))
        setattr(entity, '__rtruediv__', wrap_notimplemented_exception(rtruediv))
        setattr(entity, '__div__', wrap_notimplemented_exception(truediv))
        setattr(entity, '__rdiv__', wrap_notimplemented_exception(rtruediv))
        setattr(entity, 'floordiv', floordiv)
        setattr(entity, 'rfloordiv', rfloordiv)
        setattr(entity, 'truediv', truediv)
        setattr(entity, 'rtruediv', rtruediv)
        setattr(entity, 'div', truediv)
        setattr(entity, 'rdiv', rtruediv)


_install()
del _install
