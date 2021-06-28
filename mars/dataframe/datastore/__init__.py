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


def _install():
    from .to_csv import to_csv
    from .to_sql import to_sql
    from .to_parquet import to_parquet
    from ..operands import DATAFRAME_TYPE, SERIES_TYPE

    for cls in DATAFRAME_TYPE:
        setattr(cls, 'to_csv', to_csv)
        setattr(cls, 'to_sql', to_sql)
        setattr(cls, 'to_parquet', to_parquet)

    for cls in SERIES_TYPE:
        setattr(cls, 'to_csv', to_csv)
        setattr(cls, 'to_sql', to_sql)


_install()
del _install
