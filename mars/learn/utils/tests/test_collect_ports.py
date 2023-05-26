# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

from ..collect_ports import collect_ports


def test_collect_ports(setup_cluster):
    session = setup_cluster
    workers = [
        pool.external_address for pool in session._session.client._cluster._worker_pools
    ]
    # make sure assert works inside execution of collect ports
    collect_ports(workers * 2).execute(session=session)
