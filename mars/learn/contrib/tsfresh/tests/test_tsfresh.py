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

import pytest

try:
    import tsfresh
    from tsfresh import extract_features
    from tsfresh.examples import robot_execution_failures
    from tsfresh.feature_extraction import ComprehensiveFCParameters
    from tsfresh.utilities.dataframe_functions import impute
except ImportError:
    tsfresh = None

from .....deploy.oscar.session import new_session, get_default_session
from .. import MarsDistributor


@pytest.mark.skipif(tsfresh is None, reason="tsfresh not installed")
def test_distributed_ts_fresh(setup):
    robot_execution_failures.download_robot_execution_failures()
    df, y = robot_execution_failures.load_robot_execution_failures()
    default_session = get_default_session()
    sync_session = new_session(default_session.address, default=False)
    dist = MarsDistributor(session=sync_session)

    df = df.iloc[:200].copy()

    extraction_settings = ComprehensiveFCParameters()
    extract_features(
        df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=extraction_settings,
        # we impute = remove all NaN features automatically
        impute_function=impute,
        distributor=dist,
    )
