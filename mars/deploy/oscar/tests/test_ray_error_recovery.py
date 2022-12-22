import os
import numpy as np
import pandas as pd
import pytest

from ....oscar.errors import ServerClosed
from ....remote import spawn
from ....services.tests.fault_injection_manager import (
    FaultType,
    FaultPosition,
    create_fault_injection_manager,
    ExtraConfigKey,
)
from ....tests.core import require_ray
from ....utils import lazy_import

from .... import tensor as mt
from .... import dataframe as md
from ..ray import new_cluster

ray = lazy_import("ray")

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "local_test_with_ray_config.yml")
FAULT_INJECTION_CONFIG_FILE = os.path.join(
    os.path.dirname(__file__), "fault_injection_config_with_fo.yml"
)


@pytest.fixture
async def fault_ray_cluster(request):
    param = getattr(request, "param", {})
    client = await new_cluster(
        config=param.get("config", CONFIG_FILE),
        worker_num=2,
        worker_cpu=2,
    )
    async with client:
        yield client


@pytest.mark.parametrize(
    "fault_ray_cluster", [{"config": FAULT_INJECTION_CONFIG_FILE}], indirect=True
)
@pytest.mark.parametrize(
    "fault_config",
    [
        [
            FaultType.ProcessExit,
            {FaultPosition.ON_RUN_SUBTASK: 1},
            pytest.raises(ServerClosed),
            ["_UnretryableException", "*"],
        ],
    ],
)
@require_ray
@pytest.mark.asyncio
async def test_node_failover(fault_ray_cluster, fault_config):
    fault_type, fault_count, expect_raises, exception_match = fault_config
    name = await create_fault_injection_manager(
        session_id=fault_ray_cluster.session.session_id,
        address=fault_ray_cluster.session.address,
        fault_count=fault_count,
        fault_type=fault_type,
    )

    columns = list("ABCDEFGHIJ")
    width = len(columns)

    df1 = md.DataFrame(
        mt.random.randint(
            1,
            100,
            size=(100, width),
            chunk_size=(20, width),
        ),
        columns=columns,
    )
    df2 = df1.execute()
    pd1 = df2.to_pandas()

    df3 = df2.rechunk(chunk_size=(10, width))
    df4 = df3.execute()

    def f(x):
        return x + 1

    r = spawn(f, args=(1,), retry_when_fail=False)
    with expect_raises:
        r.execute(extra_config={ExtraConfigKey.FAULT_INJECTION_MANAGER_NAME: name})

    df5 = df4.apply(
        f,
        axis=1,
        dtypes=pd.Series([np.dtype(np.int64)] * width, index=columns),
        output_type="dataframe",
    )
    df6 = df5.execute()
    pd2 = df6.to_pandas()

    pd.testing.assert_frame_equal(pd1 + 1, pd2)
