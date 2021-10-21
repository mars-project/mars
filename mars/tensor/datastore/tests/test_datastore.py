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

import shutil
import tempfile

import numpy as np
import pytest

try:
    import tiledb
except (ImportError, OSError):  # pragma: no cover
    tiledb = None

from ....core import tile
from ... import random
from .. import totiledb
from ..utils import get_tiledb_schema_from_tensor, check_tiledb_array_with_tensor


@pytest.mark.skipif(tiledb is None, reason="TileDB not installed")
def test_get_tile_db_schema():
    ctx = tiledb.Ctx()

    nsplits = ((1, 2), (3, 1), (2, 2, 1))
    a = random.rand(3, 4, 5, dtype=np.float64, chunk_size=nsplits)
    schema = get_tiledb_schema_from_tensor(a, ctx, nsplits)
    assert schema.ndim == 3
    assert schema.shape == (3, 4, 5)
    assert [schema.domain.dim(i).tile for i in range(a.ndim)] == [2, 3, 2]
    assert schema.attr(0).dtype == a.dtype


@pytest.mark.skipif(tiledb is None, reason="TileDB not installed")
def test_check_tile_db():
    ctx = tiledb.Ctx()

    tempdir = tempfile.mkdtemp()
    try:
        np_a = np.random.rand(2, 3)
        tiledb_a = tiledb.DenseArray.from_numpy(ctx=ctx, uri=tempdir, array=np_a)

        with pytest.raises(ValueError):
            # ndim not match
            check_tiledb_array_with_tensor(random.rand(2, 3, 4), tiledb_a)

        with pytest.raises(ValueError):
            # shape not matchn
            check_tiledb_array_with_tensor(random.rand(2, 4), tiledb_a)

        with pytest.raises(ValueError):
            # dtype not match
            check_tiledb_array_with_tensor(
                random.rand(2, 3, dtype=np.float32), tiledb_a
            )

        # legal
        check_tiledb_array_with_tensor(random.rand(2, 3), tiledb_a)
    finally:
        shutil.rmtree(tempdir)


@pytest.mark.skipif(tiledb is None, reason="TileDB not installed")
def test_store_tile_db():
    ctx = tiledb.Ctx()
    tempdir = tempfile.mkdtemp()
    try:
        t = random.rand(50, 30, chunk_size=13)
        t2 = t + 1

        saved = totiledb(tempdir, t2)
        assert saved.shape == (0, 0)
        assert saved.op.tiledb_config is None
        assert saved.op.tiledb_uri == tempdir

        with pytest.raises(tiledb.TileDBError):
            tiledb.DenseArray(ctx=ctx, uri=tempdir)

        # tiledb array is created in the tile
        saved = tile(saved)

        # no error
        tiledb.DenseArray(ctx=ctx, uri=tempdir)

        # TileDB consolidation
        assert len(saved.chunks) == 1

        assert saved.chunks[0].inputs[0].op.axis_offsets == (0, 0)
        assert saved.chunks[0].inputs[1].op.axis_offsets == (0, 13)
        assert saved.chunks[0].inputs[2].op.axis_offsets == (0, 26)  # input (0, 2)
        assert saved.chunks[0].inputs[5].op.axis_offsets == (13, 26)  # input (1, 2)
        assert saved.chunks[0].inputs[11].op.axis_offsets == (39, 26)  # input (3, 2)

        with pytest.raises(ValueError):
            t3 = random.rand(30, 50)
            totiledb(tempdir, t3, ctx=ctx)  # shape incompatible
    finally:
        shutil.rmtree(tempdir)
