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

import os
import shutil
import tempfile
import time

import numpy as np
import scipy.sparse as sps
import pytest
try:
    import tiledb
except (ImportError, OSError):  # pragma: no cover
    tiledb = None
try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None
try:
    import zarr
    from numcodecs import Zstd, Delta, Blosc
except ImportError:  # pragma: no cover
    zarr = None

from mars.config import option_context
from mars.tensor import tensor, arange, totiledb, tohdf5, tozarr
from mars.tests import new_test_session

try:
    import vineyard
except ImportError:
    vineyard = None

_exec_timeout = 120 if 'CI' in os.environ else -1


@pytest.fixture(scope='module')
def setup():
    sess = new_test_session(default=True)
    with option_context({'show_progress': False}):
        try:
            yield sess
        finally:
            sess.stop_server()


@pytest.mark.skipif(tiledb is None, reason='tiledb not installed')
def test_store_tiledb_execution(setup):
    ctx = tiledb.Ctx()

    tempdir = tempfile.mkdtemp()
    try:
        # store TileDB dense array
        expected = np.random.rand(8, 4, 3)
        a = tensor(expected, chunk_size=(3, 3, 2))
        save = totiledb(tempdir, a, ctx=ctx)
        save.execute()

        with tiledb.DenseArray(uri=tempdir, ctx=ctx) as arr:
            np.testing.assert_allclose(expected, arr.read_direct())
    finally:
        shutil.rmtree(tempdir)

    tempdir = tempfile.mkdtemp()
    try:
        # store tensor with 1 chunk to TileDB dense array
        a = arange(12)
        save = totiledb(tempdir, a, ctx=ctx)
        save.execute()

        with tiledb.DenseArray(uri=tempdir, ctx=ctx) as arr:
            np.testing.assert_allclose(np.arange(12), arr.read_direct())
    finally:
        shutil.rmtree(tempdir)

    tempdir = tempfile.mkdtemp()
    try:
        # store 2-d TileDB sparse array
        expected = sps.random(8, 7, density=0.1)
        a = tensor(expected, chunk_size=(3, 5))
        save = totiledb(tempdir, a, ctx=ctx)
        save.execute()

        with tiledb.SparseArray(uri=tempdir, ctx=ctx) as arr:
            data = arr[:, :]
            coords = data['coords']
            value = data[arr.attr(0).name]
            ij = tuple(coords[arr.domain.dim(k).name] for k in range(arr.ndim))
            result = sps.coo_matrix((value, ij), shape=arr.shape)

            np.testing.assert_allclose(expected.toarray(), result.toarray())
    finally:
        shutil.rmtree(tempdir)

    tempdir = tempfile.mkdtemp()
    try:
        # store TileDB dense array
        expected = np.asfortranarray(np.random.rand(8, 4, 3))
        a = tensor(expected, chunk_size=(3, 3, 2))
        save = totiledb(tempdir, a, ctx=ctx)
        save.execute()

        with tiledb.DenseArray(uri=tempdir, ctx=ctx) as arr:
            np.testing.assert_allclose(expected, arr.read_direct())
            assert arr.schema.cell_order == 'col-major'
    finally:
        shutil.rmtree(tempdir)


@pytest.mark.skipif(h5py is None, reason='h5py not installed')
def test_store_hdf5_execution(setup):
    raw = np.random.RandomState(0).rand(10, 20)

    group_name = 'test_group'
    dataset_name = 'test_dataset'

    t1 = tensor(raw, chunk_size=20)
    t2 = tensor(raw, chunk_size=9)

    with pytest.raises(TypeError):
        tohdf5(object(), t2)

    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, f'test_store_{int(time.time())}.hdf5')

        # test 1 chunk
        r = tohdf5(filename, t1, group=group_name, dataset=dataset_name)
        r.execute()

        with h5py.File(filename, 'r') as f:
            result = np.asarray(f[f'{group_name}/{dataset_name}'])
            np.testing.assert_array_equal(result, raw)

        # test filename
        r = tohdf5(filename, t2, group=group_name, dataset=dataset_name)
        r.execute()

        with h5py.File(filename, 'r') as f:
            result = np.asarray(f[f'{group_name}/{dataset_name}'])
            np.testing.assert_array_equal(result, raw)

        with pytest.raises(ValueError):
            tohdf5(filename, t2)

        with h5py.File(filename, 'r') as f:
            # test file
            r = tohdf5(f, t2, group=group_name, dataset=dataset_name)
        r.execute()

        with h5py.File(filename, 'r') as f:
            result = np.asarray(f[f'{group_name}/{dataset_name}'])
            np.testing.assert_array_equal(result, raw)

        with pytest.raises(ValueError):
            with h5py.File(filename, 'r') as f:
                tohdf5(f, t2)

        with h5py.File(filename, 'r') as f:
            # test dataset
            ds = f[f'{group_name}/{dataset_name}']
            # test file
            r = tohdf5(ds, t2)
        r.execute()

        with h5py.File(filename, 'r') as f:
            result = np.asarray(f[f'{group_name}/{dataset_name}'])
            np.testing.assert_array_equal(result, raw)


@pytest.mark.skipif(zarr is None, reason='zarr not installed')
def test_store_zarr_execution(setup):
    raw = np.random.RandomState(0).rand(10, 20)

    group_name = 'test_group'
    dataset_name = 'test_dataset'

    t = tensor(raw, chunk_size=6)

    with pytest.raises(TypeError):
        tozarr(object(), t)

    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, f'test_store_{int(time.time())}.zarr')
        path = f'{filename}/{group_name}/{dataset_name}'

        r = tozarr(filename, t, group=group_name, dataset=dataset_name, compressor=Zstd(level=3))
        r.execute()

        arr = zarr.open(path)
        np.testing.assert_array_equal(arr, raw)
        assert arr.compressor == Zstd(level=3)

        r = tozarr(path, t + 2)
        r.execute()

        arr = zarr.open(path)
        np.testing.assert_array_equal(arr, raw + 2)

        filters = [Delta(dtype='i4')]
        compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE)
        arr = zarr.open(path, compressor=compressor, filters=filters)

        r = tozarr(arr, t + 1)
        r.execute()
        result = zarr.open_array(path)
        np.testing.assert_array_equal(result, raw + 1)


# @pytest.mark.skipif(vineyard is None, reason='vineyard not installed')
# @flaky(max_runs=3)
# def test_to_vineyard():
#     def run_with_given_session(session, **kw):
#         ipc_socket = os.environ.get('VINEYARD_IPC_SOCKET', '/tmp/vineyard/vineyard.sock')
#         with option_context({'vineyard.socket': ipc_socket}):
#             tensor1 = tensor(np.arange(12).reshape(3, 4), chunk_size=2)
#             object_id = tovineyard(tensor1).execute(session=session, **kw).fetch(session=session)
#             tensor2 = from_vineyard(object_id)
#
#             tensor1_value = tensor1.execute(session=session, **kw).fetch(session=session)
#             tensor2_value = tensor2.execute(session=session, **kw).fetch(session=session)
#             np.testing.assert_array_equal(tensor1_value, tensor2_value)
#
#     with new_session().as_default() as session:
#         run_with_given_session(session)
#
#     with new_cluster(scheduler_n_process=2, worker_n_process=2,
#                      shared_memory='20M', web=False) as cluster:
#         with new_session(cluster.endpoint).as_default() as session:
#             run_with_given_session(session, timeout=_exec_timeout)
