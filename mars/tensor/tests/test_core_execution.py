#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import numpy as np

from .. import (
    ones,
    add,
    swapaxes,
    moveaxis,
    atleast_1d,
    atleast_2d,
    atleast_3d,
    squeeze,
    tensor,
)


def test_array_function(setup):
    a = ones((10, 20), chunk_size=8)

    # test sum
    np.testing.assert_equal(np.sum(a).execute().fetch(), 200)

    # test qr
    q, r = np.linalg.qr(a)
    np.testing.assert_array_almost_equal(np.dot(q, r).execute().fetch(), a)


def test_view_data_on_slice(setup):
    data = np.random.rand(10, 20)
    a = tensor(data, chunk_size=8)
    b = a[:5, 5:10]
    b[:3, :3] = 3

    npa = data.copy()
    npb = npa[:5, 5:10]
    npb[:3, :3] = 3

    np.testing.assert_array_equal(b.execute(), npb)
    np.testing.assert_array_equal(a.execute(), npa)

    data = np.random.rand(10, 20)
    a = tensor(data, chunk_size=8)
    b = a[:7]
    b += 1

    npa = data.copy()
    npb = npa[:7]
    npb += 1

    np.testing.assert_array_equal(a.execute(), npa)
    np.testing.assert_array_equal(b.execute(), npb)


def test_set_item_on_view(setup):
    a = ones((5, 8), dtype=int)
    b = a[:3]
    b[0, 0] = 2
    c = b.ravel()  # create view
    c[1] = 4

    npa = np.ones((5, 8), dtype=int)
    npb = npa[:3]
    npb[0, 0] = 2
    npc = npb.ravel()  # create view
    npc[1] = 4

    np.testing.assert_array_equal(a.execute(), npa)
    np.testing.assert_array_equal(b.execute(), npb)
    np.testing.assert_array_equal(c.execute(), npc)


def test_view_data_on_transpose(setup):
    data = np.random.rand(10, 20)
    a = tensor(data, chunk_size=6)
    b = a.T
    add(b, 1, out=b)

    np.testing.assert_array_equal(b.execute(), data.T + 1)
    np.testing.assert_array_equal(a.execute(), data + 1)


def test_view_data_on_swapaxes(setup):
    data = np.random.rand(10, 20)
    a = tensor(data, chunk_size=6)
    b = swapaxes(a, 1, 0)
    a[1] = 10

    npa = data.copy()
    npb = np.swapaxes(npa, 1, 0)
    npa[1] = 10

    np.testing.assert_array_equal(b.execute(), npb)
    np.testing.assert_array_equal(a.execute(), npa)


def test_view_data_on_moveaxis(setup):
    data = np.random.rand(10, 20)
    a = tensor(data, chunk_size=6)
    b = moveaxis(a, 1, 0)
    a[0][1] = 10

    npa = data.copy()
    npb = np.moveaxis(npa, 1, 0)
    npa[0][1] = 10

    np.testing.assert_array_equal(b.execute(), npb)
    np.testing.assert_array_equal(a.execute(), npa)


def test_view_data_on_atleast1d(setup):
    a = atleast_1d(1)
    b = a[:]
    b[0] = 10

    np.testing.assert_array_equal(b.execute(), np.array([10]))
    np.testing.assert_array_equal(a.execute(), np.array([10]))


def test_view_data_on_atleast2d(setup):
    data = np.random.rand(10)
    a = atleast_2d(tensor(data, chunk_size=5))
    b = add(a[:, :5], 1, out=a[:, 5:])

    npa = np.atleast_2d(data.copy())
    npb = np.add(npa[:, :5], 1, out=npa[:, 5:])

    np.testing.assert_array_equal(a.execute(), npa)
    np.testing.assert_array_equal(b.execute(), npb)


def test_view_data_on_atleast3d(setup):
    data = np.random.rand(10, 20)
    a = atleast_3d(tensor(data, chunk_size=5))
    b = a[:, :5, :10][0]
    c = add(b[:4], b[1:], out=a[0, 16:])

    npa = np.atleast_3d(data.copy())
    npb = npa[:, :5, :10][0]
    npc = np.add(npb[:4], npb[1:], out=npa[0, 16:])

    np.testing.assert_array_equal(a.execute(), npa)
    np.testing.assert_array_equal(b.execute(), npb)
    np.testing.assert_array_equal(c.execute(), npc)


def test_view_data_on_squeeze(setup):
    data = np.random.rand(1, 4, 1)
    a = tensor(data, chunk_size=2)
    b = squeeze(a, axis=0)
    b[:3] = 10

    npa = data.copy()
    npb = np.squeeze(npa, axis=0)
    npb[:3] = 10

    np.testing.assert_array_equal(b.execute(), npb)
    np.testing.assert_array_equal(a.execute(), npa)


def test_view_data_on_reshape(setup):
    data = np.random.RandomState(0).random((3, 4, 5))
    a = tensor(data.copy(), chunk_size=2)
    b = a.reshape((5, 4, 3))
    b[:3] = 10

    npa = data.copy()
    npb = npa.reshape((5, 4, 3))
    npb[:3] = 10

    np.testing.assert_array_equal(b.execute(), npb)
    np.testing.assert_array_equal(a.execute(), npa)

    data = np.random.RandomState(0).random((4, 5))
    a2 = tensor(data.copy(), chunk_size=2)
    b2 = a2.reshape((5, 4), order="F")
    b2[:3] = 10

    npa = data.copy()
    npb = npa.reshape((5, 4), order="F")
    npb[:3] = 10

    b2_result = b2.execute()
    np.testing.assert_array_equal(a2.execute(), npa)
    np.testing.assert_array_equal(b2_result, npb)


def test_view_data_on_ravel(setup):
    # ravel creates a view
    data = np.random.rand(3, 4, 5)
    a = tensor(data, chunk_size=2)
    b = a.ravel()
    b[:10] = 10

    npa = data.copy()
    npb = npa.ravel()
    npb[:10] = 10

    np.testing.assert_array_equal(b.execute(), npb)
    np.testing.assert_array_equal(a.execute(), npa)

    # flatten creates a copy
    data = np.random.rand(3, 4, 5)
    a = tensor(data, chunk_size=2)
    b = a.flatten()
    b[:10] = 10

    npa = data.copy()
    npb = npa.flatten()
    npb[:10] = 10

    np.testing.assert_array_equal(b.execute(), npb)
    np.testing.assert_array_equal(a.execute(), npa)


def test_copy_and_view(setup):
    data = np.random.rand(10, 20)
    a = tensor(data, chunk_size=6)
    b = a.view()
    b[:5] = 10

    npa = data.copy()
    npb = npa.view()
    npb[:5] = 10

    np.testing.assert_array_equal(b.execute(), npb)
    np.testing.assert_array_equal(a.execute(), npa)

    data = np.random.rand(10, 20)
    a = tensor(data.copy(), chunk_size=6)
    b = a.copy()
    b[:5] = 10

    npa = data.copy()
    npb = npa.copy()
    npb[:5] = 10

    np.testing.assert_array_equal(b.execute(), npb)
    np.testing.assert_array_equal(a.execute(), npa)

    a = tensor(data.copy(), chunk_size=6)
    b = a[:5, :4]
    c = b.copy()
    c[0, 0] = 10

    npa = data.copy()
    npb = npa[:5, :4]
    npc = npb.copy()
    npc[0, 0] = 10

    np.testing.assert_array_equal(c.execute(), npc)
    np.testing.assert_array_equal(a.execute(), npa)


def test_flat(setup):
    data = np.random.rand(10, 20)
    a = tensor(data, chunk_size=4)
    fl = a.flat
    fl[1:10] = 10
    b = fl[10:20]
    b[0:4] = 20

    npa = data.copy()
    npfl = npa.flat
    npfl[1:10] = 10
    npb = npfl[10:20]
    npb[0:4] = 20

    np.testing.assert_array_equal(b.execute(), npb)
    np.testing.assert_array_equal(a.execute(), npa)
