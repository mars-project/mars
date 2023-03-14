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

import warnings
import tempfile

import numpy as np
import pandas as pd
import pytest

try:
    import matplotlib
except ImportError:  # pragma: no cover
    matplotlib = None

from .... import tensor as mt
from .... import dataframe as md


def close(fignum=None):  # pragma: no cover
    from matplotlib.pyplot import get_fignums, close as _close

    if fignum is None:
        for fignum in get_fignums():
            _close(fignum)
    else:
        _close(fignum)


def assert_is_valid_plot_return_object(objs):  # pragma: no cover
    import matplotlib.pyplot as plt

    if isinstance(objs, (pd.Series, np.ndarray)):
        for el in objs.ravel():
            msg = (
                "one of 'objs' is not a matplotlib Axes instance, "
                f"type encountered {type(el).__name__}"
            )
            assert isinstance(el, (plt.Axes, dict)), msg
    else:
        msg = (
            "objs is neither an ndarray of Artist instances nor a single "
            f"ArtistArtist instance, tuple, or dict, 'objs' is a {type(objs).__name__}"
        )
        assert isinstance(objs, (plt.Artist, tuple, dict)), msg


def _check_plot_works(f, filterwarnings="always", **kwargs):  # pragma: no cover
    import matplotlib.pyplot as plt

    ret = None
    with warnings.catch_warnings():
        warnings.simplefilter(filterwarnings)
        try:
            try:
                fig = kwargs["figure"]
            except KeyError:
                fig = plt.gcf()

            plt.clf()

            kwargs.get("ax", fig.add_subplot(211))
            ret = f(**kwargs)

            assert_is_valid_plot_return_object(ret)

            if f is pd.plotting.bootstrap_plot:
                assert "ax" not in kwargs
            else:
                kwargs["ax"] = fig.add_subplot(212)

            ret = f(**kwargs)
            assert_is_valid_plot_return_object(ret)

            with tempfile.TemporaryFile() as path:
                plt.savefig(path)
        finally:
            close(fig)

        return ret


@pytest.mark.skipif(matplotlib is None, reason="matplotlib is not installed")
def test_plot(setup):
    raw = pd.DataFrame(
        {
            "a": ["s" + str(i) for i in range(10)],
            "b": np.random.RandomState(0).randint(10, size=10),
        }
    )
    df = md.DataFrame(raw, chunk_size=3)

    _check_plot_works(df.plot, x="a", y="b")
    _check_plot_works(df.plot, x="a", y=mt.tensor("b"))
    _check_plot_works(df.plot.line)

    raw = pd.DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
            "C": np.random.randn(8),
            "D": np.random.randn(8),
        }
    )
    df = md.DataFrame(raw, chunk_size=3)
    _check_plot_works(df.groupby("A").plot)
