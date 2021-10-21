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

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

try:
    import xgboost
except ImportError:
    xgboost = None

from ..... import tensor as mt
from ..... import dataframe as md
from ..classifier import XGBClassifier

n_rows = 1000
n_columns = 10
chunk_size = 200
rs = mt.random.RandomState(0)
X_raw = rs.rand(n_rows, n_columns, chunk_size=chunk_size)
y_raw = rs.rand(n_rows, chunk_size=chunk_size)
X_df_raw = md.DataFrame(X_raw)


@pytest.mark.skipif(xgboost is None, reason="XGBoost not installed")
def test_local_classifier(setup):
    y = (y_raw * 10).astype(mt.int32)
    classifier = XGBClassifier(verbosity=1, n_estimators=2)
    classifier.fit(X_raw, y, eval_set=[(X_raw, y)])
    prediction = classifier.predict(X_raw)

    assert prediction.ndim == 1
    assert prediction.shape[0] == len(X_raw)

    history = classifier.evals_result()

    assert isinstance(prediction, mt.Tensor)
    assert isinstance(history, dict)

    assert list(history)[0] == "validation_0"
    # default metrics may differ, see https://github.com/dmlc/xgboost/pull/6183
    eval_metric = list(history["validation_0"])[0]
    assert eval_metric in ("merror", "mlogloss")
    assert len(history["validation_0"]) == 1
    assert len(history["validation_0"][eval_metric]) == 2

    prob = classifier.predict_proba(X_raw)
    assert prob.shape == X_raw.shape

    # test dataframe
    X_df = X_df_raw
    classifier = XGBClassifier(verbosity=1, n_estimators=2)
    classifier.fit(X_df, y)
    prediction = classifier.predict(X_df)

    assert prediction.ndim == 1
    assert prediction.shape[0] == len(X_raw)

    # test weight
    weights = [
        mt.random.rand(X_raw.shape[0]),
        md.Series(mt.random.rand(X_raw.shape[0])),
        md.DataFrame(mt.random.rand(X_raw.shape[0])),
    ]
    y_df = md.DataFrame(y)
    for weight in weights:
        classifier = XGBClassifier(verbosity=1, n_estimators=2)
        classifier.fit(X_raw, y_df, sample_weight=weight)
        prediction = classifier.predict(X_raw)

        assert prediction.ndim == 1
        assert prediction.shape[0] == len(X_raw)

    # should raise error if weight.ndim > 1
    with pytest.raises(ValueError):
        XGBClassifier(verbosity=1, n_estimators=2).fit(
            X_raw, y_df, sample_weight=mt.random.rand(1, 1)
        )

    # test binary classifier
    new_y = (y > 0.5).astype(mt.int32)
    classifier = XGBClassifier(verbosity=1, n_estimators=2)
    classifier.fit(X_raw, new_y)
    prediction = classifier.predict(X_raw)

    assert prediction.ndim == 1
    assert prediction.shape[0] == len(X_raw)

    # test predict data with unknown shape
    X2 = X_raw[X_raw[:, 0] > 0.1].astype(mt.int32)
    prediction = classifier.predict(X2)

    assert prediction.ndim == 1

    # test train with unknown shape
    cond = X_raw[:, 0] > 0
    X3 = X_raw[cond]
    y3 = y[cond]
    classifier = XGBClassifier(verbosity=1, n_estimators=2)
    classifier.fit(X3, y3)
    prediction = classifier.predict(X_raw)

    assert prediction.ndim == 1
    assert prediction.shape[0] == len(X_raw)

    classifier = XGBClassifier(verbosity=1, n_estimators=2)
    with pytest.raises(TypeError):
        classifier.fit(X_raw, y, wrong_param=1)
    classifier.fit(X_raw, y)
    with pytest.raises(TypeError):
        classifier.predict(X_raw, wrong_param=1)


@pytest.mark.skipif(xgboost is None, reason="XGBoost not installed")
def test_local_classifier_from_to_parquet(setup):
    n_rows = 1000
    n_columns = 10
    rs = np.random.RandomState(0)
    X = rs.rand(n_rows, n_columns)
    y = rs.rand(n_rows)
    df = pd.DataFrame(X, columns=[f"c{i}" for i in range(n_columns)])
    df["id"] = [f"i{i}" for i in range(n_rows)]

    booster = xgboost.train({}, xgboost.DMatrix(X, y), num_boost_round=2)

    with tempfile.TemporaryDirectory() as d:
        m_name = os.path.join(d, "c.model")
        result_dir = os.path.join(d, "result")
        os.mkdir(result_dir)
        data_dir = os.path.join(d, "data")
        os.mkdir(data_dir)

        booster.save_model(m_name)

        df.iloc[:500].to_parquet(os.path.join(d, "data", "data1.parquet"))
        df.iloc[500:].to_parquet(os.path.join(d, "data", "data2.parquet"))

        df = md.read_parquet(data_dir).set_index("id")
        model = XGBClassifier()
        model.load_model(m_name)
        result = model.predict(df, run=False)
        r = md.DataFrame(result).to_parquet(result_dir)

        # tiles to ensure no iterative tiling exists
        r.execute()

        ret = md.read_parquet(result_dir).to_pandas().iloc[:, 0].to_numpy()
        model2 = xgboost.XGBClassifier()
        model2.load_model(m_name)
        expected = model2.predict(X)
        expected = np.stack([1 - expected, expected]).argmax(axis=0)
        np.testing.assert_array_equal(ret, expected)
