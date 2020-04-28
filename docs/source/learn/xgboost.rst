.. _xgboost:

**********************
Integrate with XGBoost
**********************

.. currentmodule:: mars.learn.contrib.xgboost

This is an introduction about how to use XGBoost for training and prediction in Mars.

Installation
------------

If you are trying to use Mars on a single machine e.g. on your laptop,
make sure XGBoost is installed.

You can install XGBoost via pip:

.. code-block:: bash

    pip install xgboost

Visit `installation guide for XGBoost <https://xgboost.readthedocs.io/en/latest/build.html>`_
for more information.

On the other hand, if you are using Mars on a cluster, make sure
XGBoost is installed on each worker.

Prepare data
------------

First, we use scikit-learn to load the Boston Housing dataset.

.. code-block:: ipython

   In [1]: from sklearn.datasets import load_boston
   In [2]: boston = load_boston()

Then create Mars DataFrame from the dataset.

.. code-block:: ipython

   In [3]: import mars.dataframe as md
   In [4]: data = md.DataFrame(boston.data, columns=boston.feature_names)

Explore the top 5 rows data of the DataFrame.

.. code-block:: ipython

   In [5]: data.head().execute()
   Out[5]:
         CRIM    ZN  INDUS  CHAS    NOX  ...  RAD    TAX  PTRATIO       B  LSTAT
   0  0.00632  18.0   2.31   0.0  0.538  ...  1.0  296.0     15.3  396.90   4.98
   1  0.02731   0.0   7.07   0.0  0.469  ...  2.0  242.0     17.8  396.90   9.14
   2  0.02729   0.0   7.07   0.0  0.469  ...  2.0  242.0     17.8  392.83   4.03
   3  0.03237   0.0   2.18   0.0  0.458  ...  3.0  222.0     18.7  394.63   2.94
   4  0.06905   0.0   2.18   0.0  0.458  ...  3.0  222.0     18.7  396.90   5.33

   [5 rows x 13 columns]

:meth:`mars.dataframe.DataFrame.describe` gives summary statistics of the columns.

.. code-block:: ipython

   In [6]: data.describe().execute()
   Out[6]:
                CRIM          ZN       INDUS  ...     PTRATIO           B       LSTAT
   count  506.000000  506.000000  506.000000  ...  506.000000  506.000000  506.000000
   mean     3.613524   11.363636   11.136779  ...   18.455534  356.674032   12.653063
   std      8.601545   23.322453    6.860353  ...    2.164946   91.294864    7.141062
   min      0.006320    0.000000    0.460000  ...   12.600000    0.320000    1.730000
   25%      0.082045    0.000000    5.190000  ...   17.400000  375.377500    6.950000
   50%      0.256510    0.000000    9.690000  ...   19.050000  391.440000   11.360000
   75%      3.677083   12.500000   18.100000  ...   20.200000  396.225000   16.955000
   max     88.976200  100.000000   27.740000  ...   22.000000  396.900000   37.970000

   [8 rows x 13 columns]

We can shuffle the sequence of the data, and separate the data into train and test parts.

.. code-block:: ipython

   In [7]: from mars.learn.utils import shuffle
   In [8]: X, y = shuffle(data, boston.target)
   In [9]: train_size = int(X.shape[0] * 0.7)
   In [10]: X_train, X_test = X[:train_size], X[train_size:]
   In [11]: y_train, y_test = y[:train_size], y[train_size:]

Now we can create a :class:`MarsDMatrix` which is very similar to
`xgboost.DMatrix <https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.DMatrix>`_.

.. code-block:: ipython

   In [12]: from mars.learn.contrib import xgboost as xgb

   In [13]: train_dmatrix = xgb.MarsDMatrix(data=X_train, label=y_train)
   In [14]: test_dmatrix = xgb.MarsDMatrix(data=X_test, label=y_test)

Training
--------

We can train data in two ways:

1. Call :meth:`train` which accepts a :class:`MarsDMatrix`.
2. Use scikit-learn API including :class:`XGBClassifier` and :class:`XGBRegressor`.

For :meth:`train`, you can run the snippet.

.. code-block:: ipython

   In [15]: params = {'objective': 'reg:squarederror','colsample_bytree': 0.3,'learning_rate': 0.1,
       ...:           'max_depth': 5, 'alpha': 10, 'n_estimators': 10}

   In [16]: booster = xgb.train(dtrain=train_dmatrix, params=params)

On the other hand, run the snippet below for scikit-learn API.

.. code-block:: ipython

   In [17]: xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3,
       ...:                           learning_rate=0.1, max_depth=5, alpha=10, n_estimators=10)

   In [18]: xg_reg.fit(X_train, y_train)
   Out[18]:
   XGBRegressor(alpha=10, base_score=0.5, booster='gbtree', colsample_bylevel=1,
                colsample_bynode=1, colsample_bytree=0.3, gamma=0,
                importance_type='gain', learning_rate=0.1, max_delta_step=0,
                max_depth=5, min_child_weight=1, missing=None, n_estimators=10,
                n_jobs=1, nthread=None, objective='reg:squarederror',
                random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                seed=None, silent=None, subsample=1, verbosity=1)

Prediction
----------

For prediction, there are still two ways

1. Call :meth:`predict` which accepts a :class:`MarsDMatrix` as well.
2. Call :meth:`XGBClassifier.predict` or :meth:`XGBRegressor.predict`
   which has been fitted.

For :meth:`predict`, we call it with trained model.

.. code-block:: ipython

   In [19]: xgb.predict(booster, X_test)
   Out[19]: Series <op=XGBPredict, key=60208a66c4922453e720572c0909ff28>

For :meth:`XGBRegressor.predict`, you can run the snippet.

.. code-block:: ipython

   In [46]: xg_reg.predict(X_test)
   Out[46]: Series <op=XGBPredict, key=8415591102455e915a020dc8bc3541f4>

Distributed training and prediction
-----------------------------------

Refer to :ref:`deploy` section for deployment, or :ref:`k8s` section for running Mars on Kubernetes.

Once a cluster exists, you can either set the session as default,
the training and prediction shown above will be submitted to the cluster,
or you can specify `session=***` explicitly as well.

Take :meth:`XGBRegressor.fit` as an example.

.. code-block:: python

   # A cluster has been configured, and web UI is started on <web_ip>:<web_port>
   from mars.session import new_session
   # set the session as the default one
   sess = new_session('http://<web_ip>:<web_port>').as_default()

   reg = xgb.XGBRegressor()

   # training will submitted to cluster by default
   reg.fit(X_train)

   # Or, session could be specified as well
   reg.fit(X_train, session=sess)
