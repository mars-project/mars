.. _lightgbm:

***********************
Integrate with LightGBM
***********************

.. currentmodule:: mars.learn.contrib.lightgbm

This is an introduction about how to use LightGBM for training and prediction
in Mars.

Installation
------------

If you are trying to use Mars on a single machine e.g. on your laptop, make
sure LightGBM is installed.

You can install LightGBM via pip:

.. code-block:: bash

    pip install lightgbm

Visit `installation guide for LightGBM
<https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html>`_ for more
information.

On the other hand, if you are using Mars on a cluster, make sure LightGBM is
installed on each worker.

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

We can shuffle the sequence of the data, and separate the data into train and
test parts.

.. code-block:: ipython

   In [7]: from mars.learn.utils import shuffle
   In [8]: X, y = shuffle(data, boston.target)
   In [9]: train_size = int(X.shape[0] * 0.7)
   In [10]: X_train, X_test = X[:train_size], X[train_size:]
   In [11]: y_train, y_test = y[:train_size], y[train_size:]

Training
--------

We can train data with scikit-learn API similar to the API implemented in
LightGBM.

.. code-block:: ipython

   In [12]: from mars.learn.contrib import lightgbm as lgb
   In [13]: lg_reg = lgb.LGBMRegressor(colsample_bytree=0.3, learning_rate=0.1,
       ...:                            max_depth=5, reg_alpha=10, n_estimators=10)

   In [14]: lg_reg.fit(X_train, y_train)
   Out[14]:
   LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=0.3,
                 importance_type='split', learning_rate=0.1, max_depth=5,
                 min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
                 n_estimators=10, n_jobs=-1, num_leaves=31, objective=None,
                 out_dtype_=dtype('float64'), random_state=None, reg_alpha=10,
                 reg_lambda=0.0, silent=True, subsample=1.0,
                 subsample_for_bin=200000, subsample_freq=0)

Prediction
----------

For prediction, scikit-learn API can also be used.

.. code-block:: ipython

   In [15]: lg_reg.predict(X_test)
   Out[15]:
   476    20.044731
   490    18.540266
   304    26.985506
   216    21.946456
   256    34.913612
            ...
   250    24.234580
   224    34.980905
   500    21.376179
   134    19.605267
   248    23.253156
   Name: predictions, Length: 152, dtype: float64

Distributed training and prediction
-----------------------------------

Refer to :ref:`deploy` section for deployment, or :ref:`k8s` section for
running Mars on Kubernetes.

Once a cluster exists, you can either set the session as default, the training
and prediction shown above will be submitted to the cluster, or you can specify
`session=***` explicitly as well.

Take :meth:`LGBMRegressor.fit` as an example.

.. code-block:: python

   # A cluster has been configured, and web UI is started on <web_ip>:<web_port>
   from mars.session import new_session
   # set the session as the default one
   sess = new_session('http://<web_ip>:<web_port>').as_default()

   lg_reg = lgb.LGBMRegressor()

   # training will submitted to cluster by default
   lg_reg.fit(X_train)

   # Or, session could be specified as well
   lg_reg.fit(X_train, session=sess)
