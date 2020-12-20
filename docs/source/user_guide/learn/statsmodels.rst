.. _statsmodels:

**************************
Integrate with StatsModels
**************************

.. currentmodule:: mars.learn.contrib.statsmodels

This is an introduction about how to use `StatsModels
<https://www.statsmodels.org>`_ for model fitting and prediction in Mars.

Installation
------------

If you are trying to use Mars on a single machine e.g. on your laptop, make
sure StatsModels is installed.

You can install StatsModels via pip:

.. code-block:: bash

    pip install statsmodels

Visit `installation guide for StatsModels
<https://www.statsmodels.org/stable/install.html>`_ for more information.

On the other hand, if you are using Mars on a cluster, make sure StatsModels is
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

   In [7]: from mars.learn.model_selection import train_test_split
   In [8]: X_train, X_test, y_train, y_test = \
      ...:     train_test_split(data, boston.target, train_size=0.7, random_state=0)

Training
--------

We can fit a model with API similar to the `distributed estimation API
<https://www.statsmodels.org/stable/examples/notebooks/generated/distributed_estimation.html>`_
implemented in StatsModels.

.. code-block:: ipython

   In [9]: from mars.learn.contrib import statsmodels as msm
   In [10]: model = msm.MarsDistributedModel(num_partitions=5)
   In [11]: results = model.fit(y_train, X_train, alpha=0.2)
   In [12]: results
   Out[12]: <mars.learn.contrib.statsmodels.api.MarsResults at 0x7fd47a118f70>

Arguments for ``DistributedModel`` like ``model_class``, ``estimation_method``
and ``join_method`` can be added to the constructor of
``MarsDistributedModel``.

Prediction
----------

For prediction,

.. code-block:: ipython

   In [13]: results.predict(X_test)
   Out[13]:
   377    20.475695
   218    20.792441
   216    23.158081
   78     19.912593
   467    14.290641
            ...
   94     24.798897
   120    22.196336
   53     23.714524
   165    19.824247
   319    22.138279
   Length: 152, dtype: float64

Distributed fitting and prediction
-----------------------------------

Refer to :ref:`deploy` section for deployment, or :ref:`k8s` section for
running Mars on Kubernetes.

Once a cluster exists, you can either set the session as default, the fitting
and prediction shown above will be submitted to the cluster, or you can specify
``session=***`` explicitly as well.

Take :meth:`MarsDistributedModel.fit` as an example.

.. code-block:: python

   # A cluster has been configured, and web UI is started on <web_ip>:<web_port>
   from mars.session import new_session
   # set the session as the default one
   sess = new_session('http://<web_ip>:<web_port>').as_default()

   # specify partition number
   model = msm.MarsDistributedModel(num_partitions=5)
   # or specify factor for cluster size,
   # num_partitions will be int(factor * num_cores)
   model = msm.MarsDistributedModel(factor=1.2)

   # fitting will submitted to cluster by default
   results = model.fit(y_train, X_train, alpha=1.2)

   # Or, session could be specified as well
   results = model.fit(y_train, X_train, alpha=1.2, session=sess)
