.. _getting_started_learn:

Mars Learn
==========

Mars learn mimics scikit-learn API and leverages the ability of Mars tensor and
DataFrame to process large data and execute in parallel.

Let's take :class:`mars.learn.neighbors.NearestNeighbors` as an example.

.. code-block:: python

   >>> import mars.tensor as mt
   >>> from mars.learn.neighbors import NearestNeighbors
   >>> data = mt.random.rand(100, 3)
   >>> nn = NearestNeighbors(n_neighbors=3)
   >>> nn.fit(data)
   NearestNeighbors(algorithm='auto', leaf_size=30, metric='minkowski',
                    metric_params=None, n_neighbors=3, p=2, radius=1.0)
   >>> neighbors = nn.kneighbors(df)
   >>> neighbors
   (array([[0.0560703 , 0.1836808 , 0.19055679],
           [0.07100642, 0.08550266, 0.10617568],
           [0.13348483, 0.16597596, 0.20287617]]),
    array([[91, 10, 29],
           [68, 77, 29],
           [63, 82, 21]]))

Remember that functions like ``fit``, ``predict`` will trigger execution instantly.
In the above example, ``fit`` and ``kneighbors`` will trigger execution internally.

For implemented learn API, refer to :ref:`learn API reference <learn_api>`.

Mars learn can integrate with XGBoost, LightGBM, TensorFlow and PyTorch.

- For XGBoost, refer to :ref:`integrate_xgboost`.
- For LightGBM, refer to :ref:`integrate_lightgbm`.
- For TensorFlow, refer to :ref:`integrate_tensorflow`.
- For PyTorch, doc is coming soon.
- For Joblib, refer to :ref:`integrate_joblib`.
- For Statsmodels, refer to :ref:`integrate_statsmodels`.
