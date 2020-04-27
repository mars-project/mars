.. _api_ref:

=============
API Reference
=============

This is the class and function reference of Mars learn.

.. currentmodule:: mars.learn

.. autosummary::
   :toctree: generated/

.. _datasets_ref:

Datasets
========

.. automodule:: mars.learn.datasets
   :no-members:
   :no-inherited-members:

Samples generator
-----------------

.. currentmodule:: mars.learn

.. autosummary::
   :toctree: generated/

   datasets.make_blobs
   datasets.make_classification
   datasets.make_low_rank_matrix

.. _decomposition_ref:

Matrix Decomposition
====================

.. automodule:: mars.learn.decomposition
   :no-members:
   :no-inherited-members:

.. currentmodule:: mars.learn

.. autosummary::
   :toctree: generated/

   decomposition.PCA
   decomposition.TruncatedSVD


.. _metrics_ref:

Metrics
=======

.. automodule:: mars.learn.metrics
   :no-members:
   :no-inherited-members:

.. currentmodule:: mars.learn

Classification metrics
----------------------

.. autosummary::
   :toctree: generated/

   metrics.accuracy_score

Pairwise metrics
----------------

.. automodule:: mars.learn.metrics.pairwise
   :no-members:
   :no-inherited-members:

.. currentmodule:: mars.learn

.. autosummary::
   :toctree: generated/

   metrics.pairwise.cosine_similarity
   metrics.pairwise.cosine_distances
   metrics.pairwise.euclidean_distances
   metrics.pairwise.haversine_distances
   metrics.pairwise.manhattan_distances
   metrics.pairwise.rbf_kernel
   metrics.pairwise_distances

.. _neighbors_ref:

Nearest Neighbors
=================

.. automodule:: mars.learn.neighbors
   :no-members:
   :no-inherited-members:

.. currentmodule:: mars.learn

.. autosummary::
   :toctree: generated/

   neighbors.NearestNeighbors

.. _preprocessing_ref:

Preprocessing and Normalization
===============================

.. automodule:: mars.learn.preprocessing
   :no-members:
   :no-inherited-members:

.. currentmodule:: mars.learn

.. autosummary::
   :toctree: generated/

   preprocessing.normalize

.. _semi_supervised_ref:

Semi-Supervised Learning
========================

.. automodule:: mars.learn.semi_supervised
   :no-members:
   :no-inherited-members:

.. currentmodule:: mars.learn

.. autosummary::
   :toctree: generated/

   semi_supervised.LabelPropagation

.. _utils_ref:

Utilities
=========

.. automodule:: mars.learn.utils
   :no-members:
   :no-inherited-members:

.. currentmodule:: mars.learn

.. autosummary::
   :toctree: generated/

   utils.assert_all_finite
   utils.check_X_y
   utils.check_array
   utils.check_consistent_length
   utils.multiclass.type_of_target
   utils.multiclass.is_multilabel
   utils.shuffle
   utils.validation.check_is_fitted
   utils.validation.column_or_1d

.. _xgboost_ref:

TensorFlow Integration
======================

.. automodule:: mars.learn.contrib.tensorflow
   :no-members:
   :no-inherited-members:

.. currentmodule:: mars.learn

.. autosummary::
   :toctree: generated/

   contrib.tensorflow.run_tensorflow_script

PyTorch Integration
======================

.. automodule:: mars.learn.contrib.tensorflow
   :no-members:
   :no-inherited-members:

.. currentmodule:: mars.learn

.. autosummary::
   :toctree: generated/

   contrib.pytorch.run_pytorch_script
   contrib.pytorch.MarsDataset
   contrib.pytorch.MarsDistributedSampler
   contrib.pytorch.MarsRandomSampler

XGBoost Integration
===================

.. automodule:: mars.learn.contrib.xgboost
   :no-members:
   :no-inherited-members:

.. currentmodule:: mars.learn

.. autosummary::
   :toctree: generated/

   contrib.xgboost.MarsDMatrix
   contrib.xgboost.train
   contrib.xgboost.predict
   contrib.xgboost.XGBClassifier
   contrib.xgboost.XGBRegressor
