.. _learn_api:

==========
Mars Learn
==========

This is the class and function reference of Mars learn.

.. currentmodule:: mars.learn

.. autosummary::
   :toctree: generated/

.. _cluster_ref:

Clustering
==========

.. automodule:: mars.learn.cluster
   :no-members:
   :no-inherited-members:

Classes
-------
.. currentmodule:: mars.learn

.. autosummary::
   :toctree: generated/

   cluster.KMeans

Functions
---------
.. currentmodule:: mars.learn

.. autosummary::
   :toctree: generated/

   cluster.k_means

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
   datasets.make_regression

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

.. _ensemble_ref:

Ensemble Methods
================

.. automodule:: mars.learn.metrics
   :no-members:
   :no-inherited-members:

.. currentmodule:: mars.learn

.. autosummary::
   :toctree: generated/

   ensemble.BaggingClassifier
   ensemble.BaggingRegressor
   ensemble.BlockwiseVotingClassifier
   ensemble.BlockwiseVotingRegressor
   ensemble.IsolationForest

.. _linear_model_ref:

Linear Models
=============

Classical linear regressors
---------------------------

.. currentmodule:: mars.learn

.. autosummary::
   :toctree: generated/

   linear_model.LinearRegression

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
   metrics.auc
   metrics.f1_score
   metrics.fbeta_score
   metrics.log_loss
   metrics.multilabel_confusion_matrix
   metrics.precision_score
   metrics.precision_recall_fscore_support
   metrics.recall_score
   metrics.roc_curve

Regression metrics
------------------

.. autosummary::
   :toctree: generated/

   metrics.r2_score


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

.. _modelselection_ref:

Model Selection
===============

Splitter Classes
----------------

.. currentmodule:: mars.learn

.. autosummary::
   :toctree: generated/

   model_selection.KFold

Splitter Functions
------------------

.. currentmodule:: mars.learn

.. autosummary::
   :toctree: generated/

   model_selection.train_test_split

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

   preprocessing.LabelBinarizer
   preprocessing.LabelEncoder
   preprocessing.MinMaxScaler
   preprocessing.minmax_scale
   preprocessing.label_binarize
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

.. _learn_misc_ref：

Misc
====

.. automodule:: mars.learn.wrappers
   :no-members:
   :no-inherited-members:

.. currentmodule:: mars.learn

.. autosummary::
   :toctree: generated/

   wrappers.ParallelPostFit

.. _lightgbm_ref:

LightGBM Integration
====================

.. automodule:: mars.learn.contrib.lightgbm
   :no-members:
   :no-inherited-members:

.. currentmodule:: mars.learn

.. autosummary::
   :toctree: generated/

   contrib.lightgbm.LGBMClassifier
   contrib.lightgbm.LGBMRegressor
   contrib.lightgbm.LGBMRanker

.. _pytorch_ref:

PyTorch Integration
======================

.. automodule:: mars.learn.contrib.pytorch
   :no-members:
   :no-inherited-members:

.. currentmodule:: mars.learn

.. autosummary::
   :toctree: generated/

   contrib.pytorch.run_pytorch_script
   contrib.pytorch.MarsDataset
   contrib.pytorch.SequentialSampler
   contrib.pytorch.RandomSampler
   contrib.pytorch.SubsetRandomSampler
   contrib.pytorch.DistributedSampler

.. _statsmodels_ref:

StatsModels Integration
=======================

.. automodule:: mars.learn.contrib.statsmodels
   :no-members:
   :no-inherited-members:

.. currentmodule:: mars.learn

.. autosummary::
   :toctree: generated/

   contrib.statsmodels.MarsDistributedModel
   contrib.statsmodels.MarsResults

.. _tensorflow_ref:

TensorFlow Integration
======================

.. automodule:: mars.learn.contrib.tensorflow
   :no-members:
   :no-inherited-members:

.. currentmodule:: mars.learn

.. autosummary::
   :toctree: generated/

   contrib.tensorflow.run_tensorflow_script
   contrib.tensorflow.gen_tensorflow_dataset

.. _xgboost_ref:

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
