.. mars documentation master file, created by
   sphinx-quickstart on Mon Mar 26 11:56:11 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: images/mars-logo-title.png

Mars is a tensor-based unified framework for large-scale data computation.

Mars tensor
-----------

:doc:`documentation <tensor/index>`

Mars tensor provides a familiar interface like Numpy.

+-----------------------------------------------+-----------------------------------------------+
| **Numpy**                                     | **Mars tensor**                               |
+-----------------------------------------------+-----------------------------------------------+
|.. code-block:: python                         |.. code-block:: python                         |
|                                               |                                               |
|    import numpy as np                         |    import mars.tensor as mt                   |
|    N = 200_000_000                            |    N = 200_000_000                            |
|    a = np.random.uniform(-1, 1, size=(N, 2))  |    a = mt.random.uniform(-1, 1, size=(N, 2))  |
|    print((np.linalg.norm(a, axis=1) < 1)      |    print(((mt.linalg.norm(a, axis=1) < 1)     |
|          .sum() * 4 / N)                      |            .sum() * 4 / N).execute())         |
|                                               |                                               |
+-----------------------------------------------+-----------------------------------------------+
|.. code-block::                                |.. code-block::                                |
|                                               |                                               |
|    3.14151712                                 |     3.14161908                                |
|    CPU times: user 12.5 s, sys: 7.16 s,       |     CPU times: user 17.5 s, sys: 3.56 s,      |
|               total: 19.7 s                   |                total: 21.1 s                  |
|    Wall time: 21.8 s                          |     Wall time: 5.59 s                         |
|                                               |                                               |
+-----------------------------------------------+-----------------------------------------------+

Mars can leverage multiple cores, even on a laptop, and could be even faster for a distributed setting.

Mars dataframe
--------------

:doc:`documentation <dataframe/index>`

Mars DataFrame provides a familiar interface like pandas.

+-----------------------------------------+-----------------------------------------+
| **Pandas**                              | **Mars DataFrame**                      |
+-----------------------------------------+-----------------------------------------+
|.. code-block:: python                   |.. code-block:: python                   |
|                                         |                                         |
|    import numpy as np                   |    import mars.tensor as mt             |
|    import pandas as pd                  |    import mars.dataframe as md          |
|    df = pd.DataFrame(                   |    df = md.DataFrame(                   |
|        np.random.rand(100000000, 4),    |        mt.random.rand(100000000, 4),    |
|        columns=list('abcd'))            |        columns=list('abcd'))            |
|    print(df.sum())                      |    print(df.sum().execute())            |
|                                         |                                         |
+-----------------------------------------+-----------------------------------------+
|.. code-block::                          |.. code-block::                          |
|                                         |                                         |
|    CPU times: user 10.9 s, sys: 2.69 s, |    CPU times: user 16.5 s, sys: 3.52 s, |
|               total: 13.6 s             |               total: 20 s               |
|    Wall time: 11 s                      |    Wall time: 3.6 s                     |
+-----------------------------------------+-----------------------------------------+

Mars learn
----------

:doc:`documentation <learn/index>`

Mars learn provides a familiar interface like scikit-learn.

+---------------------------------------------+----------------------------------------------------+
| **Scikit-learn**                            | **Mars learn**                                     |
+---------------------------------------------+----------------------------------------------------+
|.. code-block:: python                       |.. code-block:: python                              |
|                                             |                                                    |
|    from sklearn.datasets import make_blobs  |    from mars.learn.datasets import make_blobs      |
|    from sklearn.decomposition import PCA    |    from mars.learn.decomposition import PCA        |
|    X, y = make_blobs(                       |    X, y = make_blobs(                              |
|        n_samples=100000000, n_features=3,   |        n_samples=100000000, n_features=3,          |
|        centers=[[3, 3, 3], [0, 0, 0],       |        centers=[[3, 3, 3], [0, 0, 0],              |
|                 [1, 1, 1], [2, 2, 2]],      |                  [1, 1, 1], [2, 2, 2]],            |
|        cluster_std=[0.2, 0.1, 0.2, 0.2],    |        cluster_std=[0.2, 0.1, 0.2, 0.2],           |
|        random_state=9)                      |        random_state=9)                             |
|    pca = PCA(n_components=3)                |    pca = PCA(n_components=3)                       |
|    pca.fit(X)                               |    pca.fit(X)                                      |
|    print(pca.explained_variance_ratio_)     |    print(pca.explained_variance_ratio_)            |
|    print(pca.explained_variance_)           |    print(pca.explained_variance_)                  |
|                                             |                                                    |
+---------------------------------------------+----------------------------------------------------+

Mars remote
-----------

:doc:`documentation <remote/index>`

Mars remote allows users to execute functions in parallel.

+-------------------------------------------+--------------------------------------------+
| **Vanilla function calls**                | **Mars remote**                            |
+-------------------------------------------+--------------------------------------------+
|.. code-block:: python                     |.. code-block:: python                      |
|                                           |                                            |
|    import numpy as np                     |    import numpy as np                      |
|                                           |    import mars.remote as mr                |
|                                           |                                            |
|    def calc_chunk(n, i):                  |    def calc_chunk(n, i):                   |
|        rs = np.random.RandomState(i)      |        rs = np.random.RandomState(i)       |
|        a = rs.uniform(-1, 1, size=(n, 2)) |        a = rs.uniform(-1, 1, size=(n, 2))  |
|        d = np.linalg.norm(a, axis=1)      |        d = np.linalg.norm(a, axis=1)       |
|        return (d < 1).sum()               |        return (d < 1).sum()                |
|                                           |                                            |
|    def calc_pi(fs, N):                    |    def calc_pi(fs, N):                     |
|        return sum(fs) * 4 / N             |        return sum(fs) * 4 / N              |
|                                           |                                            |
|    N = 200_000_000                        |    N = 200_000_000                         |
|    n = 10_000_000                         |    n = 10_000_000                          |
|                                           |                                            |
|    fs = [calc_chunk(n, i)                 |    fs = [mr.spawn(calc_chunk, args=(n, i)) |
|          for i in range(N // n)]          |          for i in range(N // n)]           |
|    pi = calc_pi(fs, N)                    |    pi = mr.spawn(calc_pi, args=(fs, N))    |
|    print(pi)                              |    print(pi.execute().fetch())             |
|                                           |                                            |
+-------------------------------------------+--------------------------------------------+
|.. code-block::                            |.. code-block::                             |
|                                           |                                            |
|    3.1416312                              |    3.1416312                               |
|    CPU times: user 32.2 s, sys: 4.86 s,   |    CPU times: user 16.9 s, sys: 5.46 s,    |
|               total: 37.1 s               |               total: 22.3 s                |
|    Wall time: 12.4 s                      |    Wall time: 4.83 s                       |
|                                           |                                            |
+-------------------------------------------+--------------------------------------------+

Easy to scale in and scale out
------------------------------

Mars can scale in to a single machine, and scale out to a cluster with hundreds of machines.
Both the local and distributed version share the same piece of code,
it's fairly simple to migrate from a single machine to a cluster due to the increase of data.

Mars can run in a few ways:

- :ref:`Local thread-based scheduling <threaded>`
- :ref:`Local process-basesd scheduling <local_cluster>`
- :ref:`Run on cluster <deploy>`
- :ref:`Run on Kubernetes <k8s>`


.. toctree::
   :maxdepth: 2
   :caption: Install
   :hidden:

   install
   kubernetes

.. toctree::
   :maxdepth: 2
   :caption: Getting started
   :hidden:

   getting-started
   execution
   gpu

.. toctree::
   :maxdepth: 2
   :caption: Tensor Interface
   :hidden:

   tensor/overview
   tensor/datasource
   tensor/ufunc
   tensor/routines
   tensor/sparse

.. toctree::
   :maxdepth: 2
   :caption: DataFrame Interface
   :hidden:

   dataframe/user_guide/10min
   dataframe/reference/index

.. toctree::
   :maxdepth: 2
   :caption: Learn Interface
   :hidden:

   learn/tensorflow
   learn/xgboost
   learn/lightgbm
   learn/reference

.. toctree::
   :maxdepth: 2
   :caption: Remote Interface
   :hidden:

   remote/guide

.. toctree::
   :maxdepth: 2
   :caption: Distributed Scheduling
   :hidden:

   distributed/architecture
   distributed/prepare
   distributed/schedule-policy
   distributed/states
   distributed/worker-schedule
   distributed/fault-tolerance

.. toctree::
   :maxdepth: 2
   :caption: Contribution Guide
   :hidden:

   contributing
