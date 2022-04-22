.. image:: https://raw.githubusercontent.com/mars-project/mars/master/docs/source/images/mars-logo-title.png

|PyPI version| |Docs| |Build| |Coverage| |Quality| |License|

Mars is a tensor-based unified framework for large-scale data computation
which scales numpy, pandas, scikit-learn and many other libraries.

`Documentation`_, `中文文档`_

Installation
------------

Mars is easy to install by

.. code-block:: bash

    pip install pymars


Installation for Developers
```````````````````````````

When you want to contribute code to Mars, you can follow the instructions below to install Mars
for development:

.. code-block:: bash

    git clone https://github.com/mars-project/mars.git
    cd mars
    pip install -e ".[dev]"

More details about installing Mars can be found at
`installation <https://docs.pymars.org/en/latest/installation/index.html>`_ section in
Mars document.


Architecture Overview
---------------------

.. image:: https://raw.githubusercontent.com/mars-project/mars/master/docs/source/images/architecture.png


Getting Started
---------------

Starting a new runtime locally via:

.. code-block:: python

    >>> import mars
    >>> mars.new_session()

Or connecting to a Mars cluster which is already initialized.

.. code-block:: python

    >>> import mars
    >>> mars.new_session('http://<web_ip>:<ui_port>')


Mars Tensor
-----------

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
|    3.14174502                                 |     3.14161908                                |
|    CPU times: user 11.6 s, sys: 8.22 s,       |     CPU times: user 966 ms, sys: 544 ms,      |
|               total: 19.9 s                   |                total: 1.51 s                  |
|    Wall time: 22.5 s                          |     Wall time: 3.77 s                         |
|                                               |                                               |
+-----------------------------------------------+-----------------------------------------------+

Mars can leverage multiple cores, even on a laptop, and could be even faster for a distributed setting.


Mars DataFrame
--------------

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
|    CPU times: user 10.9 s, sys: 2.69 s, |    CPU times: user 1.21 s, sys: 212 ms, |
|               total: 13.6 s             |               total: 1.42 s             |
|    Wall time: 11 s                      |    Wall time: 2.75 s                    |
+-----------------------------------------+-----------------------------------------+


Mars Learn
----------

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

Mars learn also integrates with many libraries:

- `TensorFlow <https://docs.pymars.org/en/latest/user_guide/learn/tensorflow.html>`_
- `PyTorch <https://docs.pymars.org/en/latest/user_guide/learn/pytorch.html>`_
- `XGBoost <https://docs.pymars.org/en/latest/user_guide/learn/xgboost.html>`_
- `LightGBM <https://docs.pymars.org/en/latest/user_guide/learn/lightgbm.html>`_
- `Joblib <https://docs.pymars.org/en/latest/user_guide/learn/joblib.html>`_
- `Statsmodels <https://docs.pymars.org/en/latest/user_guide/learn/statsmodels.html>`_

Mars remote
-----------

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
|    CPU times: user 32.2 s, sys: 4.86 s,   |    CPU times: user 616 ms, sys: 307 ms,    |
|               total: 37.1 s               |               total: 923 ms                |
|    Wall time: 12.4 s                      |    Wall time: 3.99 s                       |
|                                           |                                            |
+-------------------------------------------+--------------------------------------------+

DASK on Mars
------------

Refer to `DASK on Mars`_ for more information.

Eager Mode
```````````

Mars supports eager mode which makes it friendly for developing and easy to debug.

Users can enable the eager mode by options, set options at the beginning of the program or console session.

.. code-block:: python

    >>> from mars.config import options
    >>> options.eager_mode = True

Or use a context.

.. code-block:: python

    >>> from mars.config import option_context
    >>> with option_context() as options:
    >>>     options.eager_mode = True
    >>>     # the eager mode is on only for the with statement
    >>>     ...

If eager mode is on, tensor, DataFrame etc will be executed immediately
by default session once it is created.

.. code-block:: python

    >>> import mars.tensor as mt
    >>> import mars.dataframe as md
    >>> from mars.config import options
    >>> options.eager_mode = True
    >>> t = mt.arange(6).reshape((2, 3))
    >>> t
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> df = md.DataFrame(t)
    >>> df.sum()
    0    3
    1    5
    2    7
    dtype: int64


Mars on Ray
------------
Mars also has deep integration with Ray and can run on `Ray <https://docs.ray.io/en/latest/>`_ efficiently and
interact with the large ecosystem of machine learning and distributed systems built on top of the core Ray.

Starting a new Mars on Ray runtime locally via:

.. code-block:: python

    import ray
    ray.init()
    import mars
    mars.new_ray_session(worker_num=2)
    import mars.tensor as mt
    mt.random.RandomState(0).rand(1000_0000, 5).sum().execute()

Or connecting to a Mars on Ray runtime which is already initialized.

.. code-block:: python

    import mars
    mars.new_ray_session('http://<web_ip>:<ui_port>')
    # perform computation

Interact with Ray Dataset:

.. code-block:: python

    import mars.tensor as mt
    import mars.dataframe as md
    df = md.DataFrame(
        mt.random.rand(1000_0000, 4),
        columns=list('abcd'))
    # Convert mars dataframe to ray dataset
    ds = md.to_ray_dataset(df)
    print(ds.schema(), ds.count())
    ds.filter(lambda row: row["a"] > 0.5).show(5)
    # Convert ray dataset to mars dataframe
    df2 = md.read_ray_dataset(ds)
    print(df2.head(5).execute())

Refer to `Mars on Ray`_ for more information.


Easy to scale in and scale out
------------------------------

Mars can scale in to a single machine, and scale out to a cluster with thousands of machines.
It's fairly simple to migrate from a single machine to a cluster to
process more data or gain a better performance.


Bare Metal Deployment
`````````````````````

Mars is easy to scale out to a cluster by starting different components of
mars distributed runtime on different machines in the cluster.

A node can be selected as supervisor which integrated a web service,
leaving other nodes as workers.  The supervisor can be started with the following command:

.. code-block:: bash

    mars-supervisor -h <host_name> -p <supervisor_port> -w <web_port>

Workers can be started with the following command:

.. code-block:: bash

    mars-worker -h <host_name> -p <worker_port> -s <supervisor_endpoint>

After all mars processes are started, users can run

.. code-block:: python

    >>> sess = new_session('http://<web_ip>:<ui_port>')
    >>> # perform computation


Kubernetes Deployment
`````````````````````

Refer to `Run on Kubernetes`_ for more information.


Yarn Deployment
```````````````

Refer to `Run on Yarn`_ for more information.


Getting involved
----------------

- Read `development guide <https://docs.pymars.org/en/latest/development/index.html>`_.
- Join our Slack workgroup: `Slack <https://join.slack.com/t/mars-computing/shared_invite/zt-17pw2cfua-NRb2H4vrg77pr9T4g3nQOQ>`_.
- Join the mailing list: send an email to `mars-dev@googlegroups.com`_.
- Please report bugs by submitting a `GitHub issue`_.
- Submit contributions using `pull requests`_.

Thank you in advance for your contributions!


.. |Build| image:: https://github.com/mars-project/mars/workflows/Mars%20CI%20Core/badge.svg
   :target: https://github.com/mars-project/mars/actions
.. |Coverage| image:: https://codecov.io/gh/mars-project/mars/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/mars-project/mars
.. |Quality| image:: https://img.shields.io/codacy/grade/6a80bb4659ed410eb33795f580c8615e.svg
   :target: https://app.codacy.com/project/mars-project/mars/dashboard
.. |PyPI version| image:: https://img.shields.io/pypi/v/pymars.svg
   :target: https://pypi.python.org/pypi/pymars
.. |Docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg
   :target: `Documentation`_
.. |License| image:: https://img.shields.io/pypi/l/pymars.svg
   :target: https://github.com/mars-project/mars/blob/master/LICENSE
.. _`mars-dev@googlegroups.com`: https://groups.google.com/forum/#!forum/mars-dev
.. _`GitHub issue`: https://github.com/mars-project/mars/issues
.. _`pull requests`: https://github.com/mars-project/mars/pulls
.. _`Documentation`: https://docs.pymars.org
.. _`中文文档`: https://docs.pymars.org/zh_CN/latest/
.. _`Mars on Ray`: https://docs.pymars.org/en/latest/installation/ray.html
.. _`Run on Kubernetes`: https://docs.pymars.org/en/latest/installation/kubernetes.html
.. _`Run on Yarn`: https://docs.pymars.org/en/latest/installation/yarn.html
.. _`DASK on Mars`: https://docs.pymars.org/en/latest/user_guide/contrib/dask.html
