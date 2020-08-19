.. image:: https://raw.githubusercontent.com/mars-project/mars/master/docs/source/images/mars-logo-title.png

|PyPI version| |Docs| |Build| |Coverage| |Quality| |License|

Mars is a tensor-based unified framework for large-scale data computation
which scales Numpy, Pandas and Scikit-learn.

`Documentation`_, `中文文档`_

Installation
------------

Mars is easy to install by

.. code-block:: bash

    pip install pymars

When you need to install dependencies needed by the distributed version, you can use the command below.

.. code-block:: bash

    pip install 'pymars[distributed]'

For now, distributed version is only available on Linux and Mac OS.


Developer Install
`````````````````

When you want to contribute code to Mars, you can follow the instructions below to install Mars
for development:

.. code-block:: bash

    git clone https://github.com/mars-project/mars.git
    cd mars
    pip install -e ".[dev]"

More details about installing Mars can be found at
`getting started <https://docs.pymars.org/en/latest/install.html>`_ section in
Mars document.


Mars tensor
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
|    3.14151712                                 |     3.14161908                                |
|    CPU times: user 12.5 s, sys: 7.16 s,       |     CPU times: user 17.5 s, sys: 3.56 s,      |
|               total: 19.7 s                   |                total: 21.1 s                  |
|    Wall time: 21.8 s                          |     Wall time: 5.59 s                         |
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
|    CPU times: user 10.9 s, sys: 2.69 s, |    CPU times: user 16.5 s, sys: 3.52 s, |
|               total: 13.6 s             |               total: 20 s               |
|    Wall time: 11 s                      |    Wall time: 3.6 s                     |
+-----------------------------------------+-----------------------------------------+


Mars learn
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
|    CPU times: user 32.2 s, sys: 4.86 s,   |    CPU times: user 16.9 s, sys: 5.46 s,    |
|               total: 37.1 s               |               total: 22.3 s                |
|    Wall time: 12.4 s                      |    Wall time: 4.83 s                       |
|                                           |                                            |
+-------------------------------------------+--------------------------------------------+


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


Easy to scale in and scale out
------------------------------

Mars can scale in to a single machine, and scale out to a cluster with thousands of machines.
Both the local and distributed version share the same piece of code,
it's fairly simple to migrate from a single machine to a cluster due to the increase of data.

Running on a single machine including thread-based scheduling,
local cluster scheduling which bundles the whole distributed components.
Mars is also easy to scale out to a cluster by starting different components of
mars distributed runtime on different machines in the cluster.

Threaded
````````

``execute`` method will by default run on the thread-based scheduler on a single machine.

.. code-block:: python

    >>> import mars.tensor as mt
    >>> a = mt.ones((10, 10))
    >>> a.execute()

Users can create a session explicitly.

.. code-block:: python

    >>> from mars.session import new_session
    >>> session = new_session()
    >>> (a * 2).execute(session=session)
    >>> # session will be released when out of with statement
    >>> with new_session() as session2:
    >>>     (a / 3).execute(session=session2)


Local cluster
`````````````

Users can start the local cluster bundled with the distributed runtime on a single machine.
Local cluster mode requires mars distributed version.

.. code-block:: python

    >>> from mars.deploy.local import new_cluster

    >>> # cluster will create a session and set it as default
    >>> cluster = new_cluster()

    >>> # run on the local cluster
    >>> (a + 1).execute()

    >>> # create a session explicitly by specifying the cluster's endpoint
    >>> session = new_session(cluster.endpoint)
    >>> (a * 3).execute(session=session)


Distributed
```````````

After installing the distributed version on every node in the cluster,
A node can be selected as scheduler and another as web service,
leaving other nodes as workers.  The scheduler can be started with the following command:

.. code-block:: bash

    mars-scheduler -a <scheduler_ip> -p <scheduler_port>

Web service can be started with the following command:

.. code-block:: bash

    mars-web -a <web_ip> -s <scheduler_endpoint> --ui-port <ui_port_exposed_to_user>

Workers can be started with the following command:

.. code-block:: bash

    mars-worker -a <worker_ip> -p <worker_port> -s <scheduler_endpoint>

After all mars processes are started, users can run

.. code-block:: python

    >>> sess = new_session('http://<web_ip>:<ui_port>')
    >>> a = mt.ones((2000, 2000), chunk_size=200)
    >>> b = mt.inner(a, a)
    >>> b.execute(session=sess)


Getting involved
----------------

- Read `development guide <https://docs.pymars.org/en/latest/development/index.html>`_.
- Join the mailing list: send an email to `mars-dev@googlegroups.com`_.
- Please report bugs by submitting a `GitHub issue`_.
- Submit contributions using `pull requests`_.

Thank you in advance for your contributions!


.. |Build| image:: https://github.com/mars-project/mars/workflows/Mars%20CI/badge.svg
   :target: https://github.com/mars-project/mars/actions
.. |Coverage| image:: https://codecov.io/gh/mars-project/mars/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/mars-project/mars
.. |Quality| image:: https://img.shields.io/codacy/grade/4e15343492d14335847d67630bb3c319.svg
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
