Mars
====

|PyPI version| |Docs| |Build| |Coverage| |Quality| |License|

Mars is a tensor-based unified framework for large-scale data computation. `Documentation`_.

Installation
------------

Mars is easy to install by

.. code-block:: bash

    pip install pymars

The distributed version can be installed by

.. code-block:: bash

    pip install 'pymars[distributed]'

For now, distributed version is only available on Linux and Mac OS.


Mars tensor
-----------

Mars tensor provides a familiar interface like Numpy.

+------------------------------------------------+----------------------------------------------------+
| **Numpy**                                      | **Mars tensor**                                    |
+------------------------------------------------+----------------------------------------------------+
|.. code-block:: python                          |.. code-block:: python                              |
|                                                |                                                    |
|    import numpy as np                          |    import mars.tensor as mt                        |
|    a = np.random.rand(1000, 2000)              |    a = mt.random.rand(1000, 2000)                  |
|    (a + 1).sum(axis=1)                         |    (a + 1).sum(axis=1).execute()                   |
|                                                |                                                    |
+------------------------------------------------+----------------------------------------------------+


The following is a brief overview of supported subset of Numpy interface.

- Arithmetic and mathematics: ``+``, ``-``, ``*``, ``/``, ``exp``, ``log``, etc.
- Reduction along axes (``sum``, ``max``, ``argmax``, etc).
- Most of the `array creation routines <https://docs.scipy.org/doc/numpy/reference/routines.array-creation.html>`_
  (``empty``, ``ones_like``, ``diag``, etc). What's more, Mars does not only support create array/tensor on GPU,
  but also support create sparse tensor.
- Most of the `array manipulation routines <https://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html>`_
  (``reshape``, ``rollaxis``, ``concatenate``, etc.)
- `Basic indexing <https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`_
  (indexing by ints, slices, newaxes, and Ellipsis)
- Fancy indexing along single axis with lists or numpy arrays, e.g. x[[1, 4, 8], :5]
- `universal functions <https://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_
  for elementwise operations.
- `Linear algebra functions <https://docs.scipy.org/doc/numpy/reference/routines.linalg.html>`_,
  including product (``dot``, ``matmul``, etc.) and decomposition (``cholesky``, ``svd``, etc.).

However, Mars has not implemented entire Numpy interface, either the time limitation or difficulty is the main handicap.
Any contribution from community is sincerely welcomed. The main feature not implemented are listed below:

- Tensor with unknown shape does not support all operations.
- Only small subset of ``np.linalg`` are implemented.
- Operations like ``sort`` which is hard to execute in parallel are not implemented.
- Mars tensor doesn't implement interface like ``tolist`` and ``nditer`` etc,
  because the iteration or loops over a large tensor is very inefficient.


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

    import mars.tensor as mt

    a = mt.ones((10, 10))
    a.execute()

Users can create a session explicitly.

.. code-block:: python

    from mars.session import new_session

    session = new_session()
    session.run(a + 1)
    (a * 2).execute(session=session)

    # session will be released when out of with statement
    with new_session() as session2:
        session2.run(a / 3)


Local cluster
`````````````

Users can start the local cluster bundled with the distributed runtime on a single machine.
Local cluster mode requires mars distributed version.

.. code-block:: python

    from mars.deploy.local import new_cluster

    # cluster will create a session and set it as default
    cluster = new_cluster()

    # run on the local cluster
    (a + 1).execute()

    # create a session explicitly by specifying the cluster's endpoint
    session = new_session(cluster.endpoint)
    session.run(a * 3)


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

    sess = new_session('http://<web_ip>:<ui_port>')
    a = mt.ones((2000, 2000), chunk_size=200)
    b = mt.inner(a, a)
    sess.run(b)


Getting involved
----------------

- Join the mailing list: send an email to `mars-dev@googlegroups.com`_.
- Please report bugs by submitting a `GitHub issue`_.
- Submit contributions using `pull requests`_.

Thank you in advance for your contributions!


.. |Build| image:: https://img.shields.io/travis/mars-project/mars.svg?style=flat-square
   :target: https://travis-ci.org/mars-project/mars
.. |Coverage| image:: https://img.shields.io/coveralls/github/mars-project/mars.svg?style=flat-square
   :target: https://coveralls.io/github/mars-project/mars
.. |Quality| image:: https://img.shields.io/codacy/grade/4e15343492d14335847d67630bb3c319.svg?style=flat-square
   :target: https://app.codacy.com/project/mars-project/mars/dashboard
.. |PyPI version| image:: https://img.shields.io/pypi/v/pymars.svg?style=flat-square
   :target: https://pypi.python.org/pypi/pymars
.. |Docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat-square
   :target: `Documentation`_
.. |License| image:: https://img.shields.io/pypi/l/pymars.svg?style=flat-square
   :target: https://github.com/mars-project/mars/blob/master/LICENSE
.. _`mars-dev@googlegroups.com`: https://groups.google.com/forum/#!forum/mars-dev
.. _`GitHub issue`: https://github.com/mars-project/mars/issues
.. _`pull requests`: https://github.com/mars-project/mars/pulls
.. _`Documentation`: https://docs.mars-project.io
