.. _local:

Installation
============

You can simply install Mars via pip:

.. code-block:: bash

    pip install pymars

To run Mars on a single machine, there are two ways.

* Threaded: a thread-based scheduling which is by default.
* Local cluster: a process-based scheduling which owns the entire distributed runtime.

.. _threaded:

Threaded
--------

After installation, you can simply open a Python console and run

.. code-block:: python

    import mars.tensor as mt
    from mars.session import new_session

    a = mt.ones((5, 5), chunk_size=3)
    b = a * 4
    # if there isn't a local session,
    # execute will create a default one first
    b.execute()

    # or create a session explicitly
    sess = new_session()
    b.execute(session=sess)  # run b


.. _local_cluster:

Local cluster
-------------

Users can start the distributed runtime of Mars on a single machine. First,
install Mars distributed by run

.. code-block:: bash

    pip install 'pymars[distributed]'

For now, local cluster mode can only run on Linux and Mac OS.

Then start a local cluster by run

.. code-block:: python

    import mars.tensor as mt
    from mars.deploy.local import new_cluster
    from mars.session import new_session

    cluster = new_cluster()

    # new cluster will start a session and set it as default one
    # execute will then run in the local cluster
    a = mt.random.rand(10, 10)
    a.dot(a.T).execute()

    # cluster.session is the session created
    (a + 1).execute(session=cluster.session)

    # users can also create a session explicitly
    # cluster.endpoint needs to be passed to new_session
    session2 = new_session(cluster.endpoint)
    (a * 2).execute(session=session2)
