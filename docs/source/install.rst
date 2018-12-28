Standalone mode
===============

Threaded
--------

You can install Mars via pip:

.. code-block:: bash

    pip install pymars

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
    sess.run(b)  # run b


Local cluster
-------------

Users can start the distributed runtime of Mars on a single machine.
First, install Mars distributed by run

.. code-block:: bash

    pip install 'pymars[distributed]'

For now, local cluster mode can only run on Linux and Mac OS.

Then start a local cluster by run

.. code-block:: python

    from mars.deploy.local import new_cluster

    cluster = new_cluster()

    # new cluster will start a session and set it as default one
    # execute will then run in the local cluster
    a = mt.random.rand(10, 10)
    a.dot(a.T).execute()

    # cluster.session is the session created
    cluster.session.run(a + 1)

    # users can also create a session explicitly
    # cluster.endpoint needs to be passed to new_session
    session2 = new_session(cluster.endpoint)
    session2.run(a * 2)


Run on Clusters
===============
Mars can be deployed on a cluster. First, yu need to run

.. code-block:: bash

    pip install 'pymars[distributed]'

on every node in the cluster. This will install dependencies needed for
distributed execution on your cluster. After that, you may select a node as
scheduler and another as web service, leaving other nodes as workers.  The
scheduler can be started with the following command:

.. code-block:: bash

    mars-scheduler -a <scheduler_ip> -p <scheduler_port>

Web service can be started with the following command:

.. code-block:: bash

    mars-web -a <web_ip> -s <scheduler_ip> -p <communicator_port> --ui-port <ui_port_exposed_to_user>

Workers can be started with the following command:

.. code-block:: bash

    mars-worker -a <worker_ip> -p <worker_port> -s <scheduler_ip>

After all Mars processes are started, you can open a Python console and run

.. code-block:: python

    import mars.tensor as mt
    from mars.session import new_session
    sess = new_session('http://<web_ip>:<ui_port>')
    a = mt.ones((2000, 2000), chunk_size=200)
    b = mt.inner(a, a)
    sess.run(b)

You can open a web browser and type ``http://<web_ip>:<ui_port>`` to open Mars
UI to look up resource usage of workers and execution progress of the task
submitted just now.
