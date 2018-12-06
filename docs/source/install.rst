Standalone install
==================
You can install Mars via pip:

.. code-block:: bash

    pip install pymars

After installation, you can simply open a Python console and run

.. code-block:: python

    import mars.tensor as mt
    from mars.session import new_session
    sess = new_session()
    a = mt.ones((5, 5), chunks=3)
    b = a * 4
    sess.run(b)

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
    a = mt.ones((2000, 2000), chunks=200)
    b = mt.inner(a, a)
    sess.run(b)

You can open a web browser and type ``http://<web_ip>:<ui_port>`` to open Mars
UI to look up resource usage of workers and execution progress of the task
submitted just now.
