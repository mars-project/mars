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

Users can start the distributed runtime of Mars on a single machine.  First,
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
    cluster.session.run(a + 1)

    # users can also create a session explicitly
    # cluster.endpoint needs to be passed to new_session
    session2 = new_session(cluster.endpoint)
    session2.run(a * 2)


.. _deploy:

Run on Clusters
===============

Basic Steps
-----------

Mars can be deployed on a cluster. First, you need to run

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

    mars-web -a <web_ip> -p <web_port> -s <scheduler_ip>:<scheduler_port>

Workers can be started with the following command:

.. code-block:: bash

    mars-worker -a <worker_ip> -p <worker_port> -s <scheduler_ip>:<scheduler_port>

After all Mars processes are started, you can open a Python console and run

.. code-block:: python

    import mars.tensor as mt
    from mars.session import new_session
    sess = new_session('http://<web_ip>:<web_port>')
    a = mt.ones((2000, 2000), chunk_size=200)
    b = mt.inner(a, a)
    sess.run(b)

You can open a web browser and type ``http://<web_ip>:<web_port>`` to open Mars
UI to look up resource usage of workers and execution progress of the task
submitted just now.

Using Command Lines
-------------------
When running Mars with command line, you can specify arguments to control the
behavior of Mars processes. All Mars services have common arguments listed
below.

+------------------+----------------------------------------------------------------+
| Argument         | Description                                                    |
+==================+================================================================+
| ``-a``           | Advertise address exposed to other processes in the cluster,   |
|                  | useful when the server has multiple IP addresses, or the       |
|                  | service is deployed inside a VM or container                   |
+------------------+----------------------------------------------------------------+
| ``-H``           | Service IP binding, ``0.0.0.0`` by default                     |
+------------------+----------------------------------------------------------------+
| ``-p``           | Port of the service. If absent, a randomized port will be used |
+------------------+----------------------------------------------------------------+
| ``-s``           | List of scheduler endpoints, separated by commas. Useful for   |
|                  | workers and webs to spot schedulers, or when you want to run   |
|                  | more than one schedulers                                       |
+------------------+----------------------------------------------------------------+
| ``--log-level``  | Log level, can be ``debug``, ``info``, ``warning``, ``error``  |
+------------------+----------------------------------------------------------------+
| ``--log-format`` | Log format, can be Python logging format                       |
+------------------+----------------------------------------------------------------+
| ``--log-conf``   | Python logging configuration file, ``logging.conf`` by default |
+------------------+----------------------------------------------------------------+

Extra arguments for schedulers are listed below.

+------------------+----------------------------------------------------------------+
| Argument         | Description                                                    |
+==================+================================================================+
| ``--nproc``      | Number of processes. If absent, the value will be the          |
|                  | available number of cores                                      |
+------------------+----------------------------------------------------------------+

Extra arguments for workers are listed below. Details about memory tuning can
be found at the next section.

+-------------------+----------------------------------------------------------------+
| Argument          | Description                                                    |
+===================+================================================================+
| ``--cpu-procs``   | Number of computation processes on CPUs. If absent, the value  |
|                   | will be the available number of cores                          |
+-------------------+----------------------------------------------------------------+
| ``--net-procs``   | Number of processes for network transfer. 4 by default         |
+-------------------+----------------------------------------------------------------+
| ``--cuda-device`` | Index of the CUDA device to use. If not specified, CPUs will   |
|                   | be used only.                                                  |
+-------------------+----------------------------------------------------------------+
| ``--phy-mem``     | Limit of physical memory, can be percentages of total memory   |
|                   | or multiple of bytes. For instance, ``4g`` or ``80%`` are both |
|                   | acceptable. If absent, the size of physical memory will be     |
|                   | used                                                           |
+-------------------+----------------------------------------------------------------+
| ``--cache-mem``   | Size of shared memory, can be percentages of total memory or   |
|                   | multiple of bytes. For instance, ``4g`` or ``80%`` are both    |
|                   | acceptable. If absent, 50% of free memory will be used         |
+-------------------+----------------------------------------------------------------+
| ``--min-mem``     | Minimal free memory to start worker, can be percentages of     |
|                   | total memory or multiple of bytes. For instance, ``4g`` or     |
|                   | ``80%`` are both acceptable. ``128m`` by default               |
+-------------------+----------------------------------------------------------------+
| ``--spill-dir``   | Directories to spill to, separated by : in MacOS or Linux.     |
+-------------------+----------------------------------------------------------------+
| ``--plasma-dir``  | Directory of plasma store. When specified, the size of plasma  |
|                   | store will not be considered in memory management.             |
+-------------------+----------------------------------------------------------------+

For instance, if you want to start a Mars cluster with two schedulers, two
workers and one web service, you can run commands below (memory and CPU tunings
are omitted):

On Scheduler 1 (192.168.1.10):

.. code-block:: bash

    mars-scheduler -a 192.168.1.10 -p 7001 -s 192.168.1.10:7001,192.168.1.11:7002

On Scheduler 2 (192.168.1.11):

.. code-block:: bash

    mars-scheduler -a 192.168.1.11 -p 7002 -s 192.168.1.10:7001,192.168.1.11:7002

On Worker 1 (192.168.1.20):

.. code-block:: bash

    mars-worker -a 192.168.1.20 -p 7003 -s 192.168.1.10:7001,192.168.1.11:7002 \
        --spill-dirs /mnt/disk2/spill:/mnt/disk3/spill

On Worker 2 (192.168.1.21):

.. code-block:: bash

    mars-worker -a 192.168.1.21 -p 7004 -s 192.168.1.10:7001,192.168.1.11:7002 \
        --spill-dirs /mnt/disk2/spill:/mnt/disk3/spill

On the web server (192.168.1.30):

.. code-block:: bash

    mars-web -p 7005 -s 192.168.1.10:7001,192.168.1.11:7002

.. _worker_memory_tuning:

Memory Tuning
-------------
Mars worker manages two different parts of memory. The first is private process
memory and the second is shared memory between all worker processes handled by
`plasma_store in Apache Arrow
<https://arrow.apache.org/docs/python/plasma.html>`_. When Mars Worker starts,
it will take 50% of free memory space by default as shared memory and the left
as private process memory. What's more, Mars provides soft and hard memory
limits for memory allocations, which are 75% and 90% by default. If these
configurations does not meet your need, you can configure them when Mars Worker
starts. You can use ``--cache-mem`` argument to configure the size of shared
memory, ``--phy-mem`` to configure total memory size, from which the soft and
hard limits are computed.

For instance, by using

.. code-block:: bash

    mars-worker -a localhost -p 9012 -s localhost:9010 --cache-mem 512m --phy-mem 90%

We limit the size of shared memory as 512MB and the worker can use up to 90% of
total physical memory.
