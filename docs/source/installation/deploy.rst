.. _deploy:

Run on Clusters
===============

Basic Steps
-----------

Mars can be deployed on a cluster. First, you need to run

.. code-block:: bash

    pip install pymars

on every node in the cluster. This will install dependencies needed for
distributed execution on your cluster. After that, you may select a node as
supervisor which also integrated web service, leaving other nodes as workers.

The supervisor can be started with the following command:

.. code-block:: bash

    mars-supervisor -H <host_name> -p <supervisor_port> -w <web_port>

Web service will be started as well.

Workers can be started with the following command:

.. code-block:: bash

    mars-worker -H <host_name> -p <worker_port> -s <supervisor_ip>:<supervisor_port>

After all Mars processes are started, you can open a Python console and run

.. code-block:: python

    import mars
    import mars.tensor as mt
    import mars.dataframe as md
    # create a default session that connects to the cluster
    mars.new_session('http://<web_ip>:<web_port>')
    a = mt.random.rand(2000, 2000, chunk_size=200)
    b = mt.inner(a, a)
    b.execute()  # submit tensor to cluster
    df = md.DataFrame(a).sum()
    df.execute()  # submit DataFrame to cluster


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
| ``-H``           | Service IP binding, ``0.0.0.0`` by default                     |
+------------------+----------------------------------------------------------------+
| ``-p``           | Port of the service. If absent, a randomized port will be used |
+------------------+----------------------------------------------------------------+
| ``-f``           | Path to service configuration file. Absent when use default    |
|                  | configuration.                                                 |
+------------------+----------------------------------------------------------------+
| ``-s``           | List of supervisor endpoints, separated by commas. Useful for  |
|                  | workers and webs to spot supervisors, or when you want to run  |
|                  | more than one supervisor                                       |
+------------------+----------------------------------------------------------------+
| ``--log-level``  | Log level, can be ``debug``, ``info``, ``warning``, ``error``  |
+------------------+----------------------------------------------------------------+
| ``--log-format`` | Log format, can be Python logging format                       |
+------------------+----------------------------------------------------------------+
| ``--log-conf``   | Python logging configuration file, ``logging.conf`` by default |
+------------------+----------------------------------------------------------------+
| ``--use-uvloop`` | Whether to use ``uvloop`` to accelerate, ``auto`` by default   |
+------------------+----------------------------------------------------------------+

Extra arguments for supervisors are listed below.

+------------------+----------------------------------------------------------------+
| Argument         | Description                                                    |
+==================+================================================================+
| ``-w``           | Port of web service in supervisor                              |
+------------------+----------------------------------------------------------------+

Extra arguments for workers are listed below. Details about memory tuning can
be found at the next section.

.. _deploy_extra_arguments:

+--------------------+----------------------------------------------------------------+
| Argument           | Description                                                    |
+====================+================================================================+
| ``--n-cpu``        | Number of CPU cores to use. If absent, the value will be       |
|                    | the available number of cores                                  |
+--------------------+----------------------------------------------------------------+
| ``--n-io-process`` | Number of IO processes for network operations. 1 by default    |
+--------------------+----------------------------------------------------------------+
| ``--cuda-devices`` | Index of CUDA devices to use. If not specified, all devices    |
|                    | will be used. Specifying an empty string will ignore all       |
|                    | devices                                                        |
+--------------------+----------------------------------------------------------------+

For instance, if you want to start a Mars cluster with two supervisors and two
workers, you can run commands below (memory and CPU tunings are omitted):

On Supervisor 1 (192.168.1.10):

.. code-block:: bash

    mars-supervisor -H 192.168.1.10 -p 7001 -w 7005 -s 192.168.1.10:7001,192.168.1.11:7002

On Supervisor 2 (192.168.1.11):

.. code-block:: bash

    mars-supervisor -H 192.168.1.11 -p 7002 -s 192.168.1.10:7001,192.168.1.11:7002

On Worker 1 (192.168.1.20):

.. code-block:: bash

    mars-worker -H 192.168.1.20 -p 7003 -s 192.168.1.10:7001,192.168.1.11:7002

On Worker 2 (192.168.1.21):

.. code-block:: bash

    mars-worker -H 192.168.1.21 -p 7004 -s 192.168.1.10:7001,192.168.1.11:7002
