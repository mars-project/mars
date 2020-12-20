.. _mars_yarn:

Run on YARN
===========

Mars can be deployed on `YARN
<https://hadoop.apache.org/docs/current/hadoop-yarn/hadoop-yarn-site/YARN.html>`_
clusters. You can use ``mars.deploy.yarn`` to start Mars clusters in Hadoop
environments.

Basic steps
-----------

Mars uses `Skein <https://jcrist.github.io/skein/>`_ to deploy itself into YARN
clusters.  This library bridges Java interfaces of YARN applications and Python
interfaces.

Before starting Mars in YARN, you need to check your environments first. As
Skein supports Linux only, you need to work on a Linux client, otherwise you
need to fix and compile a number of packages yourself. Skein library is also
needed on client side. You may install Skein with conda

.. code-block:: bash

    conda install -c conda-forge skein

or install with pip

.. code-block:: bash

    pip install skein

Then you need to check Python environment inside your cluster. If you have a
Python environment installed within your YARN nodes with every required
packages installed, it will save a lot of time for you to start your cluster.
Otherwise you need to pack your local environment and specify it to Mars.

You may use `conda-pack <https://conda.github.io/conda-pack/>`_ to pack your
environment when you are using Conda:

.. code-block:: bash

    conda activate local-env
    conda install -c conda-forge conda-pack
    conda-pack

or use `venv-pack <https://jcrist.github.io/venv-pack/>`_ to pack your
environment when you are using virtual environments:

.. code-block:: bash

    source local-env/bin/activate
    pip install venv-pack
    venv-pack

Both commands will create a ``tar.gz`` archive, and you can use it when
deploying your Mars cluster.

Then it is time to start your Mars cluster. Select different lines when you are
starting from existing a conda environment, virtual environment, Python
executable or pre-packed environment archive:

.. code-block:: python

    import os
    from mars.deploy.yarn import new_cluster

    # specify location of Hadoop and JDK on client side
    os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-1.8.0-openjdk'
    os.environ['HADOOP_HOME'] = '/usr/local/hadoop'
    os.environ['PATH'] = '/usr/local/hadoop:' + os.environ['PATH']

    # use a conda environment at /path/to/remote/conda/env
    cluster = new_cluster(enviromnent='conda:///path/to/remote/conda/env')

    # use a virtual environment at /path/to/remote/virtual/env
    cluster = new_cluster(enviromnent='venv:///path/to/remote/virtual/env')

    # use a remote python executable
    cluster = new_cluster(enviromnent='python:///path/to/remote/python')

    # use a local packed environment archive
    cluster = new_cluster(enviromnent='path/to/local/env/pack.tar.gz')

    # get web endpoint, may be used elsewhere
    from mars.session import Session
    print(Session.default_or_local().endpoint)

    # new cluster will start a session and set it as default one
    # execute will then run in the local cluster
    a = mt.random.rand(10, 10)
    a.dot(a.T).execute()

    # after all jobs executed, you can turn off the cluster
    cluster.stop()

Customizing cluster
-------------------
``new_cluster`` function provides several keyword arguments for users to define
the cluster. You may use the argument ``app_name`` to customize the name of the
Yarn application, or use the argument ``timeout`` to specify timeout of cluster
creation.  Arguments for scaling up and out of the cluster are also available.

Arguments for schedulers:

+----------------------+------------------------------------------------------------+
| Argument             | Description                                                |
+======================+============================================================+
| scheduler_num        | Number of schedulers in the cluster, 1 by default          |
+----------------------+------------------------------------------------------------+
| scheduler_cpu        | Number of CPUs for every scheduler                         |
+----------------------+------------------------------------------------------------+
| scheduler_mem        | Memory size for schedulers in the cluster, in bytes or size|
|                      | units like ``1g``                                          |
+----------------------+------------------------------------------------------------+
| scheduler_extra_env  | A dict of environment variables to set in schedulers       |
+----------------------+------------------------------------------------------------+

Arguments for workers:

+--------------------+----------------------------------------------------------------+
| Argument           | Description                                                    |
+====================+================================================================+
| worker_num         | Number of workers in the cluster, 1 by default                 |
+--------------------+----------------------------------------------------------------+
| worker_cpu         | Number of CPUs for every worker                                |
+--------------------+----------------------------------------------------------------+
| worker_mem         | Memory size for workers in the cluster, in bytes or size units |
|                    | like ``1g``                                                    |
+--------------------+----------------------------------------------------------------+
| worker_spill_paths | List of spill paths for worker pods on hosts                   |
+--------------------+----------------------------------------------------------------+
| worker_cache_mem   | Size or ratio of shared memory for every worker. Details about |
|                    | memory management of Mars workers can be found in :ref:`memory |
|                    | tuning <worker_memory_tuning>` section.                        |
+--------------------+----------------------------------------------------------------+
| min_worker_num     | Minimal number of ready workers for ``new_cluster`` to return, |
|                    | ``worker_num`` by default                                      |
+--------------------+----------------------------------------------------------------+
| worker_extra_env   | A dict of environment variables to set in workers.             |
+--------------------+----------------------------------------------------------------+

Arguments for web services:

+------------------+----------------------------------------------------------------+
| Argument         | Description                                                    |
+==================+================================================================+
| web_num          | Number of web services in the cluster, 1 by default            |
+------------------+----------------------------------------------------------------+
| web_cpu          | Number of CPUs for every web service                           |
+------------------+----------------------------------------------------------------+
| web_mem          | Memory size for web services in the cluster, in bytes or size  |
|                  | units like ``1g``                                              |
+------------------+----------------------------------------------------------------+
| web_extra_env    | A dict of environment variables to set in web services.        |
+------------------+----------------------------------------------------------------+

For instance, if you want to create a Mars cluster with 1 scheduler, 1 web
service and 100 workers, each worker has 4 cores and 16GB memory, and stop
waiting when 95 workers are ready, you can use the code below:

.. code-block:: python

    import os
    from mars.deploy.yarn import new_cluster

    os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-1.8.0-openjdk'
    os.environ['HADOOP_HOME'] = '/usr/local/hadoop'

    cluster = new_cluster('path/to/env/pack.tar.gz', scheduler_num=1, web_num=1,
                          worker_num=100, worker_cpu=4, worker_mem='16g',
                          min_worker_num=95)
