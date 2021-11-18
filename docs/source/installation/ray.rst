.. _mars_ray:

Run on Ray
=================

Mars also has deep integration with Ray and can run on `Ray <https://docs.ray.io/en/latest/>`_ efficiently and natively.

Basic steps
-----------
Install Ray locally:

.. code-block:: bash

    pip install ray>=1.7.0

Start a Ray cluster:

.. code-block:: python

    import ray
    ray.init()

Or connecting to a existing Ray cluster using `Ray client <https://docs.ray.io/en/latest/cluster/ray-client.html>`_:

.. code-block:: python

    import ray
    ray.init(address="ray://<head_node_host>:10001")

Creating a Mars on Ray runtime in the Ray cluster and do the computing:

.. code-block:: python

    import mars
    import mars.tensor as mt
    import mars.dataframe as md
    session = mars.new_ray_session(worker_num=2, worker_mem=2 * 1024 ** 3)
    mt.random.RandomState(0).rand(1000_0000, 5).sum().execute()
    df = md.DataFrame(
        mt.random.rand(1000_0000, 4, chunk_size=500_0000),
        columns=list('abcd'))
    print(df.sum().execute())
    print(df.describe().execute())
    # Convert mars dataframe to ray dataset
    ds = md.to_ray_dataset(df)
    print(ds.schema(), ds.count())
    ds.filter(lambda row: row["a"] > 0.5).show(5)
    # Convert ray dataset to mars dataframe
    df2 = md.read_ray_dataset(ds)
    print(df2.head(5).execute())


Create a Mars on Ray runtime independently in the Ray cluster:

.. code-block:: python

    import mars
    import mars.tensor as mt
    cluster = mars.new_cluster_in_ray(worker_num=2, worker_mem=2 * 1024 ** 3)

Connect to the created Mars on Ray runtime and do the computing:

.. code-block:: python

    import mars
    import mars.tensor as mt
    session = mars.new_ray_session(address="http://ip:port", session_id="abcd", default=True)
    session.execute(mt.random.RandomState(0).rand(100, 5).sum())

Stop the created Mars on Ray runtime:

.. code-block:: python

    cluster.stop()


Customizing cluster
-------------------
``new_ray_session``/``new_cluster_in_ray`` function provides several keyword arguments for users to define
the cluster.

Arguments for supervisors:

+----------------------+-----------------------------------------------------------+
| Argument             | Description                                               |
+======================+===========================================================+
| supervisor_mem       | Memory size for supervisor in the cluster, in bytes.      |
+----------------------+-----------------------------------------------------------+

Arguments for workers:

+--------------------+-----------------------------------------------------------------+
| Argument           | Description                                                     |
+====================+=================================================================+
| worker_num         | Number of workers in the cluster, 1 by default.                 |
+--------------------+-----------------------------------------------------------------+
| worker_cpu         | Number of CPUs for every worker, 2 by default.                  |
+--------------------+-----------------------------------------------------------------+
| worker_mem         | Memory size for workers in the cluster, in bytes, 2G by default.|
+--------------------+-----------------------------------------------------------------+

For instance, if you want to create a Mars cluster with 100 workers,
each worker has 4 cores and 16GB memory, you can use the code below:

.. code-block:: python

    import mars
    import mars.tensor as mt
    cluster = mars.new_cluster_in_ray(worker_num=100, worker_cpu=4, worker_mem=16 * 1024 ** 3)
