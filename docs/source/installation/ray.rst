.. _mars_ray:

Run on Ray
=================

Mars also has deep integration with Ray and can run on `Ray <https://docs.ray.io/en/latest/>`_ efficiently and natively.

Basic steps
-----------
Install Ray locally:

.. code-block:: bash

    pip install ray

(Optional) Start a Ray cluster or Mars starts a Ray cluster automatically:

.. code-block:: python

    import ray
    ray.init()

(Optional) Or connecting to a existing Ray cluster using `Ray client <https://docs.ray.io/en/latest/cluster/ray-client.html>`_:

.. code-block:: python

    import ray
    ray.init(address='ray://<head_node_host>:10001')

Creating a Mars on Ray runtime in the Ray cluster and do the computing:

.. code-block:: python

    import mars
    import mars.tensor as mt
    import mars.dataframe as md
    # This driver is the Mars supervisor.
    session = mars.new_session(backend='ray')
    mt.random.RandomState(0).rand(1000_0000, 5).sum().execute()
    df = md.DataFrame(
        mt.random.rand(1000_0000, 4, chunk_size=500_0000),
        columns=list('abcd'))
    print(df.sum().execute())
    print(df.describe().execute())
    # Convert mars dataframe to ray dataset
    ds = md.to_ray_dataset(df)
    print(ds.schema(), ds.count())
    ds.filter(lambda row: row['a'] > 0.5).show(5)
    # Convert ray dataset to mars dataframe
    df2 = md.read_ray_dataset(ds)
    print(df2.head(5).execute())


Stop the created Mars on Ray runtime:

.. code-block:: python

    session.stop_server()


Customizing cluster
-------------------

There are two ways to initialize a Mars on Ray session:

- `mars.new_session(...) # Start Mars supervisor in current process.`
    Recommend for most use cases.
- `mars.new_ray_session(...) # Start a Ray actor for Mars supervisor.`
    Recommend for large scale compute or compute through Ray client.


Start a Ray actor for Mars supervisor:

.. code-block:: python

    import mars
    # Start a Ray actor for Mars supervisor.
    session = mars.new_ray_session(backend='ray')

Connect to the created Mars on Ray runtime and do the computing, the supervisor virtual address is the name of Ray actor for Mars supervisor,
e.g. `ray://ray-cluster-1672904753/0/0`.

.. code-block:: python

    import mars
    import mars.tensor as mt
    # Be aware that `mars.new_ray_session()` connects to an existing Mars
    # cluster requires Ray runtime.
    # e.g. Current process is a initialized Ray driver, client or worker.
    session = mars.new_ray_session(
        address='ray://<supervisor virtual address>',
        session_id='abcd',
        backend='ray',
        default=True)
    session.execute(mt.random.RandomState(0).rand(100, 5).sum())

The ``new_ray_session`` function provides several keyword arguments for users to define
the cluster.

Arguments for supervisors:

+--------------------+-----------------------------------------------------------------+
| Argument           | Description                                                     |
+====================+=================================================================+
| supervisor_cpu     | Number of CPUs for supervisor, 1 by default.                    |
+--------------------+-----------------------------------------------------------------+
| supervisor_mem     | Memory size for supervisor in bytes, 1G by default.             |
+--------------------+-----------------------------------------------------------------+

Arguments for workers:

+--------------------+-----------------------------------------------------------------+
| Argument           | Description                                                     |
+====================+=================================================================+
| worker_cpu         | Number of CPUs for every worker, 2 by default.                  |
+--------------------+-----------------------------------------------------------------+
| worker_mem         | Memory size for workers in bytes, 2G by default.                |
+--------------------+-----------------------------------------------------------------+

For instance, if you want to create a Mars cluster with a standalone supervisor,
you can use the code below (In this example, one Ray node has 16 CPUs in total):

.. code-block:: python

    import mars
    session = mars.new_ray_session(supervisor_cpu=16)
