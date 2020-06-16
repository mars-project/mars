.. _gpu:

Mars on GPU
===========

Mars can run on NVIDIA GPUs. However, extra requirements are necessary for different modules.

Installation
~~~~~~~~~~~~

For Mars tensors, CuPy is required. Assuming that your CUDA driver is 10.1, install cupy via:

.. code-block:: bash

   pip install cupy-cuda101

Refer to `install cupy <https://docs-cupy.chainer.org/en/stable/install.html>`_
for more information.

For Mars DataFrame, RAPIDS cuDF is required. Install cuDF via conda:

.. code-block:: bash

   conda install -c rapidsai -c nvidia -c conda-forge \
    -c defaults cudf=0.13 python=3.7 cudatoolkit=10.1

Refer to `install cuDF <https://rapids.ai/start.html#get-rapids>`_ for more information.

Mars tensor on CUDA
~~~~~~~~~~~~~~~~~~~

Tensor can be created on GPU via specifying ``gpu=True``.
Methods included are mentioned in :ref:`tensor creation <tensor_creation>` and
:ref:`random data <tensor_random>`.

.. code-block:: python

   >>> import mars.tensor as mt
   >>> a = mt.random.rand(10, 10, gpu=True)  # indicate to create tensor on CUDA
   >>> a.sum().execute()                     # execution will happen on CUDA

Remember that when creating tensors, no GPU memory allocation happens yet.
When ``.execute()`` is triggered, real memory allocation and computation on GPU will happen then.

For a tensor on host memory, call ``.to_gpu()`` to tell Mars to move data to GPU.

.. code-block:: python

   >>> b = mt.random.rand(10, 10)  # indicate to create on main memory
   >>> b = b.to_gpu()              # indicate to move data to GPU memory
   >>> b.sum().execute()

Call ``.to_cpu()`` to tell Mars to move data to host memory.

.. code-block:: python

   >>> c = b.to_cpu()     # b is allocated on GPU, move back to main memory
   >>> c.sum().execute()  # execution will happen on CPU

Mars DataFrame on CUDA
~~~~~~~~~~~~~~~~~~~~~~

Mars can read CSV files into GPU directly.

.. code-block:: python

   >>> import mars.dataframe as md
   >>> df = md.read_csv('data.csv', gpu=True)  # indicates to read csv into GPU memory
   >>> df.groupby('a').sum().execute()         # execution will happen on GPU

For a DataFrame that on host memory, call ``.to_gpu()`` to tell Mars to move data to GPU.

.. code-block:: python

   >>> import mars.tensor as mt
   >>> import mars.dataframe as md
   >>> df = md.DataFrame(mt.random.rand(10, 10))  # indicate to create on main memory
   >>> df = df.to_gpu()                            # indicate to move data to GPU memory

Call ``.to_cpu()`` to tell Mars to move data to host memory.

.. code-block:: python

   >>> df2 = df.to_cpu()     # df is allocated on GPU, move back to main memory
   >>> df2.sum().execute()     # execution will happen on CPU

Single GPU
~~~~~~~~~~

:ref:`Local thread-based scheduler <threaded>` can work well on a single GPU.
Examples above can work on a single GPU.

Multiple GPU
~~~~~~~~~~~~

For Mars tensor, multiple GPUs on a single machine can be utilized.

.. code-block:: python

   >>> import mars.tensor as mt
   >>> t = mt.random.rand(10000, 10000, gpu=True)
   >>> t.sum().execute()

The code above will try to leverage all the visible GPU cards to perform computation.

If you want to limit computation to some GPU cards,
you can set environment variable ``CUDA_VISIBLE_DEVICES``.

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0,3,5 ipython

This will limit the ipython to GPU 0, 3 and 5 only.
Thus all the Mars tensor executed in the ipython will run on the visible GPUs only.

For Mars DataFrame, local thread-based scheduler cannot leverage multiple GPUs
for DataFrame computation. In this case, please use distributed scheduler.

Distributed
~~~~~~~~~~~

For Mars scheduler and web, the command to start is the same. Refer to :ref:`deploy`.

For Mars worker, one worker can only bind to one GPU,
thus if you want to leverage multiple GPUs, please start as many workers as the count of GPUs.

Basic command to start a worker that binds to some GPU is:

.. code-block:: bash

   mars-worker -a <worker_ip> -p <worker_port> -s <scheduler_ip>:<scheduler_port> --cuda-device 0

The worker started will be bind to GPU 0.

Refer to :ref:`extra arguments for starting worker <deploy_extra_arguments>` for more information.

Once a Mars cluster is started, you can run the code below.

.. code-block:: python

   >>> import mars.tensor as mt
   >>> from mars.session import new_session
   >>> new_session('http://<web_ip>:<web_port>').as_default()
   >>> t = mt.random.rand(20, 20, gpu=True)
   >>> t.sum().execute()  # run on workers which are bind to GPU
