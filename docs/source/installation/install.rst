.. _local:

Installation
============

You can simply install Mars via pip:

.. code-block:: bash

    pip install pymars


Create a default session
------------------------

Mars needs to create a session to run tasks. You can use ``mars.new_session()``
to create a local session:

.. code-block:: python

    import mars
    mars.new_session()

Then calling ``execute()`` on a Mars objects e.g. tensor,
computation will be performed.

.. code-block:: python

    a = mt.random.rand(10, 10)
    a.dot(a.T).execute()

If no default session is created, Mars will create a session for you when executing a task.
You will see messages when the task is being created.

When you finish working with the local session, you can stop it using ``mars.stop_server()``:

.. code-block:: python

    mars.stop_server()
