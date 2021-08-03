.. _local:

Installation
============

You can simply install Mars via pip:

.. code-block:: bash

    pip install pymars


Create a default session
------------------------

.. code-block:: python

    import mars
    mars.new_session()

Then calling ``execute()`` on a Mars objects e.g. tensor,
computation will be performed.


.. code-block:: python

    a = mt.random.rand(10, 10)
    a.dot(a.T).execute()

