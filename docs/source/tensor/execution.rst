Local Execution
===============

When :doc:`eager mode <eager-mode>` is not enabled, which is the default
behavior, Mars tensor will not be executed unless users call ``execute`` or
``session.run`` methods.

If no session is created explicitly, the ``execute`` will create a local
session, and mark it as a default session.

Session
-------

Users can create a new session by ``new_session`` method, if no argument is
provided, a local session will be generated.

.. code-block:: python

    >>> from mars.session import new_session

    >>> sess = new_session()  # create a session


By calling ``as_default`` of a session, the session will be marked as the
default session.


.. code-block:: python

    >>> sess.as_default()


More than one mars tensors can be passed to ``session.run``, and calculate the
results for each tensor.

.. code-block:: python

   >>> a = mt.ones((5, 5), chunk_size=3)
   >>> b = a + 1
   >>> c = a * 4
   >>> sess.run(b, c)
   (array([[2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 2.]]), array([[4., 4., 4., 4., 4.],
        [4., 4., 4., 4., 4.],
        [4., 4., 4., 4., 4.],
        [4., 4., 4., 4., 4.],
        [4., 4., 4., 4., 4.]]))


Execute a tensor
----------------

For a single tensor, ``execute`` can be called.

.. code-block:: python

   >>> a = mt.random.rand(3, 4)
   >>> a.sum().execute()
   7.0293719034458455

Session can be specified by the argument ``session``.

.. code-block:: python

   >>> a.sum().execute(session=sess)
   6.12833989477539
