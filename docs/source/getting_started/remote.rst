.. _getting_started_remote:

Mars Remote
===========

.. Note:: New in version 0.4.1

Mars remote provides a simple but powerful way to execute Python functions in
parallel.

Assume we have the code below.

.. code-block:: python

   >>> def add_one(x):
   >>>     return x + 1
   >>>
   >>> def sum_all(xs):
   >>>     return sum(xs)
   >>>
   >>> x_list = []
   >>> for i in range(10):
   >>>     x_list.append(add_one(i))
   >>>
   >>> print(sum_all(x_list))
   55

Here we call ``add_one`` 10 times, then call ``sum_all`` to get the summation.

In order to make 10 ``add_one`` running in parallel, we can rewrite the code as
below.

.. code-block:: python

   >>> import mars.remote as mr
   >>>
   >>> def add_one(x):
   >>>     return x + 1
   >>>
   >>> def sum_all(xs):
   >>>     return sum(xs)
   >>>
   >>> x_list = []
   >>> for i in range(10):
   >>>    x_list.append(mr.spawn(add_one, args=(i,)))
   >>> print(mr.spawn(sum_all, args=(x_list,)).execute().fetch())
   55

The code is quite similar with the previous one, except that calls to ``add_one``
and ``sum_all`` is replaced by ``mars.remote.spawn``. ``mars.remote.spawn`` does not
trigger execution, but instead returns a Mars Object, and the object can be
passed to another ``mars.remote.spawn`` as an argument. Once ``.execute()`` is
triggered, the 10 ``add_one`` will run in parallel.  Once they were finished,
``sum_all`` will be triggered. Mars can handle the dependencies correctly, and
for the distributed setting, Users need not to worry about the data movements
between different workers, Mars can handle them automatically.

Refer to :ref:`guidance for Mars remote <remote>` for more information.
