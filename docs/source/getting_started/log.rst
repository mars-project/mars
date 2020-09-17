.. _getting_started_log:

Retrieving Log
==============

.. Note::
    New in version 0.5.2, only available for distributed scheduler.

For a distributed setting, the log output in custom functions, for instance,
submitted via ``mars.remote.spawn``, will not reveal on client,
Users may have to login to the workers to see the log.
Hence, ``fetch_log`` API is used to retrieve the log to client for display purpose.

For example, we define a function called ``calculate``.

.. code-block:: python

   >>> from mars.session import new_session
   >>> # create a distributed session
   >>> new_session('http://<ip>:<port>').as_default()
   >>>
   >>> import mars.remote as mr
   >>> def calculate(x):
   >>>     acc = 0
   >>>     for i in range(x):
   >>>         acc += i
   >>>         print(acc)
   >>>     return acc

Then we trigger the execution, we can see that no output can be revealed just like expected.

.. code-block:: python

   >>> r = mr.spawn(calculate, 10)
   >>> r.execute()
   Object <op=RemoteFunction, key=67b91ee22e4153107c615797bdb6c189>
   >>> r.fetch()
   45

You can use ``r.fetch_log()`` to retrieve the output.

.. code-block:: python

   >>> print(r.fetch_log())
   0
   1
   3
   6
   10
   15
   21
   28
   36
   45

   >>> print(r.fetch_log())  # call fetch_log again will continue to fetch

   >>> print(r.fetch_log(offsets=0))  # set offsets=0 to retrieve from beginning
   0
   1
   3
   6
   10
   15
   21
   28
   36
   45

Combining :ref:`asynchronous execution <async_execute>` with ``fetch_log``,
it's able to retrieve log continuously during running.

.. code-block::

   >>> import time
   >>> def c_calc():
   >>>     for i in range(10):
   >>>         time.sleep(1)
   >>>         print(i)
   >>>
   >>> r = mr.spawn(c_calc)
   >>>
   >>> def run():
   >>>     f = r.execute(wait=False)
   >>>     while not f.done():
   >>>         time.sleep(0.5)
   >>>         log = str(r.fetch_log()).strip()
   >>>         if log:
   >>>             print(log)
   >>>
   >>> run()
   0
   1
   2
   3
   4
   5
   6
   7
   8
   9
