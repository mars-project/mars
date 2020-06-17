.. _session:

Session
=======

Sessions can be used for local execution, connecting to a :ref:`local cluster
<local_cluster>` or an existing :ref:`Mars cluster <deploy>`.

If a session is not initialized explicitly, Mars will create a session for
local execution by default.

.. code-block::

   >>> import mars.dataframe as md
   >>> df = md.DataFrame([[1, 2], [3, 4]])
   >>> df.execute()  # will create a default session for local execution
      0  1
   0  1  2
   1  3  4
   >>> df.fetch()
      0  1
   0  1  2
   1  3  4

``new_session`` can be used to create new sessions. After created, sessions can
be specified as an argument for both ``execute`` and ``fetch``.

.. code-block:: python

   >>> from mars.session import new_session
   >>> import mars.tensor as mt
   >>> sess = new_session()
   >>> t = mt.random.rand(3, 2)
   >>> t.execute(session=sess)
   array([[0.9956293 , 0.06604185],
          [0.25585635, 0.98183162],
          [0.04446616, 0.2417941 ]])
   >>> t.fetch(session=sess)
   array([[0.9956293 , 0.06604185],
          [0.25585635, 0.98183162],
          [0.04446616, 0.2417941 ]])

Call ``.as_default()`` on a session will set the session as default, ``.execute()``
and ``.fetch()`` will be constraint to the default session.

.. code-block:: python

   >>> from mars.session import new_session
   >>> new_session().as_default()  # set default session
   <mars.session.Session at 0x1a33bbeb50>
   >>> df = md.DataFrame([[1, 2], [3, 4]])
   >>> df.execute()  # execute on the session just created
      0  1
   0  1  2
   1  3  4
   >>> df.fetch()  # fetch from the session just created
      0  1
   0  1  2
   1  3  4

Each session is isolated. Calling ``.fetch()`` on a Mars object which is executed
in another session will fail.

.. code-block:: python

   >>> from mars.session import new_session
   >>> sess = new_session()
   >>> df.fetch(session=sess)
   ---------------------------------------------------------------------------
   ValueError                                Traceback (most recent call last)
   <ipython-input-7-f10708ec743f> in <module>
   ----> 1 df.fetch(session=sess)

   ~/Workspace/mars/mars/core.py in fetch(self, session, **kw)
       377         if session is None:
       378             session = Session.default_or_local()
   --> 379         return session.fetch(self, **kw)
       380
       381     def _attach_session(self, session):

   ~/Workspace/mars/mars/session.py in fetch(self, *tileables, **kw)
       427             ret_list = True
       428
   --> 429         result = self._sess.fetch(*tileables, **kw)
       430
       431         ret = []

   ~/Workspace/mars/mars/session.py in fetch(self, n_parallel, *tileables, **kw)
       114         if n_parallel is None:
       115             kw['n_parallel'] = cpu_count()
   --> 116         return self._executor.fetch_tileables(tileables, **kw)
       117
       118     def create_mutable_tensor(self, name, shape, dtype, fill_value=None, *args, **kwargs):

   ~/Workspace/mars/mars/utils.py in _wrapped(*args, **kwargs)
       383                 _kernel_mode.eager = False
       384             _kernel_mode.eager_count = enter_eager_count + 1
   --> 385             return func(*args, **kwargs)
       386         finally:
       387             _kernel_mode.eager_count -= 1

   ~/Workspace/mars/mars/executor.py in fetch_tileables(self, tileables, **kw)
       919                 # check if the tileable is executed before
       920                 raise ValueError(
   --> 921                     'Tileable object {} to fetch must be executed first'.format(tileable))
       922
       923         # if chunk executed, fetch chunk mechanism will be triggered in execute_tileables

   ValueError: Tileable object    0  1
   0  1  2
   1  3  4 to fetch must be executed first

If ``session`` argument is not passed to ``new_session``, a local session will be
created. The local session will leverage :ref:`threaded scheduler <threaded>`
for execution.

For distributed, the URL of Web UI could be passed to ``new_session`` to connect
to an existing cluster.

.. code-block:: python

   >>> from mars.session import new_session
   >>> new_session('http://<web_ip>:<web_port>').as_default()
   >>> df = md.DataFrame([[1, 2], [3, 4]])
   >>> df.execute()  # submit to Mars cluster
      0  1
   0  1  2
   1  3  4
