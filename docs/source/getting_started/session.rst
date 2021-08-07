.. _session:

Session
=======

Sessions can be used for local execution, connecting to a :ref:`local cluster
<local>` or an existing :ref:`Mars cluster <deploy>`.

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

``new_session`` can be used to create new default sessions.

.. code-block:: python

   >>> import mars
   >>> mars.new_session()  # create a new default session
   >>> df = md.DataFrame([[1, 2], [3, 4]])
   >>> df.execute()  # execute on the session just created
      0  1
   0  1  2
   1  3  4
   >>> df.fetch()  # fetch from the session just created
      0  1
   0  1  2
   1  3  4

Sessions can be specified explicitly as an argument for both ``execute`` and ``fetch``.

.. code-block:: python

   >>> import mars
   >>> import mars.tensor as mt
   >>> sess = mars.new_session(default=False)
   >>> t = mt.random.rand(3, 2)
   >>> t.execute(session=sess)
   array([[0.9956293 , 0.06604185],
          [0.25585635, 0.98183162],
          [0.04446616, 0.2417941 ]])
   >>> t.fetch(session=sess)
   array([[0.9956293 , 0.06604185],
          [0.25585635, 0.98183162],
          [0.04446616, 0.2417941 ]])

Call ``.as_default()`` explicitly on a session will set the session as default, ``.execute()``
and ``.fetch()`` will be constraint to the default session.

Each session is isolated. Calling ``.fetch()`` on a Mars object which is executed
in another session will fail.

.. code-block:: python

    >>> import mars
    >>> sess = mars.new_session(default=False)
    >>> df.fetch(session=sess)
    ---------------------------------------------------------------------------
    ValueError                                Traceback (most recent call last)
    <ipython-input-7-f10708ec743f> in <module>
    ----> 1 df.fetch(session=sess)

    ~/Workspace/mars/mars/dataframe/core.py in fetch(self, session, **kw)
        525             return self._fetch(session=session, **kw)
        526         else:
    --> 527             batches = list(self._iter(batch_size=batch_size,
        528                                       session=session, **kw))
        529             return pd.concat(batches) if len(batches) > 1 else batches[0]

    ~/Workspace/mars/mars/dataframe/core.py in _iter(self, batch_size, session, **kw)
        509                 yield batch_data._fetch(session=session, **kw)
        510         else:
    --> 511             yield self._fetch(session=session, **kw)
        512
        513     def iterbatch(self, batch_size=1000, session=None, **kw):

    ~/Workspace/mars/mars/core/entity/executable.py in _fetch(self, session, **kw)
        120         session = _get_session(self, session)
        121         self._check_session(session, 'fetch')
    --> 122         return fetch(self, session=session, **kw)
        123
        124     def fetch(self, session: SessionType = None, **kw):

    ~/Workspace/mars/mars/deploy/oscar/session.py in fetch(tileable, session, *tileables, **kwargs)
       1391
       1392     session = _ensure_sync(session)
    -> 1393     return session.fetch(tileable, *tileables, **kwargs)
       1394
       1395

    ~/Workspace/mars/mars/deploy/oscar/session.py in fetch(self, *tileables, **kwargs)
       1240     def fetch(self, *tileables, **kwargs) -> list:
       1241         coro = _fetch(*tileables, session=self._isolated_session, **kwargs)
    -> 1242         return asyncio.run_coroutine_threadsafe(coro, self._loop).result()
       1243
       1244     @implements(AbstractSyncSession.decref)

    ~/miniconda3/envs/mars3.8/lib/python3.8/concurrent/futures/_base.py in result(self, timeout)
        437                 raise CancelledError()
        438             elif self._state == FINISHED:
    --> 439                 return self.__get_result()
        440             else:
        441                 raise TimeoutError()

    ~/miniconda3/envs/mars3.8/lib/python3.8/concurrent/futures/_base.py in __get_result(self)
        386     def __get_result(self):
        387         if self._exception:
    --> 388             raise self._exception
        389         else:
        390             return self._result

    ~/Workspace/mars/mars/deploy/oscar/session.py in _fetch(tileable, session, *tileables, **kwargs)
       1375         tileable, tileables = tileable[0], tileable[1:]
       1376     session = _get_isolated_session(session)
    -> 1377     data = await session.fetch(tileable, *tileables, **kwargs)
       1378     return data[0] if len(tileables) == 0 else data
       1379

    ~/Workspace/mars/mars/deploy/oscar/session.py in fetch(self, *tileables, **kwargs)
        807             fetch_infos_list = []
        808             for tileable in tileables:
    --> 809                 fetch_tileable, indexes = self._get_to_fetch_tileable(tileable)
        810                 chunk_to_slice = None
        811                 if indexes is not None:

    ~/Workspace/mars/mars/deploy/oscar/session.py in _get_to_fetch_tileable(self, tileable)
        751                 break
        752             else:
    --> 753                 raise ValueError(f'Cannot fetch unexecuted '
        754                                  f'tileable: {tileable}')
        755

    ValueError: Cannot fetch unexecuted tileable: DataFrame(op=DataFrameDataSource)

If ``session`` argument is not passed to ``new_session``, a local session will be
created.

For distributed, the URL of Web UI could be passed to ``new_session`` to connect
to an existing cluster.

.. code-block:: python

   >>> import mars
   >>> mars.new_session('http://<web_ip>:<web_port>')
   >>> df = md.DataFrame([[1, 2], [3, 4]])
   >>> df.execute()  # submit to Mars cluster
      0  1
   0  1  2
   1  3  4
