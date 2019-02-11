#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2017 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The files belonging to gipc are released under the following MIT license:
#
# Copyright 2012-2017 Jan-Philip Gehrcke (http://gehrcke.de)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# -*- coding: utf-8 -*-
# Copyright 2012-2017 Jan-Philip Gehrcke. See LICENSE file for details.


"""
gipc: child processes and inter-process communication (IPC) for gevent.

gipc (pronunciation “gipsy”)

* prevents negative side-effects of multiprocessing-based child process creation
  in the context of gevent.

* provides the multiprocessing.Process API in a gevent-cooperative fashion.

* comes up with a pipe-based transport layer for efficient gevent-cooperative
  inter-process communication.
"""


import os
import io
import sys
import struct
import signal
import codecs
import logging
import multiprocessing
import multiprocessing.process
from itertools import chain

try:
    import cPickle as pickle
except ImportError:
    import pickle

try:
    from pytest_cov.embed import cleanup_on_sigterm
except ImportError:  # pragma: no cover
    cleanup_on_sigterm = lambda: None

cdef:
    bint WINDOWS, PY2

WINDOWS = sys.platform == "win32"
PY2 = sys.version_info[0] == 2

if WINDOWS:
    import multiprocessing.reduction
    import msvcrt

import gevent
import gevent.os
import gevent.lock
import gevent.event


# Decide which method to use for transferring WinAPI pipe handles to children.
if WINDOWS:
    WINAPI_HANDLE_TRANSFER_STEAL = hasattr(
        multiprocessing.reduction, "steal_handle")


# Logging for debugging purposes. Usage of logging in this simple form in the
# context of multiple processes might yield mixed messages in the output.
log = logging.getLogger('gipc')


class GIPCError(Exception):
    """Is raised upon general errors. All other exception types derive from
    this one.
    """
    pass


class GIPCClosed(GIPCError):
    """Is raised upon operation on closed handle."""
    pass


class GIPCLocked(GIPCError):
    """Is raised upon attempt to close a handle which is currently locked for
    I/O.
    """
    pass


def _newpipe(encoder, decoder):
    """Create new pipe via `os.pipe()` and return `(_GIPCReader, _GIPCWriter)`
    tuple.

    os.pipe() implementation on Windows (https://goo.gl/CiIWvo):
       - CreatePipe(&read, &write, NULL, 0)
       - anonymous pipe, system handles buffer size
       - anonymous pipes are implemented using named pipes with unique names
       - asynchronous (overlapped) read and write operations not supported
    os.pipe() implementation on Unix (http://linux.die.net/man/7/pipe):
       - based on pipe()
       - common Linux: pipe buffer is 4096 bytes, pipe capacity is 65536 bytes
    """
    r, w = os.pipe()
    return (_GIPCReader(r, decoder), _GIPCWriter(w, encoder))


# Define default encoder and decoder functions for pipe data serialization.
def _default_encoder(o):
    return pickle.dumps(o, pickle.HIGHEST_PROTOCOL)


_default_decoder = pickle.loads


def pipe(duplex=False, encoder='default', decoder='default'):
    """Create a pipe-based message transport channel and return two
    corresponding handles for reading and writing data.

    Allows for gevent-cooperative transmission of data between greenlets within
    one process or across processes (created via :func:`start_process`). The
    default behavior allows for transmission of any picklable Python object.

    The transport layer is based on ``os.pipe()`` (i.e.
    `CreatePipe() <https://goo.gl/CiIWvo>`_ on Windows and
    `pipe() <http://goo.gl/it6rFW>`_ on POSIX-compliant systems).

    :arg duplex:
        - If ``False`` (default), create a unidirectional pipe-based message
          transport channel and return the corresponding handle pair, a 2-tuple
          with the first element of type  ``_GIPCReader`` and the second
          element of type ``_GIPCWriter``.
        - If ``True``, create a bidirectional message transport channel (using
          two pipes internally) and return the corresponding 2-tuple with both
          elements being of type ``_GIPCDuplexHandle``.

    :arg encoder:
        Defines the entity used for object serialization before writing object
        ``o`` to the pipe via ``put(o)``. Must be either a callable returning
        a byte string, ``None``, or ``'default'``. ``'default'`` translates to
        ``pickle.dumps`` (in this mode, any pickleable Python object can be
        provided to ``put()`` and transmitted through the pipe). When setting
        this to ``None``, no automatic object serialization is performed. In
        that case only byte strings are allowed to be provided to ``put()``,
        and a ``TypeError`` is thrown otherwise. A ``TypeError`` will also be
        thrown if the encoder callable does not return a byte string.

    :arg decoder:
        Defines the entity used for data deserialization after reading raw
        binary data from the pipe. Must be a callable retrieving a byte string
        as first and only argument, ``None`` or ``'default'``. ``'default'``
        translates to ``pickle.loads``. When setting this to ``None``, no data
        decoding is performed, and a raw byte string is returned.

    :returns:
        - ``duplex=False``: ``(reader, writer)`` 2-tuple. The first element is
          of type :class:`gipc._GIPCReader`, the second of type
          :class:`gipc._GIPCWriter`. Both inherit from
          :class:`gipc._GIPCHandle`.
        - ``duplex=True``: ``(handle, handle)`` 2-tuple. Both elements are of
          type :class:`gipc._GIPCDuplexHandle`.


    :class:`gipc._GIPCHandle` and :class:`gipc._GIPCDuplexHandle`  instances
    are recommended to be used with a context manager as indicated in the
    following examples::

        with pipe() as (r, w):
            do_something(r, w)

    ::

        reader, writer = pipe()
        with reader:
            do_something(reader)
            with writer as w:
                do_something(w)

    ::

        with pipe(duplex=True) as (h1, h2):
            h1.put(1)
            assert h2.get() == 1
            h2.put(2)
            assert h1.get() == 2

    An example for using the encoder/decoder arguments for implementing JSON
    (de)serialization::

        import json
        enc = lambda o: json.dumps(o).encode("ascii")
        dec = lambda b: json.loads(b.decode("ascii"))
        with pipe(encoder=enc, decoder=dec) as (r, w):
            ...

    Note that JSON representation is text whereas the encoder/decoder callables
    must return/accept byte strings, as ensured here by ASCII en/decoding. Also
    note that in practice JSON serializaton has normally no advantage over
    pickling, so this is just an educational example.
    """
    # Internally, `encoder` and `decoder` must be callable. Translate
    # special values `None` and `'default'` to callables here.
    if encoder is None:
        encoder = None
    elif encoder == 'default':
        encoder = _default_encoder
    elif not callable(encoder):
        raise GIPCError("pipe 'encoder' argument must be callable.")
    if decoder is None:
        decoder = None
    elif decoder == 'default':
        decoder = _default_decoder
    elif not callable(decoder):
        raise GIPCError("pipe 'decoder' argument must be callable.")
    pair1 = _newpipe(encoder, decoder)
    if not duplex:
        return _PairContext(pair1)
    pair2 = _newpipe(encoder, decoder)
    return _PairContext((
        _GIPCDuplexHandle((pair1[0], pair2[1])),
        _GIPCDuplexHandle((pair2[0], pair1[1]))))


def start_process(target, args=(), kwargs={}, daemon=None, name=None):
    """Start child process and execute function ``target(*args, **kwargs)``.
    Any existing instance of :class:`gipc._GIPCHandle` or
    :class:`gipc._GIPCDuplexHandle` can be passed to the child process via
    ``args`` and/or ``kwargs``. If any such instance is passed to the child,
    gipc automatically closes the corresponding file descriptor(s) in the
    parent.

    .. note::

        Compared to the canonical ``multiprocessing.Process()`` constructor,
        this function

        - returns a :class:`gipc._GProcess` instance which is compatible with
          the ``multiprocessing.Process`` API.
        - just as well takes the ``target``, ``arg=()``, and ``kwargs={}``
          arguments.
        - introduces the ``daemon=None`` argument.
        - does not accept the ``group`` argument (being an artifact from
          ``multiprocessing``'s compatibility with ``threading``).
        - starts the process, i.e. a subsequent call to the ``start()`` method
          of the returned object is not required.

    :arg target:
        Function to be called in the child process. Signature:
        ``target(*args, **kwargs)``.

    :arg args:
        Tuple defining the positional arguments provided to ``target``.

    :arg kwargs:
        Dictionary defining the keyword arguments provided to ``target``.

    :arg name:
        Forwarded to ``multiprocessing.Process.name``.

    :arg daemon:
        Forwarded to ``multiprocessing.Process.daemon``.

    :returns:
        :class:`gipc._GProcess` instance (inherits from
        ``multiprocessing.Process`` and re-implements some of its methods in a
        gevent-cooperative fashion).

    :func:`start_process` triggers most of the magic in ``gipc``. Process
    creation is based on ``multiprocessing.Process()``, i.e. ``fork()`` on
    POSIX-compliant systems and ``CreateProcess()`` on Windows.

    .. warning::

        Please note that in order to provide reliable signal handling in the
        context of libev, the default disposition (action) is restored for all
        signals in the child before executing the user-given ``target``
        function. You can (re)install any signal handler within ``target``. The
        notable exception is the SIGPIPE signal, whose handler is *not* reset
        to its default handler in child processes created by ``gipc``. That is,
        the SIGPIPE action in children is inherited from the parent. In
        CPython, the default action for SIGPIPE is SIG_IGN, i.e. the signal is
        ignored.
    """
    if not isinstance(args, tuple):
        raise TypeError('`args` must be a tuple.')
    if not isinstance(kwargs, dict):
        raise TypeError('`kwargs` must be a dictionary.')
    log.debug("Invoke target `%s` in child process.", target)
    childhandles = list(_filter_handles(chain(args, kwargs.values())))
    if WINDOWS:
        for h in childhandles:
            h._winapi_childhandle_prepare_transfer()
    p = _GProcess(
        target=_child,
        name=name,
        kwargs={"target": target,
                "args": args,
                "kwargs": kwargs})
    if daemon is not None:
        p.daemon = daemon
    p.start()
    p.start = lambda *a, **b: sys.stderr.write(
        "gipc WARNING: Redundant call to %s.start()\n" % p)
    # Close dispensable file handles in parent.
    for h in childhandles:
        log.debug("Invalidate %s in parent.", h)
        if WINDOWS:
            # Prepare the subsequent `h.close()` call. This mainly takes care of
            # preparing `h._fd` and reverts actions taken during
            # `_winapi_childhandle_prepare_transfer()`.
            h._winapi_childhandle_after_createprocess_parent()
        h.close()
    return p


def _child(target, args, kwargs):
    """Wrapper function that runs in child process. Resets gevent/libev state
    and executes user-given function.

    After fork on POSIX-compliant systems, gevent's state is inherited by the
    child which may lead to undesired behavior, such as greenlets running in
    both, the parent and the child. Therefore, if not on Windows, gevent's and
    libev's state is reset before running the user-given function.
    """
    log.debug("_child start. target: `%s`", target)
    childhandles = list(_filter_handles(chain(args, kwargs.values())))
    if not WINDOWS:
        # Restore default signal handlers (SIG_DFL). Orphaned libev signal
        # watchers may not become properly deactivated otherwise. Note: here, we
        # could even reset sigprocmask (Python 2.x does not have API for it, but
        # it could be done via ctypes).
        _reset_signal_handlers()
        # `gevent.reinit` calls `libev.ev_loop_fork()`, which reinitialises
        # the kernel state for backends that have one. Must be called in the
        # child before using further libev API.
        gevent.reinit()
        log.debug("Delete current hub's threadpool.")
        hub = gevent.get_hub()
        # Delete threadpool before hub destruction, otherwise `hub.destroy()`
        # might block forever upon `ThreadPool.kill()` as of gevent 1.0rc2.
        del hub.threadpool
        hub._threadpool = None
        # Destroy default event loop via `libev.ev_loop_destroy()` and delete
        # hub. This orphans all registered events and greenlets that have been
        # duplicated from the parent via fork().
        log.debug("Destroy hub and default loop.")
        hub.destroy(destroy_loop=True)
        # Create a new hub and a new default event loop via
        # `libev.gevent_ev_default_loop`.
        h = gevent.get_hub(default=True)
        log.debug("Created new hub and default event loop.")
        assert h.loop.default, 'Could not create libev default event loop.'
        # On Unix, file descriptors are inherited by default. Also, the global
        # `_all_handles` is inherited from the parent. Close dispensable gipc-
        # related file descriptors in child.
        for h in _all_handles[:]:
            if h not in childhandles:
                log.debug("Invalidate %s in child.", h)
                h._set_legit_process()
                # At duplication time the handle might have been locked. Unlock.
                h._lock.counter = 1
                h.close()
    else:
        # On Windows, the state of module globals is not transferred to
        # children. Set `_all_handles`.
        _set_all_handles(childhandles)
    # `_all_handles` now must contain only those handles that have been
    # transferred to the child on purpose.
    for h in _all_handles:
        assert h in childhandles
    # Register transferred handles for current process.
    for h in childhandles:
        h._set_legit_process()
        if WINDOWS:
            h._winapi_childhandle_after_createprocess_child()
        log.debug("Handle `%s` is now valid in child.", h)
    # Invoke user-given function.
    target(*args, **kwargs)
    # Close file descriptors before exiting process. Usually needless (OS
    # should take care of this), but being expressive about this is clean.
    for h in childhandles:
        try:
            # The user might already have closed it.
            h.close()
        except GIPCClosed:
            pass


class _GProcess(multiprocessing.Process):
    """
    Compatible with the ``multiprocessing.Process`` API.

    For cooperativeness with gevent and compatibility with libev, it currently
    re-implements ``start()``, ``is_alive()``, ``exitcode`` on Unix and
    ``join()`` on Windows as well as on Unix.

    .. note::

        On Unix, child monitoring is implemented via libev child watchers.
        To that end, libev installs its own SIGCHLD signal handler.
        Any call to ``os.waitpid()`` would compete with that handler, so it
        is not recommended to call it in the context of this module.
        ``gipc`` prevents ``multiprocessing`` from calling ``os.waitpid()`` by
        monkey-patching multiprocessing's ``Popen.poll`` to be no-op and to
        always return ``None``. Calling ``gipc._GProcess.join()`` is not
        required for cleaning up after zombies (libev does). It just waits
        for the process to terminate.
    """
    # Remarks regarding child process monitoring on Unix:
    #
    # For each `_GProcess`, a libev child watcher is started in the modified
    # `start()` method below. The modified `join()` method is adjusted to this
    # libev child watcher-based child monitoring.
    # `multiprocessing.Process.join()` is entirely surpassed, but resembled.

    # After initialization of the first libev child watcher, i.e. after
    # initialization of the first _GProcess instance, libev handles SIGCHLD
    # signals. Dead children become reaped by the libev event loop. The
    # children's status code is evaluated by libev. In conclusion, installation
    # of the libev SIGCHLD handler renders multiprocessing's child monitoring
    # useless and even hindering.

    # Any call to os.waitpid can make libev miss certain SIGCHLD
    # events. According to
    # http://pubs.opengroup.org/onlinepubs/009695399/functions/waitpid.html
    #
    # "If [...] the implementation queues the SIGCHLD signal, then if wait()
    #  or waitpid() returns because the status of a child process is available,
    #  any pending SIGCHLD signal associated with the process ID of the child
    #  process shall be discarded."

    # On Windows, cooperative `join()` is realized via polling (non-blocking
    # calls to `Process.is_alive()`) and the original `join()` method.
    if not WINDOWS:
        # multiprocessing.process.Process.start() and other methods may
        # call multiprocessing.process._cleanup(). This and other mp methods
        # may call multiprocessing's Popen.poll() which itself invokes
        # os.waitpid(). In extreme cases (high-frequent child process
        # creation, short-living child processes), this competes with libev's
        # SIGCHLD handler and may win, resulting in libev not being able to
        # retrieve all SIGCHLD signals corresponding to started children. This
        # could make certain _GProcess.join() calls block forever.
        # -> Prevent multiprocessing's Popen.poll() from calling
        # os.waitpid(). Let libev do the job.
        try:
            from multiprocessing.forking import Popen as mp_Popen
        except ImportError:  # pragma: no cover
            # multiprocessing's internal structure has changed from 3.3 to 3.4.
            from multiprocessing.popen_fork import Popen as mp_Popen
        # Monkey-patch and forget about the name.
        mp_Popen.poll = lambda *a, **b: None
        del mp_Popen

        def start(self):
            # Start grabbing SIGCHLD within libev event loop.
            gevent.get_hub().loop.install_sigchld()
            # Run new process (based on `fork()` on POSIX-compliant systems).
            super(_GProcess, self).start()
            # The occurrence of SIGCHLD is recorded asynchronously in libev.
            # This guarantees proper behavior even if the child watcher is
            # started after the child exits. Start child watcher now.
            self._sigchld_watcher = gevent.get_hub().loop.child(self.pid)
            self._returnevent = gevent.event.Event()
            self._sigchld_watcher.start(
                self._on_sigchld, self._sigchld_watcher)
            log.debug("SIGCHLD watcher for %s started.", self.pid)

        def _on_sigchld(self, watcher):
            """Callback of libev child watcher. Called when libev event loop
            catches corresponding SIGCHLD signal.
            """
            watcher.stop()
            # Status evaluation copied from `multiprocessing.forking` in Py2.7.
            if os.WIFSIGNALED(watcher.rstatus):
                self._popen.returncode = -os.WTERMSIG(watcher.rstatus)
            else:
                assert os.WIFEXITED(watcher.rstatus)
                self._popen.returncode = os.WEXITSTATUS(watcher.rstatus)
            self._returnevent.set()
            log.debug("SIGCHLD watcher callback for %s invoked. Exitcode "
                      "stored: %s", self.pid, self._popen.returncode)

        def is_alive(self):
            assert self._popen is not None, "Process not yet started."
            if self._popen.returncode is None:
                return True
            return False

        @property
        def exitcode(self):
            if self._popen is None:
                return None
            return self._popen.returncode

        def __repr__(self):
            """Based on original __repr__ from CPython 3.4's mp package.

            Reasons for re-implementing:

            * The original code would invoke os.waitpid() through
              _popen.poll(). This is forbidden in the context of gipc.
              This method instead reads the exitcode property which is set
              asynchronously by a libev child watcher callback.

            * The original code distinguishes 'initial' state from 'started'
              state. This is not necessary, as gipc starts processes right
              away.

            * This method removes the `if self is _current_process` check
              without changing output behavior (that's still 'started' status).
            """
            exitcodedict = multiprocessing.process._exitcode_to_name
            status = 'started'
            if self._parent_pid != os.getpid():
                status = 'unknown'
            elif self.exitcode is not None:
                status = self.exitcode
            if status == 0:
                status = 'stopped'
            elif isinstance(status, int):
                status = 'stopped[%s]' % exitcodedict.get(status, status)
            return '<%s(%s, %s%s)>' % (
                type(self).__name__,
                self._name,
                status,
                self.daemon and ' daemon' or ''
                )

    def join(self, timeout=None):
        """
        Wait cooperatively until child process terminates or timeout occurs.

        :arg timeout: ``None`` (default) or a a time in seconds. The method
            simply returns upon timeout expiration. The state of the process
            has to be identified via ``is_alive()``.
        """
        assert self._parent_pid == os.getpid(), "I'm not parent of this child."
        assert self._popen is not None, 'Can only join a started process.'
        if not WINDOWS:
            # Resemble multiprocessing's join() method while replacing
            # `self._popen.wait(timeout)` with
            # `self._returnevent.wait(timeout)`
            self._returnevent.wait(timeout)
            if self._popen.returncode is not None:
                if hasattr(multiprocessing.process, '_children'):
                    # This is for Python 3.4.
                    kids = multiprocessing.process._children
                else:
                    # For Python 2.6, 2.7, 3.3.
                    kids = multiprocessing.process._current_process._children
                kids.discard(self)
            return
        with gevent.Timeout(timeout, False):
            while self.is_alive():
                # This frequency seems reasonable, but that's not 100 % certain.
                gevent.sleep(0.01)
        # Clean up after child as designed by Process class (non-blocking).
        super(_GProcess, self).join(timeout=0)


cdef class _GIPCHandle:
    """
    The ``_GIPCHandle`` class implements common features of read and write
    handles. ``_GIPCHandle`` instances are created via :func:`pipe`.

    .. todo::

        Implement destructor?
        http://eli.thegreenplace.net/2009/06/12/
        safely-using-destructors-in-python/
    """
    cdef public str _id
    cdef public int _legit_pid
    cdef public object _lock
    cdef public bint _closed
    cdef public object _fd

    def __init__(self):
        global _all_handles
        # Generate label of text/unicode type from three random bytes.
        if PY2:
            self._id = codecs.encode(os.urandom(3), "hex_codec")
        else:
            self._id = codecs.encode(os.urandom(3), "hex_codec").decode("ascii")
        self._legit_pid = os.getpid()
        self._make_nonblocking()
        self._lock = gevent.lock.Semaphore(value=1)
        self._closed = False
        _all_handles.append(self)

    cdef void _make_nonblocking(self):
        if hasattr(gevent.os, 'make_nonblocking'):
            # On POSIX-compliant systems, the file descriptor flags are
            # inherited after forking, i.e. it is sufficient to make fd
            # nonblocking only once.
            gevent.os.make_nonblocking(self._fd)

    def close(self):
        """Close underlying file descriptor and de-register handle from further
        usage. Is called on context exit.

        Raises:
            - :exc:`GIPCError`
            - :exc:`GIPCClosed`
            - :exc:`GIPCLocked`
        """
        global _all_handles
        self._validate()
        if not self._lock.acquire(blocking=False):
            raise GIPCLocked(
                "Can't close handle %s: locked for I/O operation." % self)
        log.debug("Invalidating %s ...", self)
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if self in _all_handles:
            # Remove the handle from the global list of valid handles.
            _all_handles.remove(self)
        self._closed = True
        self._lock.release()

    def _set_legit_process(self):
        log.debug("Legitimate %s for current process.", self)
        self._legit_pid = os.getpid()

    cdef _validate(self):
        """Raise exception if this handle is closed or not registered to be
        used in the current process.

        Intended to be called before every operation on `self._fd`.
        Reveals wrong usage of this module in the context of multiple
        processes. Might prevent tedious debugging sessions. Has little
        performance impact.
        """
        if self._closed:
            raise GIPCClosed(
                "GIPCHandle has been closed before.")
        if os.getpid() != self._legit_pid:
            raise GIPCError(
                "GIPCHandle %s not registered for current process %s." % (
                    self, os.getpid()))

    def _winapi_childhandle_prepare_transfer(self):
        """Prepare file descriptor for transfer to child process on Windows.

        What follows now is an overview for the process of transferring a
        Windows pipe handle to a child process, for different Python versions
        (explanation / background can be found below):

        Python versions < 3.4:
            1)   In the parent, get WinAPI handle from C file descriptor via
                 msvcrt.get_osfhandle().
            2)   WinAPI call DuplicateHandle(... ,bInheritHandle=True) in
                 parent. Close old handle, let inheritable duplicate live on.
            2.5) multiprocessing internals invoke WinAPI call CreateProcess(...,
                 InheritHandles=True).
            3)   Close the duplicate in the parent.
            4)   Use msvcrt.open_osfhandle() in child for converting the Windows
                 pipe handle to a C file descriptor, and for setting the
                 (read/write)-only access flag. This file descriptor will be
                 used by user code.

        Python versions >= 3.4:
            1)   Same as above.
            2)   Store handle and process ID. Both are integers that will be
                 pickled to and used by the child.
            2.5) multiprocessing internals invoke WinAPI call
                 CreateProcess(..., InheritHandles=False).
            3)   Steal the Windows pipe handle from the parent process: in the
                 child use the parent's process ID for getting a WinAPI handle
                 to the parent process via WinAPI call OpenProcess(). Use option
                 PROCESS_DUP_HANDLE as desired access right. Invoke
                 DuplicateHandle() in the child and use the handle to the parent
                 process as source process handle. Use the
                 DUPLICATE_CLOSE_SOURCE and the DUPLICATE_SAME_ACCESS flags. The
                 result is a Windows pipe handle, "stolen" from the parent.
            4)   Same as above.

        Background:

        By default, file descriptors are not inherited by child processes on
        Windows. However, they can be made inheritable via calling the system
        function `DuplicateHandle` while setting `bInheritHandle` to True.
        From MSDN:
            bInheritHandle:
                A variable that indicates whether the handle is inheritable.
                If TRUE, the duplicate handle can be inherited by new processes
                created by the target process. If FALSE, the new handle cannot
                be inherited.
        The internals of Python's `subprocess` and `multiprocessing` make use of
        this. There is, however, no officially exposed Python API. Nevertheless,
        the function `multiprocessing.forking.duplicate` (in Python versions
        smaller than 3.4) and `multiprocessing.reduction.duplicate` (>= 3.4)
        seems to be safely usable. In all versions, `duplicate` is part of
        `multiprocessing.reduction`. As of 2015-07-20, the reduction module is
        part of multiprocessing in all Python versions from 2.6 to 3.5.

        The just outlined approach (DuplicateHandle() in parent, automatically
        inherit it in child) only works for Python versions smaller than 3.4:
        from Python 3.4 on, the child process is created with CreateProcess()'s
        `InheritHandles` attribute set to False (this was explicitly set to True
        in older Python versions). A different method needs to be used, referred
        to as "stealing" the handle: DuplicateHandle can be called in the child
        for retrieving the handle from the parent while using the
        _winapi.DUPLICATE_CLOSE_SOURCE flag, which automatically closes the
        handle in the parent. This method is used by
        `multiprocessing.popen_spawn_win32` and implemented in
        `multiprocessing.reduction.steal_handle`.

        Refs:
        https://msdn.microsoft.com/en-us/library/windows/desktop/ms684880.aspx
        https://msdn.microsoft.com/en-us/library/windows/desktop/ms684320.aspx
        https://msdn.microsoft.com/en-us/library/ks2530z6.aspx
        https://msdn.microsoft.com/en-us/library/bdts1c9x.aspx
        """
        if WINAPI_HANDLE_TRANSFER_STEAL:
            self._parent_winapihandle = msvcrt.get_osfhandle(self._fd)
            self._parent_pid = os.getpid()
            return
        # Get Windows file handle from C file descriptor.
        winapihandle = msvcrt.get_osfhandle(self._fd)
        # Duplicate file handle, rendering the duplicate inheritable by
        # processes created by the current process.
        self._inheritable_winapihandle = multiprocessing.reduction.duplicate(
            handle=winapihandle, inheritable=True)
        # Close "old" (in-inheritable) file descriptor.
        os.close(self._fd)
        # Mark file descriptor as "already closed".
        self._fd = None

    def _winapi_childhandle_after_createprocess_parent(self):
        """Called on Windows in the parent process after the CreateProcess()
        system call. This method is intended to revert the actions performed
        within `_winapi_childhandle_prepare_transfer()`. In particular, this
        method is intended to prepare a subsequent call to the handle's
        `close()` method.
        """
        if WINAPI_HANDLE_TRANSFER_STEAL:
            del self._parent_winapihandle
            del self._parent_pid
            # Setting `_fd` to None prevents the subsequent `close()` method
            # invocation (triggered in `start_process()` after child creation)
            # from actually calling `os.close()` on the file descriptor. This
            # must be prevented because at this point the handle either already
            # is or will be "stolen" by the child via a direct WinAPI call using
            # the DUPLICATE_CLOSE_SOURCE option (and therefore become
            # auto-closed, here, in the parent). The relative timing is not
            # predictable. If the child process steals first, os.close() here
            # would result in `OSError: [Errno 9] Bad file descriptor`. If
            # os.close() is called on the handle in the parent before the child
            # can steal the handle, a `OSError: [WinError 6] The handle is
            # invalid` will be thrown in the child upon the stealing attempt.
            self._fd = None
            return
        # Get C file descriptor from Windows file handle.
        self._fd = msvcrt.open_osfhandle(
            self._inheritable_winapihandle, self._fd_flag)
        del self._inheritable_winapihandle

    def _winapi_childhandle_after_createprocess_child(self):
        """Called on Windows in the child process after the CreateProcess()
        system call. This is required for making the handle usable in the child.
        """
        if WINAPI_HANDLE_TRANSFER_STEAL:
            # In this case the handle has not been inherited by the child
            # process during CreateProcess(). Steal it from the parent.
            new_winapihandle = multiprocessing.reduction.steal_handle(
                self._parent_pid, self._parent_winapihandle)
            del self._parent_winapihandle
            del self._parent_pid
            # Restore C file descriptor with (read/write)only flag.
            self._fd = msvcrt.open_osfhandle(new_winapihandle, self._fd_flag)
            return
        # In this case the handle has been inherited by the child process during
        # the CreateProcess() system call. Get C file descriptor from Windows
        # file handle.
        self._fd = msvcrt.open_osfhandle(
            self._inheritable_winapihandle, self._fd_flag)
        del self._inheritable_winapihandle

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except GIPCClosed:
            # Tolerate handles that have been closed within context.
            pass
        except GIPCLocked:
            # Locked for I/O outside of context, which is not fine.
            raise GIPCLocked((
                "Context manager can't close handle %s. It's locked for I/O "
                "operation out of context." % self))

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        fd = self._fd
        if hasattr(self, "_ihfd"):
            fd = "WIN_%s" % self._ihfd
        return "<%s_%s fd: %s>" % (self.__class__.__name__, self._id, fd)


cdef class _GIPCReader(_GIPCHandle):
    """
    A ``_GIPCReader`` instance manages the read end of a pipe. It is created
    via :func:`pipe`.
    """

    cdef int _fd_flag
    cdef object _decoder

    def __init__(self, int pipe_read_fd, object decoder):
        self._fd = pipe_read_fd
        self._fd_flag = os.O_RDONLY
        _GIPCHandle.__init__(self)
        self._decoder = decoder

    cdef object _recv_in_buffer(self, int n, object buf=None):
        """Cooperatively read `n` bytes from file descriptor to buffer."""

        cdef object readbuf
        cdef int remaining
        cdef int received
        cdef bytes chunk

        # In rudimentary tests I have observed frequent creation of a new buffer
        # to be faster than re-using an existing  buffer via seek(0)/truncate().
        readbuf = buf or io.BytesIO()
        remaining = n
        while remaining > 0:
            # Attempt to read at most 65536 bytes from pipe, which is the
            # pipe capacity on common Linux systems. Although unexpected,
            # requesting larger amounts leads to a slow-down of the system
            # call. This has been measured for Linux 2.6.32 and 3.2.0. At
            # the same time this works around a bug in Mac OS X' read()
            # syscall. These findings are documented in
            # https://bitbucket.org/jgehrcke/gipc/issue/13.
            if remaining > 65536:
                chunk = _read_nonblocking(self._fd, 65536)
            else:
                chunk = _read_nonblocking(self._fd, remaining)
            received = len(chunk)
            if received == 0:
                if remaining == n:
                    raise EOFError(
                        "Most likely, the other pipe end is closed.")
                else:
                    raise IOError("Message interrupted by EOF.")
            readbuf.write(chunk)
            remaining -= received
        return readbuf

    cpdef bytes get(self, timeout=None):
        """Receive, decode and return data from the pipe. Block
        gevent-cooperatively until data is available or timeout expires. The
        default decoder is ``pickle.loads``.

        :arg timeout: ``None`` (default) or a ``gevent.Timeout``
            instance. The timeout must be started to take effect and is
            canceled when the first byte of a new message arrives (i.e.
            providing a timeout does not guarantee that the method completes
            within the timeout interval).

        :returns: a Python object.

        Raises:
            - :exc:`gevent.Timeout` (if provided)
            - :exc:`GIPCError`
            - :exc:`GIPCClosed`
            - :exc:`pickle.UnpicklingError`

        Recommended usage for silent timeout control::

            with gevent.Timeout(TIME_SECONDS, False) as t:
                reader.get(timeout=t)

        .. warning::

            The timeout control is currently not available on Windows,
            because Windows can't apply select() to pipe handles.
            An ``OSError`` is expected to be raised in case you set a
            timeout.
        """
        cdef object buf
        cdef int size
        cdef int msize
        cdef object h
        cdef bytes bindata

        self._validate()
        buf = io.BytesIO()
        with self._lock:
            size, = struct.unpack("!i", self._recv_in_buffer(4).getvalue())
            for _ in range(size):
                if timeout:
                    # Wait for ready-to-read event.
                    h = gevent.get_hub()
                    h.wait(h.loop.io(self._fd, 1))
                    timeout.cancel()
                msize, = struct.unpack("!i", self._recv_in_buffer(4).getvalue())
                self._recv_in_buffer(msize, buf=buf)
        bindata = buf.getvalue()
        return self._decoder(bindata) if self._decoder is not None else bindata


cdef class _GIPCWriter(_GIPCHandle):
    """
    A ``_GIPCWriter`` instance manages the write end of a pipe. It is created
    via :func:`pipe`.
    """
    cdef int _fd_flag
    cdef object _encoder

    def __init__(self, int pipe_write_fd, object encoder):
        self._fd = pipe_write_fd
        self._fd_flag = os.O_WRONLY
        _GIPCHandle.__init__(self)
        self._encoder = encoder

    cdef _write(self, bindata):
        """Write `bindata` to pipe in a gevent-cooperative manner.

        POSIX-compliant system notes (http://linux.die.net/man/7/pipe:):
            - Since Linux 2.6.11, the pipe capacity is 65536 bytes
            - Relevant for large messages (O_NONBLOCK enabled,
              n > PIPE_BUF (4096 Byte, usually)):
                "If the pipe is full, then write(2) fails, with errno set
                to EAGAIN. Otherwise, from 1 to n bytes may be written (i.e.,
                a "partial write" may occur; the caller should check the
                return value from write(2) to see how many bytes were
                actually written), and these bytes may be interleaved with
                writes by other processes."

            EAGAIN is handled within _write_nonblocking; partial writes here.
        """
        cdef int bytes_written

        bindata = memoryview(bindata)
        while True:
            # Causes OSError when read end is closed (broken pipe).
            bytes_written = _write_nonblocking(self._fd, bindata)
            if bytes_written == len(bindata):
                break
            bindata = bindata[bytes_written:]

    def put(self, *oo):
        """Encode objects ``oo`` and write them to the pipe.
        Block gevent-cooperatively until all data is written. The default
        encoder is ``pickle.dumps``.

        :arg oo: Python objects that is encodable with the encoder of choice.

        Raises:
            - :exc:`GIPCError`
            - :exc:`GIPCClosed`
            - :exc:`pickle.PicklingError`

        """
        self._validate()
        with self._lock:
            self._write(struct.pack("!i", len(oo)))
            for o in oo:
                bindata = self._encoder(o) if self._encoder is not None else o
                self._write(struct.pack("!i", len(bindata)))
                self._write(bindata)


class _PairContext(tuple):
    """
    Generic context manager for a 2-tuple containing two entities supporting
    context enter and exit themselves. Returns 2-tuple upon entering the
    context, attempts to exit both tuple elements upon context exit.
    """
    def __enter__(self):
        for e in self:
            e.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Call `__exit__()` for both, e1 and e2 entities, in any case, as
        expected from a context manager. Exit e2 first, as it is used as
        writer in case of `_PairContext((reader1, writer1))` and `os.close()`
        on reader might block on Windows otherwise. If an exception occurs
        during e2 exit, store it, exit e1 and re-raise it afterwards. If an
        exception is raised during both, e1 and e2 exit, only raise the e1
        exit exception.
        """
        e2_exit_exception = None

        try:
            self[1].__exit__(exc_type, exc_value, traceback)
        except:  # noqa: E722
            e2_exit_exception = sys.exc_info()

        self[0].__exit__(exc_type, exc_value, traceback)

        if e2_exit_exception:
            _reraise(*e2_exit_exception)


class _GIPCDuplexHandle(_PairContext):
    """
    A ``_GIPCDuplexHandle`` instance manages one end of a bidirectional
    pipe-based message transport created via :func:`pipe()` with
    ``duplex=True``. It provides ``put()``, ``get()``, and ``close()``
    methods which are forwarded to the corresponding methods of
    :class:`gipc._GIPCWriter` and :class:`gipc._GIPCReader`.
    """
    def __init__(self, rwpair):
        self._reader, self._writer = rwpair
        self.put = self._writer.put
        self.get = self._reader.get

    def close(self):
        """Close associated `_GIPCHandle` instances. Tolerate if one of both
        has already been closed before. Throw GIPCClosed if both have been
        closed before.
        """
        if self._writer._closed and self._reader._closed:
            raise GIPCClosed("Reader & writer in %s already closed." % (self,))
        # Close writer first. Otherwise, reader close would block on Win.
        if not self._writer._closed:
            self._writer.close()
        if not self._reader._closed:
            self._reader.close()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "<%s(%r, %s)>" % (
            self.__class__.__name__, self._reader, self._writer)


# Define non-blocking read and write functions
if hasattr(gevent.os, 'nb_write'):
    # POSIX system -> use actual non-blocking I/O
    _read_nonblocking = gevent.os.nb_read
    _write_nonblocking = gevent.os.nb_write
else:
    # Windows -> imitate non-blocking I/O based on gevent threadpool
    _read_nonblocking = gevent.os.tp_read
    _write_nonblocking = gevent.os.tp_write


def _filter_handles(l):
    """Iterate through `l`, filter and yield `_GIPCHandle` instances.
    """
    for o in l:
        if isinstance(o, _GIPCHandle):
            yield o
        elif isinstance(o, _GIPCDuplexHandle):
            yield o._writer
            yield o._reader


# Container for keeping track of valid `_GIPCHandle`s in current process.
cdef list _all_handles = []


def _get_all_handles():
    """Return a copy of the list of all handles.
    """
    return _all_handles[:]


def _set_all_handles(handles):
    global _all_handles
    _all_handles = handles


# Inspect signal module for signals whose action is to be restored to the
# default action right after fork.
_signals_to_reset = [
    getattr(signal, s) for s in
    set([s for s in dir(signal) if s.startswith("SIG")]) -
    # Exclude constants that are not signals such as SIG_DFL and SIG_BLOCK.
    set([s for s in dir(signal) if s.startswith("SIG_")]) -
    # Leave handlers for SIG(STOP/KILL/PIPE) untouched.
    set(['SIGSTOP', 'SIGKILL', 'SIGPIPE'])]


def _reset_signal_handlers():
    for s in _signals_to_reset:
        # On FreeBSD, the numerical value of SIGRT* is larger than NSIG
        # from signal.h (which is a bug in my opinion). Do not change
        # action for these signals. This prevents a ValueError raised
        # in the signal module.
        if s < signal.NSIG:
            signal.signal(s, signal.SIG_DFL)
    cleanup_on_sigterm()
    signal.signal(signal.SIGINT, signal.default_int_handler)


PY3 = sys.version_info[0] == 3


# Define reraise which works for both Python 2, and 3. Taken from project six.
# The core issue here is that Python 2's raise syntax (with three arguments)
# is a syntax error in Python 3, which is why a workaround requires exec.
if PY3:
    def _reraise(tp, value, tb=None):
        if value is None:
            value = tp()
        if value.__traceback__ is not tb:
            raise value.with_traceback(tb)
        raise value
else:
    def __exec(_code_, _globs_=None, _locs_=None):
        """Execute code in a namespace."""
        if _globs_ is None:
            frame = sys._getframe(1)
            _globs_ = frame.f_globals
            if _locs_ is None:
                _locs_ = frame.f_locals
            del frame
        elif _locs_ is None:
            _locs_ = _globs_
        exec("""exec _code_ in _globs_, _locs_""")

    __exec("""def _reraise(tp, value, tb=None):
    try:
        raise tp, value, tb
    finally:
        tb = None
""")