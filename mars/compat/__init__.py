# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

import sys
import logging.config
import itertools
import platform
import struct
import warnings
import os
import socket
try:
    import xml.etree.cElementTree as ElementTree
except ImportError:
    import xml.etree.ElementTree as ElementTree
try:
    ElementTreeParseError = getattr(ElementTree, 'ParseError')
except AttributeError:
    ElementTreeParseError = getattr(ElementTree, 'XMLParserError')
from unicodedata import east_asian_width

from ..lib import six

PY27 = six.PY2 and sys.version_info[1] == 7
PYPY = platform.python_implementation().lower() == 'pypy'

SEEK_SET = 0
SEEK_CUR = 1
SEEK_END = 2

# Definition of East Asian Width
# http://unicode.org/reports/tr11/
# Ambiguous width can be changed by option
_EAW_MAP = {'Na': 1, 'N': 1, 'W': 2, 'F': 2, 'H': 1}

import decimal
DECIMAL_TYPES = [decimal.Decimal, ]

import json  # don't remove

if six.PY3:
    lrange = lambda *x: list(range(*x))
    lzip = lambda *x: list(zip(*x))
    lkeys = lambda x: list(x.keys())
    lvalues = lambda x: list(x.values())
    litems = lambda x: list(x.items())
    lmap = lambda *x: list(map(*x))

    irange = range
    izip = zip

    long_type = int

    import io
    StringIO = io.StringIO
    BytesIO = io.BytesIO

    if PY27:
        from ..lib import enum
    else:
        import enum

    if PY27:
        try:
            import cdecimal as decimal

            DECIMAL_TYPES.append(decimal.Decimal)
        except ImportError:
            import decimal
    else:
        import decimal

    from collections import OrderedDict

    OrderedDict3 = OrderedDict

    def u(s):
        return s

    def strlen(data, encoding=None):
        # encoding is for compat with PY2
        return len(data)

    def east_asian_len(data, encoding=None, ambiguous_width=1):
        """
        Calculate display width considering unicode East Asian Width
        """
        if isinstance(data, six.text_type):
            return sum([_EAW_MAP.get(east_asian_width(c), ambiguous_width) for c in data])
        else:
            return len(data)

    dictconfig = lambda config: logging.config.dictConfig(config)

    import builtins
    from concurrent import futures  # don't remove

    from datetime import timedelta
    total_seconds = timedelta.total_seconds

    import functools as functools32

    def np_getbuffer(n):
        return memoryview(n)

    BrokenPipeError = BrokenPipeError
    ConnectionResetError = ConnectionResetError
    TimeoutError = TimeoutError

    from itertools import accumulate
else:
    lrange = range
    lzip = zip
    lkeys = lambda x: x.keys()
    lvalues = lambda x: x.values()
    litems = lambda x: x.items()
    lmap = map

    irange = xrange  # noqa F821
    izip = itertools.izip

    long_type = long  # noqa F821

    from ..lib import enum

    try:
        import cdecimal as decimal
        DECIMAL_TYPES.append(decimal.Decimal)
    except ImportError:
        import decimal

    try:
        import cStringIO as StringIO
    except ImportError:
        import StringIO
    StringIO = BytesIO = StringIO.StringIO

    def u(s):
        return unicode(s, "unicode_escape")  # noqa F821

    def strlen(data, encoding=None):
        try:
            data = data.decode(encoding)
        except UnicodeError:
            pass
        return len(data)

    def east_asian_len(data, encoding=None, ambiguous_width=1):
        """
        Calculate display width considering unicode East Asian Width
        """
        if isinstance(data, six.text_type):
            try:
                data = data.decode(encoding)
            except UnicodeError:
                pass
            return sum([_EAW_MAP.get(east_asian_width(c), ambiguous_width) for c in data])
        else:
            return len(data)

    from collections import OrderedDict

    dictconfig = lambda config: logging.config.dictConfig(config)

    from datetime import timedelta
    total_seconds = timedelta.total_seconds

    import __builtin__ as builtins  # don't remove
    from ..lib import futures  # don't remove
    from ..lib.functools32.functools32 import OrderedDict as OrderedDict3

    from ..lib import functools32  # don't remove

    def np_getbuffer(n):
        import numpy as np
        return np.getbuffer(n)


    class TimeoutError(Exception):
        pass


    class BrokenPipeError(socket.error):
        pass


    class ConnectionResetError(socket.error):
        pass


    def accumulate(iterable, func=lambda a, b: a + b):
        'Return running totals'
        # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
        # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
        it = iter(iterable)
        try:
            total = next(it)
        except StopIteration:
            return
        yield total
        for element in it:
            total = func(total, element)
            yield total

if six.PY3:
    from contextlib import suppress
else:
    from contextlib import contextmanager

    @contextmanager
    def suppress(*exceptions):
        try:
            yield
        except exceptions:
            pass

Enum = enum.Enum
DECIMAL_TYPES = tuple(DECIMAL_TYPES)
Decimal = decimal.Decimal

if sys.version_info.major < 3:
    # Due to a bug in python 2.7 Queue.get, if a timeout isn't specified then
    # `Queue.get` can't be interrupted. A workaround is to specify an extremely
    # long timeout, which then allows it to be interrupted.
    # For more information see: https://bugs.python.org/issue1360
    def queue_get(q):
        return q.get(block=True, timeout=(365 * 24 * 60 * 60))

elif os.name == 'nt':
    # Python 3 windows Queue.get also doesn't handle interrupts properly. To
    # workaround this we poll at a sufficiently large interval that it
    # shouldn't affect performance, but small enough that users trying to kill
    # an application shouldn't care.
    def queue_get(q):
        while True:
            try:
                return q.get(block=True, timeout=0.1)
            except Empty:
                pass
else:
    def queue_get(q):
        return q.get()

from ..lib.lib_utils import isvalidattr, dir2, getargspec, getfullargspec

from ..lib.six.moves import reduce, zip_longest
from ..lib.six.moves import reload_module
from ..lib.six.moves.queue import Queue, Empty, PriorityQueue
from ..lib.six.moves.urllib.request import urlretrieve
from ..lib.six.moves import cPickle as pickle
from ..lib.six.moves.urllib.parse import urlencode, urlparse, unquote, quote, quote_plus, parse_qsl
from ..lib.six.moves import configparser as ConfigParser


try:
    import pytz
    utc = pytz.utc
    FixedOffset = pytz._FixedOffset
except ImportError:
    import datetime
    _ZERO_TIMEDELTA = datetime.timedelta(0)

    # A class building tzinfo objects for fixed-offset time zones.
    # Note that FixedOffset(0, "UTC") is a different way to build a
    # UTC tzinfo object.

    class FixedOffset(datetime.tzinfo):
        """Fixed offset in minutes east from UTC."""

        def __init__(self, offset, name=None):
            self.__offset = datetime.timedelta(minutes=offset)
            self.__name = name

        def utcoffset(self, dt):
            return self.__offset

        def tzname(self, dt):
            return self.__name

        def dst(self, dt):
            return _ZERO_TIMEDELTA


    utc = FixedOffset(0, 'UTC')


try:
    from weakref import finalize
except ImportError:
    # Backported from Python 3.6
    import itertools
    from weakref import ref

    class finalize:
        """Class for finalization of weakrefable objects

        finalize(obj, func, *args, **kwargs) returns a callable finalizer
        object which will be called when obj is garbage collected. The
        first time the finalizer is called it evaluates func(*arg, **kwargs)
        and returns the result. After this the finalizer is dead, and
        calling it just returns None.

        When the program exits any remaining finalizers for which the
        atexit attribute is true will be run in reverse order of creation.
        By default atexit is true.
        """

        # Finalizer objects don't have any state of their own.  They are
        # just used as keys to lookup _Info objects in the registry.  This
        # ensures that they cannot be part of a ref-cycle.

        __slots__ = ()
        _registry = {}
        _shutdown = False
        _index_iter = itertools.count()
        _dirty = False
        _registered_with_atexit = False

        class _Info:
            __slots__ = ("weakref", "func", "args", "kwargs", "atexit", "index")

        def __init__(self, obj, func, *args, **kwargs):
            if not self._registered_with_atexit:
                # We may register the exit function more than once because
                # of a thread race, but that is harmless
                import atexit
                atexit.register(self._exitfunc)
                finalize._registered_with_atexit = True
            info = self._Info()
            info.weakref = ref(obj, self)
            info.func = func
            info.args = args
            info.kwargs = kwargs or None
            info.atexit = True
            info.index = next(self._index_iter)
            self._registry[self] = info
            finalize._dirty = True

        def __call__(self, _=None):
            """If alive then mark as dead and return func(*args, **kwargs);
            otherwise return None"""
            info = self._registry.pop(self, None)
            if info and not self._shutdown:
                return info.func(*info.args, **(info.kwargs or {}))

        def detach(self):
            """If alive then mark as dead and return (obj, func, args, kwargs);
            otherwise return None"""
            info = self._registry.get(self)
            obj = info and info.weakref()
            if obj is not None and self._registry.pop(self, None):
                return (obj, info.func, info.args, info.kwargs or {})

        def peek(self):
            """If alive then return (obj, func, args, kwargs);
            otherwise return None"""
            info = self._registry.get(self)
            obj = info and info.weakref()
            if obj is not None:
                return (obj, info.func, info.args, info.kwargs or {})

        @property
        def alive(self):
            """Whether finalizer is alive"""
            return self in self._registry

        @property
        def atexit(self):
            """Whether finalizer should be called at exit"""
            info = self._registry.get(self)
            return bool(info) and info.atexit

        @atexit.setter
        def atexit(self, value):
            info = self._registry.get(self)
            if info:
                info.atexit = bool(value)

        def __repr__(self):
            info = self._registry.get(self)
            obj = info and info.weakref()
            if obj is None:
                return '<%s object at %#x; dead>' % (type(self).__name__, id(self))
            else:
                return '<%s object at %#x; for %r at %#x>' % \
                       (type(self).__name__, id(self), type(obj).__name__, id(obj))

        @classmethod
        def _select_for_exit(cls):
            # Return live finalizers marked for exit, oldest first
            L = [(f,i) for (f,i) in cls._registry.items() if i.atexit]
            L.sort(key=lambda item:item[1].index)
            return [f for (f,i) in L]

        @classmethod
        def _exitfunc(cls):
            # At shutdown invoke finalizers for which atexit is true.
            # This is called once all other non-daemonic threads have been
            # joined.
            reenable_gc = False
            try:
                if cls._registry:
                    import gc
                    if gc.isenabled():
                        reenable_gc = True
                        gc.disable()
                    pending = None
                    while True:
                        if pending is None or finalize._dirty:
                            pending = cls._select_for_exit()
                            finalize._dirty = False
                        if not pending:
                            break
                        f = pending.pop()
                        try:
                            # gc is disabled, so (assuming no daemonic
                            # threads) the following is the only line in
                            # this function which might trigger creation
                            # of a new finalizer
                            f()
                        except Exception:
                            sys.excepthook(*sys.exc_info())
                        assert f not in cls._registry
            finally:
                # prevent any more finalizers from executing during shutdown
                finalize._shutdown = True
                if reenable_gc:
                    gc.enable()


__all__ = ['PY27', 'sys', 'builtins', 'logging.config', 'OrderedDict', 'dictconfig', 'suppress',
           'reduce', 'reload_module', 'Queue', 'PriorityQueue', 'Empty', 'ElementTree', 'ElementTreeParseError',
           'urlretrieve', 'pickle', 'urlencode', 'urlparse', 'unquote', 'quote', 'quote_plus', 'parse_qsl',
           'Enum', 'ConfigParser', 'decimal', 'Decimal', 'DECIMAL_TYPES', 'FixedOffset', 'utc', 'finalize',
           'functools32', 'zip_longest', 'OrderedDict3', 'BrokenPipeError', 'TimeoutError', 'ConnectionResetError',
           'izip', 'accumulate']
