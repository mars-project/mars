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

import ast
import keyword
import inspect
from collections import namedtuple

from . import six


if six.PY3:
    def isvalidattr(ident):
        return not keyword.iskeyword(ident) and ident.isidentifier()
else:
    def isvalidattr(ident):
        """Determines, if string is valid Python identifier."""

        # Smoke test - if it's not string, then it's not identifier, but we don't
        # want to just silence exception. It's better to fail fast.
        if not isinstance(ident, str):
            raise TypeError('expected str, but got {!r}'.format(type(ident)))

        # Resulting AST of simple identifier is <Module [<Expr <Name "foo">>]>
        try:
            root = ast.parse(ident)
        except SyntaxError:
            return False

        if not isinstance(root, ast.Module) \
                or len(root.body) != 1 \
                or not isinstance(root.body[0], ast.Expr) \
                or not isinstance(root.body[0].value, ast.Name) \
                or root.body[0].value.id != ident:
            return False

        return True


if six.PY3:
    def dir2(obj):
        return object.__dir__(obj)
else:
    # http://www.quora.com/How-dir-is-implemented-Is-there-any-PEP-related-to-that
    def get_attrs(obj):
        import types
        if not hasattr(obj, '__dict__'):
            return []  # slots only
        if not isinstance(obj.__dict__, (dict, types.DictProxyType)):
            raise TypeError("%s.__dict__ is not a dictionary"
                            "" % obj.__name__)
        return obj.__dict__.keys()

    def dir2(obj):
        attrs = set()
        if not hasattr(obj, '__bases__'):
            # obj is an instance
            if not hasattr(obj, '__class__'):
                # slots
                return sorted(get_attrs(obj))
            klass = obj.__class__
            attrs.update(get_attrs(klass))
        else:
            # obj is a class
            klass = obj

        for cls in klass.__bases__:
            attrs.update(get_attrs(cls))
            attrs.update(dir2(cls))
        attrs.update(get_attrs(obj))
        return sorted(list(attrs))


def _getargspec(func):
    ArgSpec = namedtuple('ArgSpec', 'args varargs keywords defaults')

    args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, ann = \
        getfullargspec(func)
    if kwonlyargs or ann:
        raise ValueError("Function has keyword-only arguments or annotations"
                         ", use getfullargspec() API which can support them")
    return ArgSpec(args, varargs, varkw, defaults)


def _getfullargspec(func):  # noqa: C901
    from inspect import Parameter
    FullArgSpec = namedtuple('FullArgSpec',
                             'args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations')

    sig = inspect.signature(func)
    args = []
    varargs = None
    varkw = None
    kwonlyargs = []
    annotations = {}
    defaults = ()
    kwdefaults = {}

    if sig.return_annotation is not sig.empty:
        annotations['return'] = sig.return_annotation

    for param in sig.parameters.values():
        kind = param.kind
        name = param.name

        if kind is Parameter.POSITIONAL_ONLY:
            args.append(name)
        elif kind is Parameter.POSITIONAL_OR_KEYWORD:
            args.append(name)
            if param.default is not param.empty:
                defaults += (param.default,)
        elif kind is Parameter.VAR_POSITIONAL:
            varargs = name
        elif kind is Parameter.KEYWORD_ONLY:
            kwonlyargs.append(name)
            if param.default is not param.empty:
                kwdefaults[name] = param.default
        elif kind is Parameter.VAR_KEYWORD:
            varkw = name

        if param.annotation is not param.empty:
            annotations[name] = param.annotation

    if not kwdefaults:
        # compatibility with 'func.__kwdefaults__'
        kwdefaults = None

    if not defaults:
        # compatibility with 'func.__defaults__'
        defaults = None

    return FullArgSpec(args, varargs, varkw, defaults,
                       kwonlyargs, kwdefaults, annotations)


if hasattr(inspect, 'signature'):
    getargspec, getfullargspec = _getargspec, _getfullargspec
else:
    from inspect import getargspec  # noqa: F401
    if hasattr(inspect, 'getfullargspec'):
        from inspect import getfullargspec
    else:
        getfullargspec = None
