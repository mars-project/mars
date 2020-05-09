"""
This package provides the `asyncinit` decorator, which enables an asynchronous constructor
to be called like any other asynchronous function.
## Example
```python3
from asyncinit import asyncinit
@asyncinit
class MyClass:
    async def __init__(self, param):
        self.val = await self.deferredFn(param)
    async def deferredFn(self, x):
        # ...
        return x + 2
obj = await MyClass(42)
assert obj.val == 44
```
## Inheritance
Note that if a parent class is decorated with `@asyncinit`, its child classes
automatically inherit async `__init__` functionality and don't require decoration
themselves (as long as they don't override `__new__`.)
"""

import functools
import inspect


def asyncinit(obj):
    """
    Add async `__init__` functionality to the given class.
    """

    if not inspect.isclass(obj):
        raise ValueError("decorated object must be a class")

    if obj.__new__ is object.__new__:
        cls_new = _new
    else:
        cls_new = _force_async(obj.__new__)

    @functools.wraps(obj.__new__)
    async def new(cls, *args, **kwargs):
        self = await cls_new(cls, *args, **kwargs)

        cls_init = _force_async(self.__init__)
        await cls_init(*args, **kwargs)

        return self

    obj.__new__ = new

    return obj


# Force the given function to be `await`-able.
def _force_async(fn):
    if inspect.iscoroutinefunction(fn):
        return fn

    async def wrapped(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapped


# Wraps `object.__new__` in a coroutine, only passing it the class object. This kludge is
# required because that function throws `TypeError: object() takes no parameters` if
# passed any other parameters.
async def _new(cls, *args, **kwargs):
    return object.__new__(cls)
