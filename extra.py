"""
File: extra.py
Date: 2017-10-07

Additional utilities.
"""
from collections.abc import Iterable, Iterator
from functools import wraps
import itertools

def namedtuple_varargs(field):
    """Wrap a NamedTuple to put varags into the given field.

    Also adds a 'cls' field to the wrapped object to allow access to the
    actual class, such as for use in isinstance.
    """
    def decorator(cls):
        @wraps(cls)
        def wrapper(*args, **kwargs):
            std = {}
            kw = {}
            for name, value in kwargs.items():
                if name in cls._fields:
                    std[name] = value
                else:
                    kw[name] = value
            varargs = {field: kw} if len(kw) > 0 else {}
            return cls(*args, **std, **varargs)

        if not isinstance(cls, type):
            raise TypeError(f"'{cls}' is not a type")
        if not hasattr(cls, '_fields'):
            raise TypeError(f"'{cls.__name__}' is not a NnamedTuple")
        if field not in cls._fields:
            raise ValueError(f"No such field '{field}' in NamedTuple '{cls.__name__}'")
        wrapper.cls = cls
        return wrapper
    return decorator


class ChainableCreator(object):
    "Class for methods that create Chainable iterators."
    def __init__(self, base):
        self.base = base
    def get(self):
        return self.base
    def __call__(self, *args, **kwargs):
        return Chainable(self.base(*args, **kwargs))
    def __repr__(self):
        return f"ChainableCreator({self.base!r})"

class Chainable(Iterator):
    "Class that allows use of itertools.chain by the addition operator."
    def __init__(self, iterable, *components):
        self.base = iterable
        if not components:
            self.components = (iterable,)
        else:
            comp = []
            for i in components:
                if isinstance(i, Chainable):
                    comp.extend(i.components)
                else:
                    comp.append(i)
            self.components = tuple(comp)

    def get(self):
        return self.base
    def __call__(self, *args, **kwargs):
        return self.base(*args, **kwargs)
    def __add__(self, other):
        return Chainable(itertools.chain(self, other), self, other)
    def __radd__(self, other):
        return Chainable(itertools.chain(other, self), other, self)
    def __iter__(self):
        return self
    def __next__(self):
        return next(self.base)
    def __repr__(self):
        return f"Chainable{self.components!r}"


def repeat(elem, n=None):
    "Chainable itertools.repeat"
    if n is None:
        return Chainable(repeat(elem))
    return Chainable(itertools.repeat(elem, n))

def repeat_n(n, elem):
    "Chainable itertools.repeat with reversed arguments"
    return Chainable(itertools.repeat(elem, n))

def chain(*iterables):
    "Chainable itertools.chain"
    return Chainable(itertools.chain(*iterables), *iterables)



# class Chain(chain):
#     def __init__(self, *args):
#         self.args = args
#         super(chain, self).__init__(self, *args)
#     def __repr__(self):
#         return f"Chain({self.args!r})"
