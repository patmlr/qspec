# -*- coding: utf-8 -*-
"""
qspec.types
===========

Module including types for the docstrings.
"""

# noinspection PyUnresolvedReferences
from typing import Union, Iterable, Callable, Any, SupportsFloat, SupportsIndex, Optional
from numpy import number, integer, floating, complexfloating
from numpy.typing import ArrayLike, NDArray
from sympy.core.numbers import Integer, Float, Rational
from sympy.core.add import Add
from sympy.core.mul import Mul


__all__ = ['Union', 'Iterable', 'Callable', 'Any', 'SupportsFloat', 'SupportsIndex', 'Optional',
           'Integer', 'Float', 'Rational', 'Add', 'Mul',
           'quant',
           'int_like', 'float_like', 'complex_like',
           'scalar', 'complexscalar', 'scalar_like', 'complexscalar_like', 'quant_like',
           'ndarray', 'array_like', 'array_iter',
           'sympy_core', 'sympy_like', 'sympy_quant']


# noinspection PyPep8Naming
class quant(float):
    """
    Convert a string or a number to a floating-point quantum number, if possible.
    Adding and summing quantum numbers returns a new quantum number.
    For all other algebraic operations or if different data types are involved, quantum numbers behave like floats.
    """
    def __new__(cls, value):
        try:
            if isinstance(value, str):
                value = ''.join(value.split())
                if value.endswith('/2'):
                    i = value.find('/')
                    value = float(value[:i]) / float(value[i+1:])
            value = float(value)
            if value % 0.5 != 0:
                raise ValueError()
        except ValueError:
            raise ValueError(f'{value} is not a quantum number')
        return super(quant, cls).__new__(cls, value)

    def __add__(self, other):
        ret = super(quant, self).__add__(other)
        if isinstance(other, quant):
            return self.__class__(ret)
        return ret

    def __sub__(self, other):
        ret = super(quant, self).__sub__(other)
        if isinstance(other, quant):
            return self.__class__(ret)
        return ret

    def __str__(self):
        return f'{self.p}/{self.q}' if self.q == 2 else str(self.p)

    def __repr__(self):
        return f'quant({super(quant, self).__repr__()})'

    @property
    def p(self):
        return int(self // 0.5) if self % 1 else int(self)

    @property
    def q(self):
        return 2 if self % 1 else 1

    @property
    def s(self):
        return Rational(self)


int_like = Union[NDArray[integer], integer, int]
float_like = Union[NDArray[floating], floating, float]
complex_like = Union[NDArray[complexfloating], complexfloating, complex]

scalar = Union[integer, floating, int, float]
complexscalar = Union[number, int, float, complex]
scalar_like = Union[NDArray[Union[integer, floating]], scalar]
complexscalar_like = Union[NDArray[number], complexscalar]
quant_like = Union[quant, scalar]

ndarray = NDArray
array_like = ArrayLike
array_iter = Union[NDArray, Iterable]

sympy_core = Union[Integer, Float, Rational, Add, Mul]
sympy_like = Union[Integer, Float, Rational, Add, Mul, complexscalar]
sympy_quant = Union[Integer, Float, Rational, quant_like]
