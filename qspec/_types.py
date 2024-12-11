# -*- coding: utf-8 -*-
"""
qspec.types
===========

Module including types for the docstrings.
"""

# noinspection PyUnresolvedReferences
from typing import Union, Iterable, Callable, Any, SupportsFloat, SupportsIndex, Optional
from numpy import ndarray
from numpy.typing import ArrayLike, NDArray
from sympy.core.numbers import Integer, Float, Rational
from sympy.core.add import Add
from sympy.core.mul import Mul

int_like = Union[NDArray, int]
float_like = Union[NDArray, float]
complex_like = Union[NDArray, complex]

scalar = Union[int, float]
scalar_c = Union[int, float, complex]
scalar_like = Union[NDArray, int, float]
scalar_c_like = Union[NDArray, int, float, complex]

array_iter = Union[NDArray, Iterable]
array_like = Union[NDArray, Iterable, int, float]
array_c_like = Union[NDArray, Iterable, int, float, complex]

sympy_core = Union[Integer, Float, Rational, Add, Mul]
sympy_like = Union[Integer, Float, Rational, Add, Mul, int, float, complex]
sympy_qn = Union[Integer, Float, Rational, int, float]
