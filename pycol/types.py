# -*- coding: utf-8 -*-
"""
PyCLS.Types

Created on 16.06.2020

@author: Patrick Mueller

Module including types for the docstrings.
"""

# noinspection PyUnresolvedReferences
from typing import Union, Iterable, Callable, Any, SupportsFloat, SupportsIndex, Optional
from numpy import ndarray
from sympy.core.numbers import Integer, Float, Rational
from sympy.core.add import Add
from sympy.core.mul import Mul

scalar = Union[int, float]
scalar_c = Union[int, float, complex]
array_iter = Union[ndarray, Iterable]
array_like = Union[ndarray, Iterable, int, float]
sympy_core = Union[Integer, Float, Rational, Add, Mul]
sympy_like = Union[Integer, Float, Rational, Add, Mul, int, float, complex]
sympy_qn = Union[Integer, Float, Rational, int, float]
