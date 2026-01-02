"""
qspec.qtypes
============

Module including types for the docstrings.
"""

from collections.abc import Callable, Generator, Iterable, Sequence
from types import CodeType
from typing import (
    Any,
    Protocol,
    SupportsFloat,
    SupportsIndex,
    TypeGuard,
)

from numpy import asarray as np_asarray
from numpy import bool_, complexfloating, floating, integer, number, str_
from numpy.typing import ArrayLike, NDArray
from sympy import AtomicExpr, Rational, nsimplify

__all__ = [
    "Any",
    "Callable",
    "CodeType",
    "Generator",
    "Iterable",
    "Protocol",
    "Sequence",
    "SupportsFloat",
    "SupportsIndex",
    "TypeGuard",
    "array_like",
    "asarray",
    "asarray_none",
    "bool_1d",
    "bool_nd",
    "cast",
    "cast_sympy",
    "complex_like",
    "complexfloating",
    "complexscalar",
    "complexscalar_1d",
    "complexscalar_nd",
    "fix_type",
    "fix_type_1d",
    "float_like",
    "floating",
    "has_getitem",
    "has_shape",
    "int_like",
    "integer",
    "is_scalar",
    "is_sympy_expr",
    "ndarray",
    "object_1d",
    "object_nd",
    "quant",
    "scalar",
    "scalar_1d",
    "scalar_nd",
    "str_1d",
    "str_nd",
    "sympy_complexscalar",
    "sympy_expr",
    "sympy_scalar",
]


""" Scalar types """


class quant(float):
    """
    Convert a string or a number to a floating-point quantum number, if possible.
    Adding and summing quantum numbers returns a new quantum number.
    For all other algebraic operations or if different data types are involved, quantum numbers behave like floats.
    """

    def __new__(cls, value: SupportsFloat | str) -> "quant":
        try:
            if isinstance(value, str):
                value = "".join(value.split())
                if value.endswith("/2"):
                    i = value.find("/")
                    value = float(value[:i]) / float(value[i + 1 :])

            value = float(value)

            if value % 0.5 != 0:
                raise ValueError()

        except ValueError:
            raise ValueError(f"{value} is not a (half-)integer number")

        return super().__new__(cls, value)

    def __add__(self, other: Any) -> "float | quant":
        ret = super().__add__(other)
        if isinstance(other, quant):
            return self.__class__(ret)
        return ret

    def __sub__(self, other: Any) -> "float | quant":
        ret = super().__sub__(other)
        if isinstance(other, quant):
            return self.__class__(ret)
        return ret

    def __str__(self) -> str:
        return f"{self.p}/{self.q}" if self.q == 2 else str(self.p)

    def __repr__(self) -> str:
        return f"quant({super().__repr__()})"

    @property
    def p(self) -> int:
        return int(self // 0.5) if self % 1 else int(self)

    @property
    def q(self) -> int:
        return 2 if self % 1 else 1

    @property
    def s(self) -> Rational:
        return Rational(self)


type int_like = NDArray[integer] | integer | int
type float_like = NDArray[floating] | floating | float
type complex_like = NDArray[complexfloating] | complexfloating | complex

type scalar = int | integer | float | floating | quant | NDArray[integer | floating]
type complexscalar = number | scalar | complex | NDArray[number | complexfloating]

""" Array types """
ndarray = NDArray
type array_like = ArrayLike

type object_1d = ndarray | Sequence[object]
type object_nd = Sequence[object_nd] | object_1d

type scalar_1d = ndarray[integer | floating] | Sequence[scalar]
type scalar_nd = Sequence[scalar_nd] | scalar_1d

type complexscalar_1d = ndarray[integer | floating | complexfloating] | Sequence[complexscalar]
type complexscalar_nd = Sequence[complexscalar_nd] | complexscalar_1d

type bool_1d = ndarray[bool_] | Sequence[bool_]
type bool_nd = Sequence[bool_nd] | bool_1d

type str_1d = ndarray[str_] | Sequence[str_]
type str_nd = Sequence[str_nd] | str_1d

""" qspec.algebra types """
type sympy_expr = Any
type sympy_scalar = sympy_expr | scalar
type sympy_complexscalar = sympy_expr | complexscalar

""" qspec.models types """
type fix_type = str | bool | scalar | list
type fix_type_1d = ndarray | tuple[fix_type, ...] | list[fix_type]


""" type guarding methods """


class HasShape(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...


class HasGetItem(Protocol):
    def __getitem__(self, i: Any) -> Any: ...


def has_shape(a: Any) -> TypeGuard[HasShape]:
    r"""
    Check if `a` has the `shape` property and `TypeGuard` it.

    :param a: The object to check.
    :returns: (has_shape) if `a` has the `shape` property.
    """
    return hasattr(a, "shape")


def has_getitem(a: Any) -> TypeGuard[HasGetItem]:
    r"""
    Check if `a` implements `__getitem__` so that `a[i]` can be used and `TypeGuard` it.

    :param a: The object to check.
    :returns: (has_getitem) if `a` implements `__getitem__`.
    """
    if has_shape(a):
        return bool(a.shape)
    return bool(hasattr(a, "__getitem__"))


def is_scalar(a: Any) -> TypeGuard[scalar]:
    r"""
    Check if `a` is a scalar object and `TypeGuard` it.

    :param a: The object to check.
    :returns: (is_scalar) if `a` is a `scalar`.
    """
    return hasattr(a, "__float__") and not (has_shape(a) and a.shape)


def is_sympy_expr(a: Any) -> TypeGuard[sympy_expr]:
    r"""
    Check if `a` is a `sympy.AtomExpr` instance and `TypeGuard` it.

    :param a: The object to check.
    :returns: (is_sympy_expr) if `a` is a `sympy.AtomExpr` instance.
    """
    return isinstance(a, AtomicExpr)


""" Type conversion methods """


def cast[T](*args: Any, dtype: Callable[[Any], T]) -> tuple[T, ...]:
    r"""
    Cast the arguments `args` to the specified `dtype`.

    :param args: The arguments.
    :param dtype: The type of the cast arguments.
    :returns: (args_cast) The cast arguments.
    """
    return tuple(dtype(arg) for arg in args)


def cast_sympy[T](*args: sympy_scalar, as_sympy: bool = True, dtype: Callable[[Any], T] = float) -> tuple[T, ...]:
    r"""
    Cast the arguments `args` to a <a href="https://www.sympy.org/en/index.html">
    `sympy`</a> type (symbol) or the specified `dtype`.

    :param args: The arguments.
    :param as_sympy: Return the result as a symbol (`True`) or as a `float` (`False`).
    :param dtype: The type to use if `as_sympy == False`.
    :returns: (args_cast) The cast arguments.
    """
    return cast(*args, dtype=(nsimplify if as_sympy else dtype))


def asarray(*args: Any, **kwargs: Any) -> tuple[ndarray, ...]:
    r"""
    Cast the arguments `args` to a tuple of `ndarray` of the specified `dtype`.

    :param args: The arguments to cast.
    :param kwargs: Additional keywords are passed to
     <a href="https://numpy.org/doc/stable/reference/generated/numpy.asarray.html">
     `numpy.asarray`</a>.
    :returns: (a_tuple) A tuple of `ndarray`.
    """
    return tuple(np_asarray(a, **kwargs) for a in args)


def asarray_none(*args: Any | None, **kwargs: Any) -> tuple[ndarray | None, ...]:
    """
    Cast the arguments `args` to a tuple of `ndarray` of the specified `dtype` or `None`.

    :param args: The arguments to cast.
    :param kwargs: Additional keywords are passed to
     <a href="https://numpy.org/doc/stable/reference/generated/numpy.asarray.html">
     `numpy.asarray`</a>.
    :returns: (a_tuple) A tuple of `ndarray` or `None`.
    """
    return tuple(None if a is None else np_asarray(a, **kwargs) for a in args)
