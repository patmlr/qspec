"""
qspec.qtypes
============

Module including the types used in the type hints and docstrings, and type-checking and casting functions.
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
from numpy import bool_, complexfloating, floating, generic, integer, number, str_
from numpy.typing import ArrayLike, NDArray
from sympy import AtomicExpr, Rational, nsimplify

__all__ = [
    "Any",
    "Callable",
    "CodeType",
    "Generator",
    "HasGetItem",
    "HasShape",
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
    Adding and subtracting quantum numbers returns a new quantum number.
    For all other algebraic operations or if different data types are involved, quantum numbers behave like floats.

    :param value: A (half-)integer number or a string such as `"3/2"`.
    :raises ValueError: If `value` is not a (half-)integer number.
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
        """The numerator of the quantum number, e.g. 3 for 3/2."""
        return int(self // 0.5) if self % 1 else int(self)

    @property
    def q(self) -> int:
        """The denominator of the quantum number, 1 for integers and 2 for half-integers."""
        return 2 if self % 1 else 1

    @property
    def s(self) -> Rational:
        """The quantum number as a sympy `Rational`."""
        ret = Rational(self.p, self.q)
        assert isinstance(ret, Rational)
        return ret


type int_like = NDArray[integer] | integer | int
"""An integer or an array of integers."""
type float_like = NDArray[floating] | floating | float
"""A float or an array of floats."""
type complex_like = NDArray[complexfloating] | complexfloating | complex
"""A complex number or an array of complex numbers."""

type scalar = int | integer | float | floating | quant | NDArray[integer | floating]
"""A real number, including quantum numbers and 0-dimensional arrays."""
type complexscalar = number | scalar | complex | NDArray[number | complexfloating]
"""A real or complex number, including 0-dimensional arrays."""

""" Array types """
type ndarray[T: generic] = NDArray[T]
"""A numpy array with elements of type `T`."""
type array_like = ArrayLike
"""Any object that can be converted to a numpy array, e.g. scalars, lists or arrays."""

type object_1d = ndarray | Sequence[object]
"""A 1-dimensional array or sequence of objects."""
type object_nd = Sequence[object_nd] | object_1d
"""An arbitrarily nested array or sequence of objects."""

type scalar_1d = ndarray[integer | floating] | Sequence[scalar]
"""A 1-dimensional array or sequence of real numbers."""
type scalar_nd = Sequence[scalar_nd] | scalar_1d
"""An arbitrarily nested array or sequence of real numbers."""

type complexscalar_1d = ndarray[integer | floating | complexfloating] | Sequence[complexscalar]
"""A 1-dimensional array or sequence of real or complex numbers."""
type complexscalar_nd = Sequence[complexscalar_nd] | complexscalar_1d
"""An arbitrarily nested array or sequence of real or complex numbers."""

type bool_1d = ndarray[bool_] | Sequence[bool_]
"""A 1-dimensional array or sequence of booleans."""
type bool_nd = Sequence[bool_nd] | bool_1d
"""An arbitrarily nested array or sequence of booleans."""

type str_1d = ndarray[str_] | Sequence[str_]
"""A 1-dimensional array or sequence of strings."""
type str_nd = Sequence[str_nd] | str_1d
"""An arbitrarily nested array or sequence of strings."""

""" qspec.algebra types """
type sympy_expr = Any
"""A sympy expression."""
type sympy_scalar = sympy_expr | scalar
"""A sympy expression or a real number."""
type sympy_complexscalar = sympy_expr | complexscalar
"""A sympy expression or a real or complex number."""

""" qspec.models types """
type fix_type = str | bool | scalar | list
"""The fix state of a model parameter, see [`qspec.models.Model.set_fix`]."""
type fix_type_1d = ndarray | tuple[fix_type, ...] | list[fix_type]
"""A list of fix states of model parameters."""


""" type guarding methods """


class HasShape(Protocol):
    """An object with a `shape` property, e.g. a numpy array."""

    @property
    def shape(self) -> tuple[int, ...]: ...


class HasGetItem(Protocol):
    """An object that supports indexing with `[]`."""

    def __getitem__(self, i: Any) -> Any: ...


def has_shape(a: Any) -> TypeGuard[HasShape]:
    r"""
    Check if `a` has the `shape` property and `TypeGuard` it.

    :param a: The object to check.
    :returns: (has_shape) Whether `a` has the `shape` property.
    """
    return hasattr(a, "shape")


def has_getitem(a: Any) -> TypeGuard[HasGetItem]:
    r"""
    Check if `a` implements `__getitem__` so that `a[i]` can be used and `TypeGuard` it.
    Arrays with shape `()` do not support indexing.

    :param a: The object to check.
    :returns: (has_getitem) Whether `a` implements `__getitem__`.
    """
    if has_shape(a):
        return bool(a.shape)
    return bool(hasattr(a, "__getitem__"))


def is_scalar(a: Any) -> TypeGuard[scalar]:
    r"""
    Check if `a` is a scalar object and `TypeGuard` it. 0-dimensional arrays are scalars.

    :param a: The object to check.
    :returns: (is_scalar) Whether `a` is a [`scalar`].
    """
    return hasattr(a, "__float__") and not (has_shape(a) and a.shape)


def is_sympy_expr(a: Any) -> TypeGuard[sympy_expr]:
    r"""
    Check if `a` is a `sympy.AtomicExpr` instance and `TypeGuard` it.

    :param a: The object to check.
    :returns: (is_sympy_expr) Whether `a` is a `sympy.AtomicExpr` instance.
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
    Cast the arguments `args` to <a href="https://www.sympy.org/en/index.html">`sympy`</a> expressions
    using `sympy.nsimplify` or to the specified `dtype`.

    :param args: The arguments.
    :param as_sympy: Whether to cast the arguments to sympy expressions (`True`) or to `dtype` (`False`).
    :param dtype: The type to use if `as_sympy` is `False`.
    :returns: (args_cast) The cast arguments.
    """
    _dtype: Callable[[Any], Any] = nsimplify if as_sympy else dtype
    return cast(*args, dtype=_dtype)


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
