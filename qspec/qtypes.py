"""
qspec.qtypes
============

Module including types for the docstrings.
"""

from collections.abc import Callable, Generator, Iterable, Sized
from types import CodeType
from typing import Any, Optional, Protocol, SupportsFloat, SupportsIndex, TypeGuard, TypeVar

from numpy import asarray as np_asarray
from numpy import complexfloating, floating, integer, number
from numpy.typing import ArrayLike, NDArray
from sympy import nsimplify
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import Float, Integer, Rational

__all__ = [
    "Add",
    "Any",
    "Callable",
    "CodeType",
    "Float",
    "Generator",
    "Integer",
    "Iterable",
    "Mul",
    "Optional",
    "Rational",
    "Sized",
    "SupportsFloat",
    "SupportsIndex",
    "array_iter",
    "array_like",
    "asarray",
    "cast",
    "cast_sympy",
    "complex_like",
    "complexfloating",
    "complexscalar",
    "complexscalar_like",
    "float_like",
    "has_getitem",
    "has_shape",
    "int_like",
    "is_scalar",
    "ndarray",
    "quant",
    "quant_iter",
    "quant_like",
    "scalar",
    "scalar_like",
    "sympy_complexscalar",
    "sympy_core",
    "sympy_like",
    "sympy_number",
    "sympy_quant",
    "sympy_scalar",
]


T = TypeVar("T")


def cast[T](*args: object, dtype: Callable[[Any], T]) -> tuple[T, ...]:
    return tuple(dtype(arg) for arg in args)


class HasGetItem(Protocol):
    def __getitem__(self, i: Any) -> Any: ...


def has_getitem(a: object) -> TypeGuard[HasGetItem]:
    return hasattr(a, "__getitem__")


class HasShape(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...


def has_shape(a: object) -> TypeGuard[HasShape]:
    return hasattr(a, "shape")


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


int_like = NDArray[integer] | integer | int
float_like = NDArray[floating] | floating | float
complex_like = NDArray[complexfloating] | complexfloating | complex

scalar = number | integer | floating | int | float
complexscalar = number | int | float | complex
quantscalar = quant | scalar
scalar_like = NDArray[integer | floating] | scalar
complexscalar_like = NDArray[number] | complexscalar
quant_like = quant | scalar_like

ndarray = NDArray
array_like = ArrayLike
array_iter = NDArray | Iterable[complexscalar_like] | HasGetItem
quant_iter = quant_like | Iterable[quant_like] | HasGetItem

sympy_core = Integer | Float | Rational | Add | Mul
sympy_scalar = sympy_core | scalar
sympy_complexscalar = sympy_core | complexscalar
sympy_number = sympy_core | scalar | complexscalar
sympy_like = sympy_core | scalar_like
sympy_quant = sympy_core | quant_like


def is_scalar(a: object) -> TypeGuard[scalar_like]:
    return isinstance(a, scalar) or (has_shape(a) and a.shape == ())


def is_sympy_core(a: object) -> TypeGuard[sympy_core]:
    return isinstance(a, sympy_core)


def cast_sympy[T](
    *args: sympy_like, as_sympy: bool = True, dtype: Callable[[Any], T] = float
) -> tuple[T | sympy_core, ...]:
    r"""
    Cast the arguments `args` to a <a href="https://www.sympy.org/en/index.html">
    `sympy`</a> type (symbol) or the specified `dtype`.

    :param args: The arguments.
    :param as_sympy: Return the result as a symbol (`True`) or as a `float` (`False`).
    :param dtype: The type to use if `as_sympy == False`.
    :returns: (cast_args) The cast arguments.
    """
    return cast(*args, dtype=(nsimplify if as_sympy else dtype))


def asarray(*args: object, **kwargs) -> tuple[ndarray, ...]:
    return tuple(np_asarray(a, **kwargs) for a in args)
