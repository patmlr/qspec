"""
qspec.tools
===========

Module including mathematical and technical methods.
"""

import itertools
import sqlite3

import numpy as np

from qspec.qtypes import (
    Callable,
    Iterable,
    Rational,
    array_iter,
    array_like,
    asarray,
    floating,
    int_like,
    integer,
    ndarray,
    scalar,
    scalar_like,
)

__all__ = [
    "COLORS",
    "ROMAN_NUMERALS",
    "Rotation",
    "absolute",
    "absolute_complex",
    "add_nested_key",
    "angle",
    "angle_d",
    "asarray_optional",
    "check_dimension",
    "check_half_integer",
    "check_iterable",
    "check_shape",
    "check_shape_like",
    "combine_dicts",
    "convolve_dict",
    "dict_to_list",
    "e_phi",
    "e_r",
    "e_theta",
    "even",
    "factorial",
    "floor_log2",
    "floor_log10",
    "fraction",
    "get_decimals",
    "get_rgb_print_command",
    "get_val_with_unc",
    "half_integer_to_fraction",
    "half_integer_to_str",
    "import_iso_shifts_tilda",
    "in_nested",
    "list_to_dict",
    "list_to_excel",
    "make_str_iterable_unique",
    "map_corr_coeff_to_color",
    "merge_dicts",
    "merge_intervals",
    "nan_helper",
    "odd",
    "orthonormal",
    "orthonormal_rtp",
    "print_colored",
    "print_cov",
    "printf",
    "printh",
    "printw",
    "rgb_to_hex_color",
    "roman_to_int",
    "rotation_matrix",
    "rotation_to_vector",
    "round_to_n",
    "transform",
    "unit_vector",
]


ROMAN_NUMERALS = {"M": 1000, "D": 500, "C": 100, "L": 50, "X": 10, "V": 5, "I": 1}


class COLORS:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    def __init__(self) -> None:
        pass

    @classmethod
    def tab10(cls, i: int) -> str:
        """
        Returns the pyplot tab10 color for the specified index.

        :param i: The index of the pyplot color.
        :returns: The `str` format of the specified color.
        """
        return f"C{i}"


def get_rgb_print_command(r: int_like, g: int_like, b: int_like) -> str:
    """
    :param r: The fraction of red (0-255).
    :param g: The fraction of green (0-255).
    :param b: The fraction of blue (0-255).
    :returns: The command str to print rgb colors in the console.
    """
    return f"\033[38;2;{r};{g};{b}m"


def rgb_to_hex_color(r: int_like, g: int_like, b: int_like) -> str:
    """
    :param r: The fraction of red (0-255).
    :param g: The fraction of green (0-255).
    :param b: The fraction of blue (0-255).
    :returns: A hexadecimal str representation of the `rgb` tuple.
    """
    return f"#{r:02x}{g:02x}{b:02x}"


def print_colored(specifier: str, *values: object, returned: bool = False, **kwargs) -> str | None:
    """
    Print with the specified color.

    :param specifier: str of the color name defined in the COLORS class or an RGB-value (0-255, 0-255, 0-255).
    :param values: The values to print.
    :param returned: Return the str instead of printing it.
    :param kwargs: See print().
    :returns: The colored `values` if `returned == True` else `None`.
    """
    _c = eval(f"COLORS.{specifier.upper()}") if isinstance(specifier, str) else get_rgb_print_command(*specifier)
    _values = (_c, *values, COLORS.ENDC)
    if returned:
        return "{}{}{}".format(*_values)
    print("{}{}{}".format(*_values), **kwargs)
    return None


def printh(*values: object, **kwargs) -> None:
    """
    Print with the HEADER color (pink).

    :param values: The values to print.
    :param kwargs: See print().
    :returns:
    """
    _values = (COLORS.HEADER, *values, COLORS.ENDC)
    print("{}{}{}".format(*_values), **kwargs)


def printw(*values: object, **kwargs) -> None:
    """
    Print with the WARNING color (yellow).

    :param values: The values to print.
    :param kwargs: See print().
    :returns:
    """
    _values = (COLORS.WARNING, *values, COLORS.ENDC)
    print("{}{}{}".format(*_values), **kwargs)


def printf(*values: object, **kwargs) -> None:
    """
    Print with the FAIL color (red).

    :param values: The values to print.
    :param kwargs: See print().
    :returns:
    """
    _values = (COLORS.FAIL, *values, COLORS.ENDC)
    print("{}{}{}".format(*_values), **kwargs)


def map_corr_coeff_to_color(val: scalar_like, clip: bool = True) -> tuple[int, int, int]:
    """
    Maps a value between -1 and 1 to a red-to-green colormap.

    :param val: A value between -1 and 1.
    :param clip: Whether to clip vals outside -1 and 1. If False, raise ValueError.
    :returns: (r, g, b) A tuple of RGB values in the range 0-255.
    """
    if clip:
        if np.isnan(val):
            val = 0.0
        elif val < -1:
            val = -1
        elif val > 1:
            val = 1
    if np.isnan(val) or val < -1 or val > 1:
        raise ValueError("The correlation coefficient must be in [-1, 1].")
    g = int(round(float(val) * 127 + 127, 0))
    return 255 - g, g, 0


def print_cov(cov: array_like, normalize: bool = False, decimals: int = 2) -> None:
    """
    Print a covariance "as is" or as color-coded Pearson correlation coefficients.

    :param cov: A covariance matrix.
    :param normalize: Whether to normalize the covariance to be the Pearson correlation coefficient.
    :param decimals: The number of decimal places to be printed.
    :returns:
    """
    cov = np.array(cov, dtype=float)
    if normalize:
        norm = np.sqrt(np.diag(cov)[:, None] * np.diag(cov))
        nonzero = norm != 0
        cov[nonzero] /= norm[nonzero]
    cov = np.around(cov + 0.0, decimals=decimals)
    digits = int(np.floor(np.log10(np.abs(cov.shape[0])))) + 1
    for i, row in enumerate(cov):
        print(
            "{}:   {}".format(
                str(i).zfill(digits),
                "   ".join(
                    "{}{}{}".format(
                        get_rgb_print_command(*map_corr_coeff_to_color(val)),
                        f"{val:1.2f}".rjust(decimals + 3),
                        COLORS.ENDC,
                    )
                    for val in row
                ),
            )
        )


""" literal -> numeral operations """


def fraction(r: Rational | str) -> tuple[int, int]:
    """
    :param r: A sympy.Rational or a str with the signature "'num'/'denom'".
    :returns: the numerator and denominator of 'r'.
    """
    if not isinstance(r, Rational) and not isinstance(r, str):
        raise TypeError(f"Argument must be a sympy.Rational or str, but is {type(r)}: {r}.")
    string = str(r)
    slash = string.find("/")
    if slash == -1:
        num = int(string)
        denom = 1
    else:
        num = int(string[:slash])
        denom = int(string[(slash + 1) :])
    return num, denom


def check_half_integer(*args: scalar_like) -> None:
    """
    :param args: Scalar arguments.
    :returns: Checks whether the given arguments are multiples of 1/2.
    :raises ValueError: If any argument is not a multiple of 1/2.
    """
    if sum([abs(float(arg) % 0.5) for arg in args]) > 0:
        raise ValueError("The numbers must be multiples of 1/2.")


def half_integer_to_fraction(val: scalar_like) -> tuple[int, int]:
    """
    :param val: A scalar value.
    :returns: The numerator and denominator of the half-integer.
    """
    check_half_integer(val)
    if abs(float(val) % 1.0) > 0.0:
        return int(2 * val), 2
    return int(val), 1


def half_integer_to_str(val: scalar_like, symbol: str = "/") -> str:
    """
    :param val: A scalar value.
    :param symbol: The symbol to use for the fraction.
    :returns: A rational str-representation of a half-integer.
    """
    num, denom = half_integer_to_fraction(val)
    if denom == 1:
        return str(num)
    return f"{num}{symbol}2"


def get_val_with_unc(val_string: str) -> tuple[float, float]:
    """
    :param val_string: The str representation of a number with uncertainty of the format '1.234(56)'.
     Decimal separators can be used or spared arbitrarily.
    :returns: The value and its uncertainty.
    """
    err, decimals = 0.0, 0
    if val_string == "":
        return err, err

    lbrace = val_string.find("(")
    rbrace = val_string.find(")")

    val = float(val_string) if lbrace == -1 else float(val_string[:lbrace])
    err_string = val_string[(lbrace + 1) : rbrace]

    if -1 not in [lbrace, rbrace, len(err_string) - 1]:
        err = float(err_string)

        if "." in val_string[:lbrace]:
            dot = val_string.find(".")
            decimals = len(val_string[(dot + 1) : lbrace])

            if "." not in val_string[lbrace:rbrace]:
                err *= eval(f"1e-{decimals}")
            else:
                dot = val_string[lbrace:].find(".")
                decimals = max([decimals, len(val_string[(dot + 1) : rbrace])])

    return float(np.around(val, decimals)), float(np.around(err, decimals))


def roman_to_int(roman: str) -> int:
    """
    Convert from Roman numerals to an integer
    [jonrsharpe, https://codereview.stackexchange.com/questions/68297/convert-roman-to-int].

    :param roman: The str representation of a roman number.
    """
    numbers = []
    for char in roman.upper():
        numbers.append(ROMAN_NUMERALS[char])
    total = 0
    num2 = numbers[-1]
    for num1, num2 in itertools.pairwise(numbers):
        if num1 >= num2:
            total += num1
        else:
            total -= num1
    return total + num2


""" numeral -> numeral operations """


def odd(x: scalar_like) -> int:
    """
    :param x: A scalar value.
    :returns: The closest odd integer value.
    """
    return int(x) if int(x) % 2 else int(x) + 1


def even(x: scalar_like) -> int:
    """
    :param x: A scalar value.
    :returns: The closest even integer value.
    """
    return int(x) if not int(x) % 2 else int(x) + 1


def get_decimals(x: float) -> int:
    """
    :param x: A float value.
    :return: The number of displayed decimals of the float value 'x'.
    """
    s = str(x)
    dot = s.find(".")
    return len(s[(dot + 1) :])


def floor_log2(x: array_like) -> ndarray:
    """
    :param x: Scalar values.
    :returns: The closest integer values below the logarithm with basis 10 of the absolute value of 'x'.
    """
    x = np.asarray(x)
    if len(x.shape) > 0:
        nonzero = x.nonzero()
        n = np.zeros_like(x)
        n[nonzero] = np.floor(np.log2(np.abs(x[nonzero]))).astype(int)
        return n
    if x == 0:
        return np.array(0)
    return np.floor(np.log2(np.abs(x))).astype(int)


def floor_log10(x: array_like) -> ndarray:
    """
    :param x: Scalar values.
    :returns: The closest integer values below the logarithm with basis 10 of the absolute value of 'x'.
    """
    x = np.asarray(x)
    if len(x.shape) > 0:
        nonzero = x.nonzero()
        n = np.zeros_like(x)
        n[nonzero] = np.floor(np.log10(np.abs(x[nonzero]))).astype(int)
        return n
    if x == 0:
        return np.array(0)
    return np.floor(np.log10(np.abs(x))).astype(int)


def round_to_n(x: scalar_like, n: int_like) -> tuple[scalar, int]:
    """
    :param x: The input data.
    :param n: The number of significant decimal places to round to.
    :returns: x rounded to 'n' significant decimal places and
     the corresponding number of decimal places after the decimal point.
    """
    if isinstance(x, np.ndarray):
        x = x.squeeze().tolist()
        if isinstance(x, list):
            raise TypeError("Only `scalar` types are supported.")

    if isinstance(x, (float, floating)):
        _f = float
    elif isinstance(x, (int, integer)):
        _f = int
    else:
        raise TypeError(f"Only `scalar` types are supported, but got {type(x)}.")

    if x == 0.0:
        return _f(0), 0

    decimals = -int(np.floor(np.log10(np.abs(x)))) + int(n) - 1
    return _f(np.around(x, decimals=decimals)), decimals


def factorial(n: array_like) -> ndarray:
    """
    :param n: The integer numbers. Float types are cast to int types.
    :returns: n! (array compatible).
    """
    n = np.asarray(n, dtype=int)
    mask = n > 1
    ret = np.ones_like(n)
    if np.any(mask):
        ret[mask] = n[mask] * factorial(n[mask] - 1)
    return ret


""" Iterable operations """


def asarray_optional(a: array_like | None, **kwargs) -> ndarray | None:
    """
    :param a: Input data, see numpy docs.
    :param kwargs: The keyword arguments are passed to `numpy.asarray`.
    :returns: None if `a` is None else `numpy.asarray(a, **kwargs)`.
    """
    return None if a is None else np.asarray(a, **kwargs)


def in_nested(a: object, nested: object) -> bool:
    """
    :param a: The element to look for.
    :param nested: The nested list.
    :returns: Whether `a` is inside the `nested` list.
    """
    if not isinstance(nested, Iterable):
        return False

    if a in nested:
        return True

    ret = False
    for el in nested:
        ret += in_nested(a, el)
        if ret:
            return True

    return bool(ret)


def check_iterable(arg: object, dtype: type = str) -> list:
    """
    :param arg: The argument that is checked.
    :param dtype: The type that has to be matched by the elements of arg or arg itself.
    :returns: 'arg' as a list if it is a 'list', 'tuple' or 'set' of values of type 'dtype',
     a list of the single element 'arg' if it is of type 'dtype' and else an empty list.
    """
    if isinstance(arg, (list, tuple, set)):
        if all(isinstance(s, dtype) for s in arg):
            return list(arg)
        return []

    if isinstance(arg, dtype):
        return [arg]

    return []


def make_str_iterable_unique(a: Iterable[str], identifier: Iterable[str] | None = None) -> list[str]:
    """
    :param a: An Iterable of str values.
    :param identifier: An Iterable of str values to append to the values of 'a' if they appear more than once.
     If None, '_i' is used as the identifier for the ith appearance of a value in 'a'.
    :returns: 'a', but the elements which appear more than once are numerated/attached with the identifiers and bunched.
     For example: ['a', 'b', 'c', 'b', 'd'] -> ['a', 'b_0', 'b_1', 'c', 'd'] or
     ['a', 'b', 'c', 'b', 'd'] -> ['a', 'b' + identifier[0], 'b' + identifier[1], 'c', 'd'].
    """
    temp_1 = list(a)
    temp_2 = []
    if identifier is not None:
        identifier = list(identifier)

    for p in temp_1:
        if p not in temp_2:
            temp_2.append(p)

    ret = []
    for p in temp_2:
        m = temp_1.count(p)

        for i in range(m):
            if m == 1:
                ret.append(p)
            else:
                if identifier is None:
                    ret.append(p + f"_{i}")
                else:
                    ret.append(p + f"_{identifier[i % len(identifier)]}")
    return ret


def check_dimension(dim: int, axis: int, *args: array_like) -> None:
    """
    :param dim: The number of components the axis 'axis' must have.
    :param axis: The axis which must have 'dim' components.
    :param args: The arguments which must have 'dim' components along axis 'axis'.
    :returns: None
    :raises ValueError: All specified arguments must have 'dim' components along axis 'axis'.
    """
    err_list = []
    for arg in args:
        arg = np.asarray(arg)
        if arg.shape[axis] != dim:
            err_list.append(str(arg.shape[axis]))

    if err_list:
        err_msg = "One argument has {} component(s) instead of {} along axis {}.".format(", ".join(err_list), dim, axis)
        if len(err_list) > 1:
            err_msg = "{} arguments have {} component(s) instead of {} along axis {}.".format(
                len(err_list), ", ".join(err_list), dim, axis
            )
        raise ValueError(err_msg)


def check_shape_like(*args: ndarray, allow_scalar: bool = True) -> None:
    """
    :param args: The arguments which must have shapes that can be broadcast together.
    :param allow_scalar: Whether scalar values (shape=()) are seen as compatible with arbitrary shapes.
    :returns: None
    :raises ValueError: All specified arguments must have shapes that can be broadcast together.
    """
    err_list = []
    shape = args[-1].shape
    allowed = [len(shape)]
    if allow_scalar:
        allowed.append(0)
    for arg in args[:-1]:
        if len(arg.shape) in allowed:
            if arg.shape != () and any(i != j and i != 1 and j != 1 for i, j in zip(arg.shape, shape)):
                err_list.append(str(arg.shape))
        else:
            err_list.append(str(arg.shape))
    if err_list:
        err_msg = "Operands could not be broadcast together with shapes {} and {}.".format(", ".join(err_list), shape)
        raise ValueError(err_msg)


def check_shape(shape: tuple, *args: ndarray, allow_scalar: bool = True, return_mode: bool = False) -> bool | None:
    """
    :param shape: The shape which must be matched by the specified arguments.
    :param args: The arguments which must match the specified shape.
    :param allow_scalar: Whether scalar values (shape=()) are seen as compatible with arbitrary shapes.
    :param return_mode: Whether to raise a ValueError or return a boolean if any argument does not have shape 'shape'.
    :returns: whether the arguments match the specified shape.
    :raises ValueError: All specified arguments must match the specified shape.
    """
    err_list = []
    allowed = [shape]
    if allow_scalar:
        allowed.append(())

    for arg in args:
        if arg.shape not in allowed:
            err_list.append(str(arg.shape))

    if err_list:
        if return_mode:
            return False
        err_msg = "Operand(s) with shape(s) {} do not match shape {}.".format(", ".join(err_list), shape)
        raise ValueError(err_msg)

    return True


def nan_helper(y: array_like) -> tuple[ndarray, Callable]:
    """
    :param y: An array like value.
    :returns: The mask where y is NaN and a function which returns the nonzero indices of an array.
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def dict_to_list(a: dict) -> tuple[list, list]:
    """
    :param a: A dictionary.
    :returns: Two lists with the keys and values of 'a', respectively.
     The lists are in the same order, however, they are not sorted.
    """
    kv = [[k, v] for k, v in a.items()]
    return [k[0] for k in kv], [v[1] for v in kv]


def list_to_dict(values: array_like, keys: array_like | None = None) -> dict:
    """
    :param values: The values of the dictionary.
    :param keys: The keys of the dictionary. If omitted, the keys will be the indices of the values.
    :returns: A dictionary with the given 'values' and 'keys'.
    """
    v_list = list(np.asarray(values).flatten())
    k_list = list(range(len(v_list))) if keys is None else list(np.asarray(keys).flatten())
    return {k: v for k, v in zip(k_list, v_list)}


def list_to_excel(
    *args: array_iter, save: str | None = None, delimiter: str = "\t", header: str = "", align: str = "top"
) -> str:
    """

    :param args: 1- or 2-dimensional iterables to print or save in an excel-compatible format.
    :param save: The filepath, either absolute or relative to the working directory. If None, the output is printed.
    :param delimiter: The delimiter between two columns.
    :param header: Add a header above the content to the output.
    :param align: How to align the columns that are smaller than the largest column.
     Supported alignments are {"top", "bottom", else == "top"}.
    :returns: Prints or saves the 'args' in an excel-compatible way.
    """
    columns = []
    s = ""
    for arg in args:
        arg = np.asarray(arg)
        if len(arg.shape) not in [1, 2]:
            raise ValueError("All arguments must be either 1- or 2-dimensional iterables.")

        if len(arg.shape) == 2:
            for c in arg.T:
                columns.append(list(c))
                s += "{}" + delimiter
        else:
            columns.append(list(arg))
            s += "{}" + delimiter

    if header != "":
        header += "\n"

    c_sizes = [len(c) for c in columns]
    c_max = max(c_sizes)
    rows = []
    if align == "bottom":
        d = -1
        for i in range(c_max):
            rows.append(
                s[: -len(delimiter)].format(*["" if i >= n else c[n - i - 1] for c, n in zip(columns, c_sizes)])
            )
    else:
        d = 1
        for i in range(c_max):
            rows.append(s[: -len(delimiter)].format(*["" if i >= n else c[i] for c, n in zip(columns, c_sizes)]))

    ret = header + "\n".join(rows[::d])
    if save is None:
        print(ret)
    else:
        with open(save, mode="w") as file:
            file.write(ret)

    return ret


def add_nested_key(a: dict, key_list: list, val: array_like) -> dict:
    """
    :param a: The target dictionary.
    :param key_list: A list of nested keys of a.
    :param val: The value that is assigned to the last key in key_list.
    :returns: The dictionary a with val assigned to the last key in key_list.
    """
    if not isinstance(a, dict):
        raise TypeError(f"Cannot assign another key ({key_list[0]}) since `a` is not a dictionary.")

    if len(key_list) == 1:
        a[key_list[0]] = val
    else:
        if key_list[0] not in a:
            a[key_list[0]] = add_nested_key({}, key_list[1:], val)
        else:
            a[key_list[0]] = add_nested_key(a[key_list[0]], key_list[1:], val)

    return a


def merge_dicts(a: dict, b: dict, path: list | None = None) -> dict:
    """
    :param a: The first `dict`.
    :param b: The second `dict`.
    :param path: _.
    :returns: `b` merged into `a`.
    """
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                a[key] = merge_dicts(a[key], b[key], [*path, str(key)])
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a


def convolve_dict(a: dict, *key_lists: Iterable, operator: str = "+", short_keys: bool = True) -> dict:
    """
    :param a: A dictionary to be convolved.
    :param key_lists: An arbitrary number of Iterables of the same length which include keys from 'a'.
    :param operator: The operator which is used to combine the key_lists.
    :param short_keys: Whether to just use the keys of the first key_list
     or representations of the entire operations as the new keys.
    :returns: A dictionary of the shape {key_lists[0] + key_lists[1] + ... :
                                         a[key_lists[0]] + a[key_lists[1]] + ... } if operator == '+'.
    """
    operators = ["+", "-", "*", "/", "%", "gauss"]
    if operator not in operators:
        raise ValueError(f"Operator '{operator}' not available try one of {operators}.")

    if operator.lower() == "gauss":
        return {
            keys[0] if short_keys else "Gauss(" + ", ".join([str(k) for k in keys]) + ")": eval(
                "np.sqrt(" + " + ".join([f"a[{k}] ** 2" for k in keys]) + ")", {"a": a, "np": np}
            )
            for keys in zip(*key_lists)
        }

    return {
        keys[0] if short_keys else f" {operator} ".join([str(k) for k in keys]): eval(
            f" {operator} ".join([f"a[{k}]" for k in keys]), {"a": a}
        )
        for keys in zip(*key_lists)
    }


def combine_dicts(dicts: list, key_lists: Iterable, operator: str = "+", short_keys: bool = True) -> dict:
    """
    :param dicts: A list of dictionaries to be combined.
    :param key_lists: A list of Iterables of the same lengths which include keys from 'a'.
     Must have the same length as 'dicts'.
    :param operator: The operator which is used to combine the key_lists.
    :param short_keys: Whether to just use the keys of the first key_list
     or representations of the entire operations as the new keys.
    :returns: A dictionary of the shape {key_lists[0] + key_lists[1] + ... :
                                         dicts[0][key_lists[0]] + dicts[1][key_lists[1]] + ... } if operator == '+'.
    """
    operators = ["+", "-", "*", "/", "%", "gauss"]
    if operator not in operators:
        raise ValueError(f"Operator '{operator}' not available try one of {operators}.")

    if operator.lower() == "gauss":
        return {
            keys[0] if short_keys else "Gauss(" + ", ".join([str(k) for k in keys]) + ")": eval(
                "np.sqrt(" + " + ".join([f"dicts[{i}][{k}] ** 2" for i, k in enumerate(keys)]) + ")",
                {"dicts": dicts, "np": np},
            )
            for keys in zip(*key_lists)
        }

    return {
        keys[0] if short_keys else f" {operator} ".join([str(k) for k in keys]): eval(
            f" {operator} ".join([f"dicts[{i}][{k}]" for i, k in enumerate(keys)]), {"dicts": dicts}
        )
        for keys in zip(*key_lists)
    }


def merge_intervals(intervals: Iterable[Iterable[scalar_like]]) -> ndarray:
    """
    :param intervals: An iterable of intervals.
     An interval i is itself an iterable of two scalar values. If `i[1] < i[0]`, the interval is reversed.
    :returns: An iterable of non-overlapping intervals.
    """
    inter = np.asarray(intervals)
    check_shape_like(inter, np.zeros((1, 2)))
    inter = np.array([i if i[0] <= i[1] else i[::-1] for i in inter])
    sort = np.argsort(inter[:, 0])
    inter = inter[sort, :]

    new_inter = []
    n = 0
    m = inter.shape[0]
    while n < inter.shape[0] - 1:
        end = inter[n, 1]
        for m, i in enumerate(inter[(n + 1) :, :]):
            if not end < i[0]:
                end = np.max([end, i[1]])
            else:
                break
        new_inter.append([inter[n, 0], end])
        n += m + 1

    if inter.shape[0] == 1 or inter[-1, 0] > inter[-2, 1]:
        new_inter.append(inter[-1, :])

    return np.asarray(new_inter)


def get_subarray(a: ndarray, i: int_like | slice, axis: int) -> ndarray:
    """
    :param a: The array.
    :param i: The indexes to select along the 'axis'
    :param axis: The axis along which the indexes 'i' are selected.
    :returns: The subarray selected with the indexes 'i' along the 'axis'.
    """

    axis += a.ndim if axis < 0 else 0
    index = (slice(None),) * axis + (i,) + (slice(None),) * (a.ndim - axis - 1)
    return a[index]


""" Vector math """


def absolute(x: array_like, axis: int = -1) -> ndarray:
    """
    :param x: A real vector or an array of real vectors.
    :param axis: The axis along which the vector components are aligned.
    :returns: The length(s) of the vector(s) x.
    """
    x = np.asarray(x)
    return np.sqrt(np.sum(x**2, axis=axis))


def absolute_complex(x: array_like, axis: int = -1) -> ndarray:
    """
    :param x: A complex vector or an array of complex vectors.
    :param axis: The axis along which the vector components are aligned.
    :returns: The length(s) of the vector(s) x.
    """
    x = np.asarray(x)
    return np.sqrt(np.sum(np.abs(x) ** 2, axis=axis))


def angle(x: array_like, y: array_like, axis: int = -1) -> ndarray:
    """
    :param x: The first vectors (arb. units).
    :param y: The second vectors ([x]).
    :param axis: The axis along which the vector components are aligned.
    :returns: The angle between two vectors x and y (rad).
    :raises ValueError: The shapes of x and y must be compatible.
    """
    x, y = asarray(x, y)
    check_shape_like(x, y)
    return np.arccos(np.sum(x * y, axis=axis) / np.sqrt(np.sum(x**2, axis=axis) * np.sum(y**2, axis=axis)))


def angle_d(x: array_like, x_d: array_like, y: array_like, y_d: array_like, axis: int = -1) -> ndarray:
    """
    :param x: The first vectors (arb. units).
    :param x_d: The uncertainties of the first vectors ([x]).
    :param y: The second vectors ([x]).
    :param y_d: The uncertainties of the second vectors ([x]).
    :param axis: The axis along which the vector components are aligned.
    :returns: The angle between two vectors x and y (rad).
    :raises ValueError: The shapes of x and y must be compatible.
    """
    x, y = asarray(x, y)
    z = np.sum(x * y, axis=axis)
    n = np.sum(x**2, axis=axis) * np.sum(y**2, axis=axis)
    arg = z / np.sqrt(n)
    dx = np.sum(((n * y - z * x * np.sum(y**2, axis=axis)) / n**1.5 * x_d) ** 2, axis=axis)
    dy = np.sum(((n * x - z * y * np.sum(x**2, axis=axis)) / n**1.5 * y_d) ** 2, axis=axis)
    return np.sqrt((dx + dy) / (1 - arg**2))


def transform(t: array_like, vec: array_like, axis: int = -1) -> ndarray:
    """
    :param t: The transformation matrix which must hold t.shape[axis+1] == vec.shape[axis].
    :param vec: The vector to be transformed.
    :param axis: The axis along which the vector components of 'vec' are aligned and that is summed over.
    :returns: The transformed vector vec_new = t * vec.
    """
    t, vec = np.asarray(t), np.asarray(vec)
    ax = axis if axis != -1 else len(vec.shape) - 1
    # check_dimension(vec.shape[ax], ax + 1, t)
    vec = np.expand_dims(vec, ax)
    check_shape_like(t, vec)
    return np.sum(t * vec, axis=ax + 1)


def unit_vector(indexes: array_like, dim: int_like, axis: int_like = -1, dtype: type = float) -> ndarray:
    """
    :param indexes: The indexes of the vectors that are filled with a 1.
    :param dim: The dimension of the unit vectors.
    :param axis: The axis with the dimension 'dim'.
    :param dtype: The data type of the unit vectors.
    :returns: Unit vectors in the specified 'indexes' with dimension 'dim' along the specified 'axis'.
    """
    index = np.asarray(indexes, dtype=int)
    return np.moveaxis(np.choose(np.expand_dims(index, axis=-1), np.identity(int(dim), dtype=dtype)), -1, axis)


def vector_to_diag_matrix(a: array_like, axis: int = -1) -> ndarray:
    """
    :param a: The vector to be transformed into a diagonal matrix.
    :param axis: The axis of the vector elements to be aligned along the diagonal of the matrix.
    :returns: A diagonal matrix with one extra axis compared with 'a' at 'axis' + 1.
    """
    a = np.asarray(a)

    if axis < 0:
        axis += len(a.shape)

    shape = tuple(x1 if i > axis else x0 for i, (x0, x1) in enumerate(zip((*a.shape, 1), (1, *a.shape))))
    b = np.broadcast_to(np.expand_dims(a, axis=axis + 1), shape)

    ident = np.identity(shape[axis], dtype=a.dtype)
    if axis > 0:
        ident = np.expand_dims(ident, axis=tuple(range(axis)))
    if len(a.shape) - 1 > axis:
        ident = np.expand_dims(ident, axis=tuple(range(axis + 2, len(b.shape))))

    return b * ident


def e_r(theta: array_like, phi: array_like, axis: int = -1) -> ndarray:
    """
    :param theta: The angle theta.
    :param phi: The angle phi.
    :param axis: The axis along which the vector components are aligned.
    :returns: The unit vector 'e_r'. Part of an orthonormal system defined by:
        e_x = e_r(pi/2, .)
        e_y = e_r(0, pi/2)
        e_z = e_r(0, 0)
    """
    x = np.expand_dims(np.sin(theta), axis=axis)
    y = np.expand_dims(np.cos(theta) * np.sin(phi), axis=axis)
    z = np.expand_dims(np.cos(theta) * np.cos(phi), axis=axis)
    return np.concatenate([x, y, z], axis=axis)


def e_theta(theta: array_like, phi: array_like, axis: int = -1) -> ndarray:
    """
    :param theta: The angle theta.
    :param phi: The angle phi.
    :param axis: The axis along which the vector components are aligned.
    :returns: The unit vector 'e_theta'. Part of an orthonormal system defined by:
        e_x = e_r(pi/2, .)
        e_y = e_r(0, pi/2)
        e_z = e_r(0, 0)
    """
    x = np.expand_dims(np.cos(theta) * np.ones_like(phi), axis=axis)
    y = np.expand_dims(-np.sin(theta) * np.sin(phi), axis=axis)
    z = np.expand_dims(-np.sin(theta) * np.cos(phi), axis=axis)
    return np.concatenate([x, y, z], axis=axis)


def e_phi(theta: array_like, phi: array_like, axis: int = -1) -> ndarray:
    """
    :param theta: The angle theta.
    :param phi: The angle phi.
    :param axis: The axis along which the vector components are aligned.
    :returns: The unit vector 'e_phi'. Part of an orthonormal system defined by:
        e_x = e_r(pi/2, .)
        e_y = e_r(0, pi/2)
        e_z = e_r(0, 0)
    """
    x = np.expand_dims(np.zeros_like(phi) * np.ones_like(theta), axis=axis)
    y = np.expand_dims(np.cos(phi) * np.ones_like(theta), axis=axis)
    z = np.expand_dims(-np.sin(phi) * np.ones_like(theta), axis=axis)
    return np.concatenate([x, y, z], axis=axis)


def orthonormal_rtp(r: array_like, axis: int = -1) -> tuple[ndarray, ndarray, ndarray]:
    """
    :param r: The first vector of the orthonormal system. Does not have to be normalized.
    :param axis: The axis along which the vector components are aligned.
    :returns: The three orthonormal vectors 'e_r', 'e_theta' and 'e_phi'
     given the vector 'r' pointing in the 'e_r' direction.
    """
    r = np.asarray(r)
    check_dimension(3, axis, r)
    r_abs = np.expand_dims(absolute(r, axis=axis), axis=axis)

    e_r0 = r / r_abs
    theta = np.arcsin(np.take(e_r0, 0, axis=axis))
    valid = np.abs(np.abs(theta) - np.pi / 2.0) > 1e-5
    phi = np.full_like(theta, 0.0)

    if valid.size != 0:
        phi[valid] = np.arcsin(np.take(e_r0, 1, axis=axis)[valid] / np.cos(theta[valid]))

    return e_r0, e_theta(theta, phi, axis=axis), e_phi(theta, phi, axis=axis)


def orthonormal(r: array_like, axis: int = -1) -> tuple[ndarray, ndarray, ndarray]:
    """
    :param r: The first vector of the orthonormal system. Does not have to be normalized.
    :param axis: The axis along which the vector components are aligned.
    :returns: Three orthonormal vectors, given the first vector 'r'.
    """
    r1 = np.asarray(r, dtype=float)
    check_dimension(3, axis, r1)
    r1 = np.moveaxis(r1, axis, -1)
    i_seed = np.argmin(np.abs(r1), axis=-1)
    w = unit_vector(i_seed, 3)

    r2 = np.cross(r1, w, axisa=-1, axisb=-1)
    r3 = np.cross(r1, r2, axisa=-1, axisb=-1)
    r1 /= np.expand_dims(absolute(r1, axis=-1), axis=-1)
    r2 /= np.expand_dims(absolute(r2, axis=-1), axis=-1)
    r3 /= np.expand_dims(absolute(r3, axis=-1), axis=-1)
    r1 = np.moveaxis(r1, -1, axis)
    r2 = np.moveaxis(r2, -1, axis)
    r3 = np.moveaxis(r3, -1, axis)
    return r1, r2, r3


def rotation_matrix(alpha: array_like, dr: array_iter) -> ndarray:
    """
    :param alpha: The angle to rotate.
    :param dr: The vector to rotate about.
    :returns: The rotation matrix defined by the angle alpha and the vector dr.
    """
    alpha, dr = np.asarray(alpha), np.asarray(dr)
    dr /= absolute(dr)
    r = np.identity(3) * np.cos(alpha)
    r += np.array([[0.0, -dr[2], dr[1]], [dr[2], 0.0, -dr[0]], [-dr[1], dr[0], 0.0]]) * np.sin(alpha)
    r += np.einsum("i,j->ij", dr, dr) * (1.0 - np.cos(alpha))
    return r


class Rotation:
    """
    An object specifying a rotation in 3d-space. The rotation is defined
    through an angle 'alpha' and a rotational axis 'dr' by the user.
    Additional instance attributes are the angle in degree 'alpha_deg' and the rotational matrix 'R'.
    """

    def __init__(self, alpha: scalar_like = 0.0, dr: array_iter | None = None) -> None:
        """
        :param alpha: The angle of the rotation (rad).
        :param dr: The rotational axis of the rotation.
        """
        self.alpha = alpha
        self.alpha_deg = self.alpha * 180.0 / np.pi

        if dr is None:
            dr = np.array([0.0, 0.0, 1.0])
        self.dr = np.asarray(dr, dtype=float)
        check_shape((3,), self.dr, allow_scalar=False)
        self.dr /= absolute(self.dr)

        self.R = rotation_matrix(self.alpha, self.dr)


def rotation_to_vector(x: array_iter, y: array_iter) -> Rotation:
    """
    :param x: The first vector.
    :param y: The second vector.
    :returns: A Rotation object which rotates the first vector onto the second.
    """
    _x, _y = asarray(x, y)
    check_shape((3,), _x, _y, allow_scalar=False)
    _x /= absolute(_x)
    _y /= absolute(_y)

    diff = _y - _x
    if np.all(diff == 0.0):
        return Rotation()

    if np.all(diff == 2.0 * _y):
        _x, dr, _ = orthonormal(_x)
        return Rotation(alpha=np.pi, dr=dr)

    alpha = angle(_x, _y)
    dr = np.cross(_x, _y)

    return Rotation(alpha=alpha, dr=dr)


""" External program helpers """


def import_iso_shifts_tilda(db: str, iso_shifts: dict) -> None:
    """
    :param db: Location of a Tilda-compatible database.
    :param iso_shifts: A dictionary of isotope shifts
     with the structure {iso_str: {line_str: [val, stat_err, syst_err]}}.
    :returns: Writes the entries of the given dict to the specified Tilda-compatible database.
    """
    con = sqlite3.connect(db)
    cur = con.cursor()
    for iso, line_dict in iso_shifts.items():
        for line, val in line_dict.items():
            print(val)
            cur.execute("INSERT OR REPLACE INTO Combined (iso, parname, run) VALUES (?, ?, ?)", (iso, "shift", line))
            con.commit()
            cur.execute(
                "UPDATE Combined SET config=?, final=?, "
                "val = ?, statErr = ?, statErrForm=?, systErr = ?, systErrForm=? "
                "WHERE iso = ? AND parname = ? AND run = ?",
                ("[]", 0, val[0], val[1], "err", val[2] if len(val) > 2 else 0.0, 0, iso, "shift", line),
            )
            con.commit()
    con.close()
