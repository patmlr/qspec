"""
qspec.models._base
==================

Base classes for lineshape models.
"""

import numpy as np

from qspec.qtypes import (
    Any,
    CodeType,
    Iterable,
    array_like,
    asarray,
    fix_type,
    floating,
    has_shape,
    int_like,
    is_scalar,
    ndarray,
    scalar,
    scalar_nd,
)
from qspec.tools import merge_intervals

__all__ = [
    "MODEL_IS_NONE_ERROR",
    "Amplifier",
    "Custom",
    "Empty",
    "Linked",
    "Listed",
    "Model",
    "NPeak",
    "Offset",
    "Summed",
    "YPars",
]


MODEL_IS_NONE_ERROR = "Attribute `{}.model` must be a valid `Model`"
CONVOLVE_IS_NONE_ERROR = "Either attribute `Convolved.model` or `Convolved.model_1` must be a valid `Model`"


np_version = np.version.version.split(".")


def _is_unc(fix: fix_type) -> bool:
    if isinstance(fix, float):
        return True

    if not isinstance(fix, str):
        return False

    j, k = fix.find("("), fix.find(")")
    try:
        if j == -1 or k == -1:
            raise ValueError(f"{fix} is not a value with uncertainty.")
        _ = f"{float(fix[:j])}({float(fix[j + 1 : k])})"
    except ValueError:
        return False

    return True


def _val_fix_to_val(val: scalar, fix: fix_type) -> float:
    if isinstance(fix, str):
        j = fix.find("(")
        return float(fix[:j])

    if is_scalar(fix):
        return float(val)

    raise TypeError("`fix` must be `str` or `scalar`.")


def _fix_to_unc(fix: fix_type) -> float:
    if isinstance(fix, str):
        j, k = fix.find("("), fix.find(")")
        return float(fix[j + 1 : k])

    if is_scalar(fix):
        return float(fix)

    raise TypeError("`fix` must be `str` or `scalar`.")


def _args_ordered(args: scalar_nd, order: Iterable[int_like]) -> list[ndarray[floating]]:
    return [np.asarray(args[int(i)], dtype=float) for i in order]


def _poly(x: ndarray, *args: scalar) -> ndarray:
    return np.sum([args[n] * x**n for n in range(len(args))], axis=0)


class Model:
    def __init__(self, model: "Model | None" = None) -> None:
        """
        Base class for all models.

        :param model: A submodel whose parameters are adopted by this model.
        """
        self.model = model
        self.type = "Model"

    def __call__(self, x: array_like, *args: scalar, **kwargs: Any) -> ndarray:
        x = np.asarray(x, dtype=float)
        return self.evaluate(x, *self.update_args(args), **kwargs)

    def evaluate(self, x: ndarray, *args: scalar, **kwargs: Any) -> ndarray:
        """
        The main function of the model. This is executed when the model is called.
        Reimplement this function in subclasses.

        :param x: The input values $x$.
        :param args: The function parameters. Must have length `Model.size`.
        :param kwargs: Additional keyword arguments.
        :returns: The function values $y$ at the input values `x`.
        """
        return np.zeros((), dtype=float)

    def _add_arg(self, name: str, val: scalar, fix: fix_type, link: bool) -> None:
        """
        Add a new parameter to the model.

        :param name: The parameter name.
        :param val: The parameter value.
        :param fix: Whether the parameter is fixed.
        :param link: Whether the parameter is linked.
        """
        if name in self.names:
            raise ValueError(f"Parameter {name} already exists.")

        self.names.append(name)
        self.vals.append(float(val))
        self.fixes.append(fix)
        self.links.append(link)
        self.expressions.append(f"args[{self._index}]")

        self.p[name] = self._index

        self._index += 1
        self._size += 1

    @property
    def description(self) -> str:
        """
        A description of the model hierarchy.

        :returns: A `str` representing the model hierarchy.
        """
        label = ""
        super_model = self
        while super_model is not None:
            if isinstance(super_model, Listed):
                label += super_model.type + "[0]."
                super_model = super_model.models[0]
            else:
                label += super_model.type + "."
                super_model = super_model.model
        return label[:-1]

    @property
    def size(self) -> int:
        """
        The number of parameters required by the model.
        """
        return self._size

    @property
    def dx(self) -> float:
        """
        A hint for an x-axis step size for a smooth display of the model.
        """
        return 0.1 if self.model is None else self.model.dx

    @property
    def error(self) -> str:
        """
        An error message if there is an issue with the model parameters (not implemented).
        """
        return self._error

    @property
    def model(self) -> "Model | None":
        """
        The submodel.
        """
        return self._model

    @model.setter
    def model(self, value: "Model | None") -> None:
        self._model = value
        if self._model is None:
            self.names: list[str] = []
            self.vals: list[float] = []
            self.fixes: list[fix_type] = []
            self.links: list[bool] = []
            self.expressions = []
            self.p: dict[str, int] = {}
            self._index = 0
            self._size = 0
            self._error = ""
        else:
            self.names, self.vals, self.fixes, self.links = (
                self._model.names,
                self._model.vals,
                self._model.fixes,
                self._model.links,
            )
            self.expressions = self._model.expressions
            self.p = self._model.p
            self._index = len(self._model.names)
            self._size = len(self._model.names)
            self._error = self._model.error

    def get_pars(self) -> zip:
        """
        :returns: A zip-iterator over `(names, vals, fixes, links)`.
        """
        return zip(self.names, self.vals, self.fixes, self.links)

    def set_pars(self, pars: Iterable[tuple[scalar, fix_type, bool]], force: bool = False) -> None:
        """
        Set all `vals`, `fixes` and `links` with one nested list.

        :param pars: An `Iterable` of size-3 tuples. Equivalent to shape `(self.size, 3)`.
        :param force: The `force` parameter of the `set_val`, `set_fix` and `set_link` functions.
        """
        for i, p in enumerate(pars):
            self.set_val(i, p[0], force=force)
            self.set_fix(i, p[1], force=force)
            self.set_link(i, p[2], force=force)

    def set_vals(self, vals: Iterable[scalar], force: bool = False) -> None:
        """
        Set all `vals` with one list.

        :param vals: A list of shape (`self.size`, ).
        :param force: The `force` parameter of the `set_val` function.
        """
        for i, val in enumerate(vals):
            self.set_val(i, val, force=force)

    def set_fixes(self, fixes: Iterable[fix_type], force: bool = False) -> None:
        """
        Set all `fixes` with one list.

        :param fixes: A list of shape (`self.size`, ).
        :param force: The `force` parameter of the `set_fix` function.
        :returns:
        """
        for i, fix in enumerate(fixes):
            self.set_fix(i, fix, force=force)

    def set_links(self, links: Iterable[bool], force: bool = False) -> None:
        """
        Set all `links` with one `list`.

        :param links: A list of shape (`self.size`, ).
        :param force: The `force` parameter of the `set_link` function.
        :returns:
        """
        for i, link in enumerate(links):
            self.set_link(i, link, force=force)

    def set_val(self, i: int_like | str, val: scalar, force: bool = False) -> None:
        """
        Set a specific parameter value.

        :param i: The index (`int`) or the name (`str`) of the parameter.
        :param val: The new parameter value.
        :param force: Force the parameter to take exactly the new value.
         If `False`, the parameter is converted to the correct format.
        :returns:
        """
        if isinstance(i, str):
            i = self.p[i]

        if force or is_scalar(val):
            if self.model is None:
                self.vals[i] = float(val)
            else:
                self.model.set_val(i, val, force=False)  # Set val for all submodels
                # to ensure set_val is called for a ListedModel if one is part of the submodels.
                # This is only needed for the vals since these may be predicted by the specific model.
                # Everything else is handled by the top-model.
        else:
            raise ValueError(f"The parameter value {val} has the wrong format. Must be a floating.")

    def set_fix(self, i: int_like | str, fix: fix_type, force: bool = False) -> None:
        """
        Set a specific parameter fix state.

        :param i: The index (`int`) or the name (`str`) of the parameter.
        :param fix: The new parameter value.
        :param force: Force the parameter to take exactly the new fix state.
         If `False`, the parameter is converted to the correct format.
        :returns:
        """
        if isinstance(i, str):
            i = self.p[i]
        i = int(i)

        if force:
            self.fixes[i] = fix
            return

        if is_scalar(fix):
            if isinstance(fix, bool):
                fix = bool(fix)
            elif fix <= 0 or np.isinf(fix):
                fix = bool(fix <= 0)
            else:
                fix = float(fix)
            expr = f"args[{i}]"

        elif isinstance(fix, str):
            j, k = fix.find("("), fix.find(")")

            try:
                if j == -1 or k == -1:
                    raise ValueError("Not a value with uncertainty.")
                fix = f"{float(fix[:j])}({float(fix[j + 1 : k])})"
                expr = f"args[{i}]"

            except ValueError:
                _fix = fix
                for j, name in enumerate(self.names):
                    _fix = _fix.replace(name, f"eval(self.expressions[{j}])")
                expr = _fix

                try:
                    self._eval_zero_division(self.vals, expr)
                except (ValueError, TypeError, SyntaxError, NameError) as e:
                    print(f"Invalid expression for parameter '{self.names[i]}': {fix}. Got a {e!r}.")
                    return

        elif isinstance(fix, (tuple, list, np.ndarray)):
            if len(fix) == 0:
                fix = [0, 1]
            elif len(fix) == 1:
                fix = [0, fix[0]]
            else:
                fix = list(fix[:2])
            expr = f"args[{i}]"

        else:
            raise ValueError(
                f"The parameter fix state {fix} has the wrong format. Must be a scalar, bool, str or iterable."
            )

        temp_expr = self.expressions[i]
        temp_fix = self.fixes[i]
        self.expressions[i] = compile(expr, "<string>", "eval", optimize=2)  # Compile beforehand to save time.
        self.fixes[i] = fix
        try:
            self.update_args(self.vals)
        except RecursionError as e:
            print(f"Expression for {self.names[i]} with fix {fix} form a loop. Got a {e!r}.")
            self.expressions[i] = temp_expr
            self.fixes[i] = temp_fix

    def set_link(self, i: int_like | str, link: bool, force: bool = False) -> None:
        """
        Set a specific parameter link state.

        :param i: The index (`int`) or the name (`str`) of the parameter.
        :param link: The new parameter link state.
        :param force: Force the parameter to take exactly the new link state.
         If `False`, the parameter is converted to the correct format.
        :returns:
        """
        if isinstance(i, str):
            i = self.p[i]

        if force:
            self.links[i] = link
            return

        if is_scalar(link):
            self.links[i] = bool(link)
        else:
            raise ValueError(f"The parameter link state {link} has the wrong format. Must be a scalar or bool.")

    def _eval_zero_division(self, args: scalar_nd, expr: str | CodeType) -> ndarray[floating]:
        """
        Safely calculate parameter expressions including `nan`, `inf` and zero divisions.

        :param args: The parameters.
        :param expr: The parameter expression.
        :returns: The processed parameter or 0 in case of a `nan`, `inf` or zero division.
        """
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                ret = eval(expr, {}, {"self": self, "args": args})

            if isinstance(ret, np.ndarray):
                ret[np.isnan(ret) + np.isinf(ret)] = 0.0
                return ret

            return np.array(0.0) if np.isnan(ret) or np.isinf(ret) else ret

        except ZeroDivisionError:
            return np.zeros_like(args[0], dtype=float)

    def update_args(self, args: scalar_nd) -> tuple[ndarray[floating], ...]:
        """
        :param args: The parameters.
        :returns: The parameters updated with the parameter expressions.
        """
        return tuple(self._eval_zero_division(args, expr) for expr in self.expressions)

    def update(self) -> None:
        """
        Updates `self.vals` with updated parameters, see self.update_args.

        :returns:
        """
        self.set_vals(self.update_args(self.vals), force=False)

    def min(self) -> float:
        """
        :returns: A hint for an x-axis minimum for a complete display of the model.
        """
        return -1.0 if self.model is None else self.model.min()

    def max(self) -> float:
        """
        :returns: A hint for an x-axis maximum for a complete display of the model.
        """
        return 1.0 if self.model is None else self.model.max()

    def intervals(self) -> list[list[float]]:
        """
        :returns: A list of x-axis intervals for a complete display of the model.
        """
        return [[self.min(), self.max()]] if self.model is None else self.model.intervals()

    def x(self) -> ndarray:
        """
        :returns: An array of x values for a complete and smooth display of the model.
        """
        return np.concatenate([np.arange(i[0], i[1], self.dx, dtype=float) for i in self.intervals()], axis=0)

    def fit_prepare(self) -> tuple[list[fix_type], tuple[float | list[float], float | list[float]]]:
        """
        :returns: (fixed, bounds) A list of `bool` values specifying which parameters are not varied in a fit
         and a list of bounds for the fit parameters. See parameters `p0_fixed` and `bounds` of `qspec.curve_fit`.
        """
        bounds: tuple[float | list[float], float | list[float]] = (-np.inf, np.inf)
        fixed: list[fix_type] = [fix for fix in self.fixes]
        b_lower: list[float] = []
        b_upper: list[float] = []

        _bounds = False
        for i, fix in enumerate(self.fixes):
            if isinstance(fix, bool):
                b_lower.append(-np.inf)
                b_upper.append(np.inf)
            elif _is_unc(fix):
                b_lower.append(-np.inf)
                b_upper.append(np.inf)
                fixed[i] = False
            elif isinstance(fix, list):
                _bounds = True
                b_lower.append(fix[0])
                b_upper.append(fix[1])
                fixed[i] = False
            elif isinstance(fix, str):
                b_lower.append(-np.inf)
                b_upper.append(np.inf)
                fixed[i] = True
            else:
                raise TypeError(f"The type {type(fix)} of the element {fix} in self.fixes is not supported.")

        if _bounds:
            bounds = (b_lower, b_upper)
        return fixed, bounds


class Empty(Model):
    def __init__(self) -> None:
        """
        An empty model, returning `numpy.zeros_like(x)`
        """
        super().__init__(model=None)
        self.type = "Empty"

    def evaluate(self, x: ndarray, *args: scalar, **kwargs: Any) -> ndarray:
        """
        An empty model, returning `numpy.zeros_like(x)`.

        :param x: The input values $x$.
        :param args: [].
        :param kwargs: Additional keyword arguments.
        :returns: The function values $y$ at the input values `x`.
        """
        return np.zeros_like(x)


class NPeak(Model):
    def __init__(self, model: Model, n_peaks: int_like = 1) -> None:
        """
        Evaluates the given `model` at the positions $x_i$ with intensities $p_i$ and `0 <= i < n_peaks`.

        :param model: A submodel whose parameters are adopted by this model.
        :param n_peaks: The number of times the submodel is copied.
        """
        self.model: Model
        super().__init__(model=model)
        self.type = "NPeak"

        if self.model is None:
            raise ValueError(MODEL_IS_NONE_ERROR.format(type(self)))

        self.n_peaks = int(n_peaks)
        for n in range(self.n_peaks):
            self._add_arg(f"x{n}", 0.0, n == 0, False)
            self._add_arg(f"p{n}", 1.0, n == 0, False)

    def evaluate(self, x: ndarray, *args: scalar, **kwargs: Any) -> ndarray:
        return np.sum(
            [
                args[self.model.size + 2 * n + 1]
                * self.model.evaluate(x - args[self.model.size + 2 * n], *args[: self.model.size])
                for n in range(self.n_peaks)
            ],
            axis=0,
        )

    def min(self) -> float:
        """
        :returns: A hint for an x-axis minimum for a complete display of the model.
        """
        min_center = min(self.vals[self.p[f"x{n}"]] for n in range(self.n_peaks))
        return float(min_center + self.model.min())

    def max(self) -> float:
        """
        :returns: A hint for an x-axis maximum for a complete display of the model.
        """
        max_center = max(self.vals[self.p[f"x{n}"]] for n in range(self.n_peaks))
        return float(max_center + self.model.max())

    def intervals(self) -> list[list[float]]:
        """
        :returns: A list of x-axis intervals for a complete display of the model.
        """
        return merge_intervals(
            [
                [i[0] + self.vals[self.model.size + 2 * n], i[1] + self.vals[self.model.size + 2 * n]]
                for i in self.model.intervals()
                for n in range(self.n_peaks)
            ]
        ).tolist()


class Offset(Model):
    def __init__(
        self, model: Model | None = None, x_cuts: array_like | None = None, offsets: array_like | None = None
    ) -> None:
        """
        Cuts the x-axis and adds y-axis offsets to every segment.

        :param model: The submodel the offset will be added to. If None, the offset will be added to zero.
        :param x_cuts: x values where to cut the x-axis.
        :param offsets: A list of maximally considered polynomial orders for each slice.
         The list must have length len(x_cuts) + 1.
        """
        super().__init__(model=model)
        self.type = "Offset"

        if x_cuts is None:
            x_cuts = []

        x_cuts = np.asarray(x_cuts, dtype=float).flatten()
        self.x_cuts: list[float] = sorted(x_cuts.tolist())

        if offsets is None:
            offsets = [0]

        offsets = np.asarray(offsets, dtype=int).flatten()
        self.offsets: list[int] = list(int(o) for o in offsets)

        if len(self.offsets) != len(self.x_cuts) + 1:
            raise ValueError(
                "The parameter `offset` must be a list of size `len(x_cuts) + 1`"
                " and contain the maximally considered polynomial order for each slice."
            )

        self.offset_map = []
        self.offset_masks = []
        self.update_on_call = True

        self.gen_offset_map()

    def evaluate(self, x: ndarray, *args: scalar, **kwargs: Any) -> ndarray:
        if self.model is None:
            return self._offset(x, *args)

        return self.model.evaluate(x, *args[: self.model.size]) + self._offset(x, *args)

    def set_x_cuts(self, x_cuts: array_like | None) -> None:
        """
        Set the values where to cut the x-axis into intervals with individual offset parameters.

        :param x_cuts: A list of x values where to cut the x-axis.
        """
        if x_cuts is None:
            x_cuts = []
        x_cuts = np.asarray(x_cuts, dtype=float)
        self.x_cuts = sorted(x_cuts.tolist())

        if len(x_cuts) != len(self.x_cuts):
            raise ValueError("`x_cuts` must not change its size.")
        self.x_cuts = sorted(list(x_cuts))

    def _offset(self, x: ndarray, *args: scalar) -> ndarray:
        """
        :param x: The input values.
        :param args: The function parameters.
        :returns: The offset polynomial.
        """
        if self.update_on_call:
            self.gen_offset_masks(x)

        if has_shape(args[0]) and len(args[0].shape) > 1:
            ret = np.zeros((x.shape[0], args[0].shape[1]), dtype=float)

            for i, mask in enumerate(self.offset_masks):
                _mask = np.broadcast_to(mask, ret.shape)
                ret[_mask] = _poly(x[mask][:, None], *_args_ordered(args, self.offset_map[i])).flatten()

        else:
            ret = np.zeros_like(x)
            for i, mask in enumerate(self.offset_masks):
                ret[mask] = _poly(x[mask], *_args_ordered(args, self.offset_map[i]))

        return ret

    def gen_offset_map(self) -> None:
        """
        Generate the offset parameters and a map of the interval and polynomial order to the parameter index space.

        :returns:
        """
        self.offset_map = []
        for i, n in enumerate(self.offsets):
            self.offset_map.append([])
            for k in range(n + 1):
                self.offset_map[-1].append(self._index)
                self._add_arg(f"off{i}e{k}", 0.0, False, False)

    """ Preprocessing """

    def gen_offset_masks(self, x: array_like) -> None:
        """
        Generate the array masks corresponding to the `x_cuts`.

        :param x: The input values.
        :returns:
        """
        x = np.asarray(x, dtype=float)
        self.offset_masks = []
        for x0, x1 in zip([np.min(x) - 1.0, *self.x_cuts], [*self.x_cuts, np.max(x) + 1.0]):
            x_mean = 0.5 * (x0 + x1)
            self.offset_masks.append(np.abs(x - x_mean) < x1 - x_mean)

    def guess_offset(self, x: array_like, y: array_like) -> None:
        """
        Guess the first two polynomial orders for a given data set (const and linear).

        :param x: The input values.
        :param y: The y data.
        :returns:
        """
        x, y = asarray(x, y, dtype=float)

        for i, mask in enumerate(self.offset_masks):
            self.vals[self.p[f"off{i}e0"]] = 0.5 * (y[mask][0] + y[mask][-1])

            try:
                self.vals[self.p[f"off{i}e1"]] = (y[mask][-1] - y[mask][0]) / (x[mask][-1] - x[mask][0])
            except KeyError:
                return


class Amplifier(Model):
    def __init__(self, order: int_like | None = None) -> None:
        """
        A polynomial of order `order`.

        :param order: The maximum considered order of the polynomial.
        """
        super().__init__(model=None)
        self.type = "Amplifier"

        if order is None:
            order = 1
        self.order = int(order)

        for n in range(self.order + 1):
            self._add_arg(f"a{n}", 1.0 if n == 1 else 0.0, False, False)
        self._min = -10
        self._max = 10

    def evaluate(self, x: ndarray, *args: scalar, **kwargs: Any) -> ndarray:
        self._min = np.min(x)
        self._max = np.max(x)
        return _poly(x, *args)

    @property
    def dx(self) -> float:
        """
        A hint for an x-axis step size for a smooth display of the model.
        """
        return 1e-2

    def min(self) -> float:
        """
        :returns: A hint for an x-axis minimum for a complete display of the model.
        """
        return self._min

    def max(self) -> float:
        """
        :returns: A hint for an x-axis maximum for a complete display of the model.
        """
        return self._max


class Custom(Model):
    def __init__(self, model: Model | None = None, parameters: Iterable[str] | None = None) -> None:
        """
        A model with custom parameters. Without a submodel, Custom returns the user-specified parameters as an array
        regardless of the input `x`. Otherwise, the submodel is called and the custom parameters
        can be connected to other parameters by the user.

        :param model: A submodel whose parameters are adopted by this model.
        :param parameters: A list of str, representing the names of the custom parameters.
        """
        super().__init__(model=model)
        self.type = "Custom"
        if parameters is None:
            parameters = []
        self.parameters: list[str] = list(parameters)

        for p in self.parameters:
            self._add_arg(p, 0.0, False, False)

    def evaluate(self, x: ndarray, *args: scalar, **kwargs: Any) -> ndarray:
        if self.model is None:
            return np.array(args, dtype=float)
        return self.model.evaluate(x, *args[: self.model.size], **kwargs)


class YPars(Model):
    def __init__(self, model: Model) -> None:
        """
        Concatenates the *Prior* parameters of the submodel, that have uncertainties as `fix` states,
        with the y-axis array resulting from calling the submodel. This is used internally in `qspec.models.fit`.

        :param model: A submodel whose parameters are adopted by this model.
        """
        self.model: Model
        super().__init__(model=model)
        self.type = "YPars"

        if self.model is None:
            raise ValueError(MODEL_IS_NONE_ERROR.format(type(self)))

        self.p_y = [i for i, fix in enumerate(self.model.fixes) if _is_unc(fix)]

    def evaluate(self, x: ndarray, *args: scalar, **kwargs: Any) -> ndarray:
        if self.model is None:
            raise ValueError(MODEL_IS_NONE_ERROR.format(type(self)))

        return np.concatenate(
            [self.model.evaluate(x, *args, **kwargs), np.array([args[p_y] for p_y in self.p_y], dtype=float)], axis=0
        )


class Listed(Model):
    def __init__(self, models: Iterable[Model], labels: Iterable[str] | None = None) -> None:
        """
        An abstract class for models with multiple submodels.

        :param models: A list of submodels whose parameters are adopted by this model.
        :param labels: A list of labels with the same length as `models`.
         The labels are appended to the parameter names of each submodel.
        """
        super().__init__(model=None)
        self.type = "Listed"

        self.models = list(models)

        if labels is None:
            labels = [""] if len(self.models) == 1 else [f"__{i}" for i in range(len(self.models))]
        self.labels = list(labels)

        self.slices = []
        self.model_map = []
        self.index_map = []

        for i, (model, label) in enumerate(zip(self.models, self.labels)):
            self.slices.append(slice(self._index, self._index + model.size, 1))

            for j, (name, val, fix, link) in enumerate(model.get_pars()):
                self.model_map.append(i)
                self.index_map.append(j)

                if isinstance(fix, str) and not _is_unc(fix):
                    for _name in model.names:
                        fix = fix.replace(_name, f"{_name}{label}")

                self._add_arg(f"{name}{label}", val, fix, link)

        self.set_fixes(list(self.fixes))

    def set_val(self, i: int_like | str, val: scalar, force: bool = False) -> None:
        """
        Set a specific parameter value.

        :param i: The index (`int`) or the name (`str`) of the parameter.
        :param val: The new parameter value.
        :param force: Force the parameter to take exactly the new value.
         If `False`, the parameter is converted to the correct format.
        :returns:
        """
        if isinstance(i, str):
            i = self.p[i]

        super().set_val(i, val, force=force)

        if i < len(self.model_map):
            self.models[self.model_map[i]].set_val(self.index_map[i], self.vals[i], force=True)

    def set_fix(self, i: int_like | str, fix: fix_type, force: bool = False) -> None:
        """
        Set a specific parameter fix state. Also sets the fix state for the respective submodel.

        :param i: The index (`int`) or the name (`str`) of the parameter.
        :param fix: The new parameter value.
        :param force: Force the parameter to take exactly the new fix state.
         If `False`, the parameter is converted to the correct format.
        :returns:
        """
        if isinstance(i, str):
            i = self.p[i]

        super().set_fix(i, fix, force=force)

        if i < len(self.model_map):
            self.models[self.model_map[i]].set_fix(self.index_map[i], self.fixes[i], force=True)

    def set_link(self, i: int_like | str, link: bool, force: bool = False) -> None:
        """
        Set a specific parameter link state. Also sets the link state for the respective submodel.

        :param i: The index (`int`) or the name (`str`) of the parameter.
        :param link: The new parameter link state.
        :param force: Force the parameter to take exactly the new link state.
         If `False`, the parameter is converted to the correct format.
        :returns:
        """
        if isinstance(i, str):
            i = self.p[i]

        super().set_link(i, link, force=force)

        if i < len(self.model_map):
            self.models[self.model_map[i]].set_link(self.index_map[i], self.links[i], force=True)

    def inherit_vals(self, force: bool = False) -> None:
        """
        Inherit the parameter values of the submodels.

        :param force: The `force` parameter of `self.set_val`.
        :returns: None
        """
        self.set_vals([val for model in self.models for val in model.vals], force=force)

    def inherit_fixes(self, force: bool = False) -> None:
        """
        Inherit the parameter fixes of the submodels.

        :param force: The `force` parameter of `self.set_fix`.
        :returns: None
        """
        self.set_fixes([fix for model in self.models for fix in model.fixes], force=force)

    def inherit_links(self, force: bool = False) -> None:
        """
        Inherit the parameter links of the submodels.

        :param force: The `force` parameter of `self.set_link`.
        :returns: None
        """
        self.set_links([link for model in self.models for link in model.links], force=force)


class Summed(Listed):
    def __init__(self, models: Iterable[Model], labels: Iterable[str] | None = None) -> None:
        """
        A `Listed` model summing over all submodels with individual `center` and `int` parameters.

        :param models: A list of submodels whose parameters are adopted by this model.
        :param labels: A list of labels with the same length as `models`.
         The labels are appended to the parameter names of each submodel.
        """
        super().__init__(models, labels=labels)
        self.type = "Summed"

        self.indices_add = []

        for n, (model, label) in enumerate(zip(self.models, self.labels)):
            self.indices_add.append([self._index, self._index + 1])

            self._add_arg(f"center{label}", 0.0, False, False)
            self._add_arg(f"int{label}", 1.0, False, False)

    def evaluate(self, x: ndarray, *args: scalar, **kwargs: Any) -> ndarray:
        return np.sum(
            [
                args[i[1]] * model.evaluate(x - args[i[0]], *args[_slice], **kwargs)
                for model, _slice, i in zip(self.models, self.slices, self.indices_add)
            ],
            axis=0,
        )

    @property
    def dx(self) -> float:
        """
        A hint for an x-axis step size for a smooth display of the model.
        """
        return min(model.dx for model in self.models)

    def min(self) -> float:
        """
        :returns: A hint for an x-axis minimum for a complete display of the model.
        """
        return min(model.min() for model in self.models)

    def max(self) -> float:
        """
        :returns: A hint for an x-axis maximum for a complete display of the model.
        """
        return max(model.max() for model in self.models)

    def intervals(self) -> list[list[float]]:
        """
        :returns: A list of x-axis intervals for a complete display of the model.
        """
        return merge_intervals(
            [
                [i[0] + self.vals[j[0]], i[1] + self.vals[j[0]]]
                for model, j in zip(self.models, self.indices_add)
                for i in model.intervals()
            ]
        ).tolist()


class Linked(Listed):
    def __init__(self, models: Iterable[Model]) -> None:
        """
        A `Listed` model linking all `link=True` parameters of the submodels.

        :param models: A list of submodels whose parameters are adopted by this model.
        """
        super().__init__(models, labels=None)
        self.type = "Linked"

        for i, (name, val, fix, link) in enumerate(self.get_pars()):
            if link and (not fix or isinstance(fix, list) or _is_unc(fix)):
                _name = name[: name.rfind("__")]

                for j, model in enumerate(self.models):
                    if j < self.model_map[i] and _name in model.names:
                        self.set_fix(i, f"{_name}__{j}")
                        break

    def evaluate(self, x: ndarray, *args: scalar, **kwargs: Any) -> ndarray:
        return np.concatenate(
            tuple(
                model.evaluate(_x, *args[_slice], **kwargs) for model, _slice, _x in zip(self.models, self.slices, x)
            ),
            axis=0,
        )
