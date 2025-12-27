"""
qspec.models._fit
=================

Module to fit the models to data.
"""

import numpy as np

from qspec.analyze import curve_fit, odr_fit
from qspec.models import _base, _helper
from qspec.models._base import _fix_to_unc, _val_fix_to_val
from qspec.qtypes import (
    Any,
    Callable,
    Generator,
    Iterable,
    TypeGuard,
    array_like,
    floating,
    int_like,
    ndarray,
)
from qspec.tools import asarray_optional, print_colored, print_cov

__all__ = ["ROUTINES", "fit", "reduced_chi2", "residuals", "sigma_poisson"]


ROUTINES = {"curve_fit", "odr_fit"}
SIGMA_FTYPE_SIGNATURE_ERROR = (
    "`sigma_y` is a `Callable` but does not have the signature `sigma_y(x, y, y_fit, *params)`."
)


type sigma_ftype = Callable[[ndarray, ndarray, ndarray], ndarray]


class Xlist:
    def __init__(self, x: Iterable[array_like]) -> None:
        """
        Custom list to trick `curve_fit`; For linked fitting of files with different x-axis sizes.

        :param x: An `Iterable` of arrays.
        """
        self.x = [np.asarray(_x, dtype=float) for _x in x]

    def __iter__(self) -> Generator[ndarray[floating], None, None]:
        yield from self.x

    def __getitem__(self, key: int_like) -> ndarray[floating]:
        return self.x[key]

    def __setitem__(self, key: int_like, value: array_like) -> None:
        self.x[key] = np.asarray(value, dtype=float)


def is_sigma_ftype(a: Any) -> TypeGuard[sigma_ftype]:
    return bool(callable(a))


def _wrap_sigma_y(sigma_y: sigma_ftype, uncs: ndarray) -> sigma_ftype:
    i = -uncs.size if uncs.size > 0 else None

    def func(x: ndarray, y: ndarray, y_fit: ndarray, *params: Any) -> ndarray:
        return np.concatenate([sigma_y(x, y, y_fit, *params)[:i], uncs], axis=0)

    return func


def _sqrt_zero_free(x: array_like) -> ndarray[floating]:
    x = np.asarray(x, dtype=float)
    y = np.ones_like(x, dtype=float)
    y[x > 1] = np.sqrt(x)
    return y


def sigma_poisson(x: array_like, y: array_like, y_model: array_like, *params: Any) -> ndarray[floating]:
    return _sqrt_zero_free(y_model)


def residuals(model: _base.Model, x: ndarray, y: ndarray) -> ndarray:
    return y - model(x, *model.vals)


def reduced_chi2(model: _base.Model, x: ndarray, y: ndarray, sigma_y: ndarray) -> float:
    fixed, _bounds = model.fit_prepare()
    n_free = sum(int(not f) for f in fixed)
    return np.sum(residuals(model, x, y) ** 2 / sigma_y**2) / (y.size - n_free)


def fit(
    model: _base.Model,
    x: Any,
    y: array_like,
    sigma_x: array_like | None = None,
    sigma_y: array_like | Callable | None = None,
    report: bool = False,
    routine: Callable | str | None = None,
    guess_offset: bool = False,
    mc_sigma: int = 0,
    **kwargs,
) -> tuple[ndarray, ndarray, dict]:
    """
    Fit a `Model` from `qspec.models` to data. This fit routine encapsulates the `qspec.curve_fit`
    and the `qspec.odr_fit` routines to facilitate the use of all modular fit model features,
    such as *Priors* and `Linked` fitting.

    :param model: The `Model` to fit.
    :param x: The input data. Can be any object accepted by the model.
     If model is a Linked model, `x` should be a list of objects compatible with the linked models.
    :param y: The y data. This has to be a 1-d array or a list of 1-d arrays if model is a Linked model.
    :param sigma_x: The uncertainties of the x-values. This is only compatible with Monte-Carlo sampling and the
     odr_fit routine. If `sigma_x` is not `None`, no routine is specified and `mc_sigma == 0`,
     the routine is automatically set to odr_fit.
    :param sigma_y: The uncertainties of the y-values.
     This has to be a 1-d array or a list of 1-d arrays if model is a Linked model and have the same shape as `y`.
     If routine is `curve_fit`, sigma may be a function `g` such that `g(x, y, model(x, *params), *params) -> sigma`.
     `g` should accept the same `x` as the `model` while `y` and `model(x, *params)` should be 1-d arrays.
    :param report: Whether to print the fit results.
    :param routine: The routine to use for fitting. Currently supported are {'curve_fit', 'odr_fit'}.
     If `None`, `curve_fit` is used. See `sigma_x` for one exception.
    :param guess_offset: Guess initial parameters for Offset models.
     Currently, this is not working if `x` is not a 1d-array.
    :param mc_sigma: The number of samples to generate. If 0, no Monte-Carlo sampling will be done.
     This is not available with linked fitting.
    :param kwargs: Additional kwargs to pass to the fit `routine`.
    :returns: (popt, pcov, info) The optimized parameters their covariance matrix
     and a dictionary containing info about the fit.
    :raises (ValueError, TypeError): If the specified routine is not supported (`ValueError`) or
     the specified `model` is not a `qspec.models.Model` (`TypeError`).
    """
    if routine is None:
        _routine: Callable = odr_fit if sigma_x is not None and mc_sigma == 0 else curve_fit
    elif isinstance(routine, str) and routine in ROUTINES:
        _routine = eval(routine)
    elif isinstance(routine, Callable) and routine.__name__ in ROUTINES:
        _routine = routine
    else:
        raise ValueError(f"Specified routine {routine} is not supported.")

    if not isinstance(model, _base.Model):
        raise TypeError(f"`model` must have, or inherit, type `qspec.models.Model` but has type {type(model)}.")
    if model.error:
        raise ValueError(model.error)

    sigma = kwargs.get("sigma")
    if sigma_y is None:
        sigma_y = sigma
    elif sigma is not None:
        print_colored("WARNING", "Parameter `sigma` is redundant. `sigma_y` will be used.")

    if callable(sigma_y) and mc_sigma:
        raise ValueError("`sigma(_y)` must not be callable if `mc_sigma` > 0.")

    if isinstance(model, _base.Linked):
        if mc_sigma != 0:
            raise TypeError("`Linked` models are currently not supported with Monte-Carlo sampling.")

        if not isinstance(x, Iterable):
            raise ValueError("If `model` is a `Linked` model, `x` must be an `Iterable.`")
        if not isinstance(y, Iterable):
            raise ValueError("If `model` is a `Linked` model, `y` must be an `Iterable.`")

        models_offset = _helper.find_models(model, _base.Offset)
        for _offset, _x, _y in zip(models_offset, x, y, strict=True):
            if _offset is not None:
                _offset.update_on_call = False
                _offset.gen_offset_masks(_x)
                if guess_offset:
                    _offset.guess_offset(_x, _y)

        model.inherit_vals()
        x = Xlist(x)
        y = np.concatenate(np.asarray(y, dtype=float), axis=0)

        if sigma_x is not None:
            sigma_x = np.concatenate(np.asarray(sigma_x, dtype=float), axis=0)
        if sigma_y is not None and not callable(sigma_y):
            sigma_y = np.concatenate(np.asarray(sigma_y, dtype=float), axis=0)

    else:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        sigma_x = asarray_optional(sigma_x, dtype=float)
        if not callable(sigma_y):
            sigma_y = asarray_optional(sigma_y, dtype=float)

        models_offset = [_helper.find_model(model, _base.Offset)]
        if models_offset[0] is not None:
            models_offset[0].update_on_call = False
            models_offset[0].gen_offset_masks(x)
            if guess_offset:
                models_offset[0].guess_offset(x, y)

    discard_y_pars = False
    if not isinstance(model, _base.YPars):
        discard_y_pars = True
        model = _base.YPars(model)
        vals = np.array([_val_fix_to_val(model.vals[p_y], model.fixes[p_y]) for p_y in model.p_y], dtype=float)
        uncs = np.array([_fix_to_unc(model.fixes[p_y]) for p_y in model.p_y], dtype=float)

        y = np.concatenate([y, vals], axis=0)
        if is_sigma_ftype(sigma_y):
            sigma_y = _wrap_sigma_y(sigma_y, uncs)
        elif callable(sigma_y):  # Currently unreachable as all callables are accepted as `sigma_ftype`.
            raise ValueError(SIGMA_FTYPE_SIGNATURE_ERROR)
        elif sigma_y is not None:
            sigma_y = np.concatenate([sigma_y, uncs], axis=0)

    p0_fixed, bounds = model.fit_prepare()

    kwargs.update({"report": False})
    if _routine.__name__ == "curve_fit":
        kwargs.update({"bounds": bounds, "sigma": sigma_y})
    if _routine.__name__ == "odr_fit":
        kwargs.update({"sigma_x": sigma_x, "sigma_y": sigma_y})

    err = False
    warn = False
    e = None
    try:
        if mc_sigma:
            assert isinstance(x, np.ndarray)
            assert isinstance(y, np.ndarray)
            assert not callable(sigma_y)

            kwargs["sigma"] = None
            kwargs["absolute_sigma"] = False
            kwargs["sigma_x"] = None
            kwargs["sigma_y"] = None
            if _routine.__name__ == "curve_fit":
                del kwargs["sigma_x"], kwargs["sigma_y"]
            if _routine.__name__ == "odr_fit":
                del kwargs["sigma"], kwargs["absolute_sigma"]

            if sigma_x is None:
                x_samples = [x] * mc_sigma
            else:
                x_samples = np.random.default_rng().normal(loc=x, scale=sigma_x, size=(mc_sigma, *x.shape))

            if sigma_y is None:
                y_samples = [y] * mc_sigma
            else:
                y_samples = np.random.default_rng().normal(loc=y, scale=sigma_y, size=(mc_sigma, *y.shape))

            pt = np.array(
                [
                    _routine(model, x_sample, y_sample, p0=model.vals, p0_fixed=p0_fixed, **kwargs)[0]
                    for x_sample, y_sample in zip(x_samples, y_samples)
                ],
                dtype=float,
            )
            popt = np.mean(pt, axis=0)
            pcov = np.zeros((popt.size, popt.size))
            indices = np.array([i for i, fix in enumerate(p0_fixed) if not fix])
            mask = (indices[:, None], indices)
            pcov[mask] = np.cov(pt[:, indices], rowvar=False)
        else:
            popt, pcov = _routine(model, x, y, p0=model.vals, p0_fixed=p0_fixed, **kwargs)

        popt = np.array(model.update_args(popt))
        model.set_vals(popt, force=True)

        if sigma_y is None:
            chi2 = 0.0
        elif is_sigma_ftype(sigma_y):
            chi2 = reduced_chi2(model, x, y, sigma_y(x, y, model(x, *popt), *popt))
        elif callable(sigma_y):  # Currently unreachable as all callables are accepted as `sigma_ftype`.
            raise ValueError(SIGMA_FTYPE_SIGNATURE_ERROR)
        else:
            chi2 = reduced_chi2(model, x, y, sigma_y)

    except (ValueError, RuntimeError) as _e:
        warn = True
        err = True
        chi2 = 0.0
        popt = np.array(model.vals, dtype=float)
        pcov = np.zeros((popt.size, popt.size))
        e = _e

    if report:
        digits = int(np.floor(np.log10(np.abs(model.size)))) + 1
        print("Optimized parameters:")
        for j, (name, val, unc) in enumerate(zip(model.names, popt, np.sqrt(np.diag(pcov)))):
            print(f"{str(j).zfill(digits)}:   {name} = {val} +/- {unc}")
        print("\nCov. Matrix:")
        print_cov(pcov, normalize=True, decimals=2)
        print(f"\nRed. chi2 = {np.around(chi2, decimals=2)}")

        if err:
            print_colored("FAIL", f"Error while fitting: {e}\n")
        elif np.any(np.isinf(pcov)):
            warn = True
            print_colored("WARNING", "Failed to estimate uncertainties.\n")
        else:
            print_colored("OKGREEN", "Fit successful.\n")

    if discard_y_pars:
        model = model.model  # Discard the YPars model.

    if isinstance(model, _base.Linked):
        for model_offset in models_offset:
            if model_offset is not None:
                model_offset.update_on_call = True  # Reset the offset model to be updated on call.
    else:
        if models_offset[0] is not None:
            models_offset[0].update_on_call = True  # Reset the offset model to be updated on call.

    info = dict(warn=warn, err=err, chi2=chi2)
    return popt, pcov, info
