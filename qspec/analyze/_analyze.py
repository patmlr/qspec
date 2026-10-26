"""
qspec.analyze._analyze
======================

Module for analyzing/evaluating/fitting data.

Linear regression algorithms (2d):
    - york_fit(); [York et al., Am. J. Phys. 72, 367 (2004)]
    - linear_fit(); 2-dimensional maximum likelihood fit.
    - linear_monte_carlo(); based on [Gebert et al., Phys. Rev. Lett. 115, 053003 (2015), Suppl.]

Linear regression algorithms (nd):
    - linear_nd_fit(); n-dimensional maximum likelihood fit.
    - linear_nd_monte_carlo(); based on [Gebert et al., Phys. Rev. Lett. 115, 053003 (2015), Suppl.]

Curve fitting methods:
    - curve_fit(); Reimplements the scipy.optimize.curve_fit method to allow fixing parameters
      and having parameter-dependent y-uncertainties.
    - odr_fit(); Encapsulates the odrpack.odr_fit method to accept inputs similarly to curve_fit().

Classes:
    - King; Creates a King plot with isotope shifts or nuclear charge radii.

LICENSE NOTES:
    The method curve_fit is a modified version of scipy.optimize.curve_fit.
    Therefore, it is licensed under the 'BSD 3-Clause "New" or "Revised" License' provided with scipy.
"""

import inspect
import warnings
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import LinAlgError, cholesky, solve_triangular, svd
from scipy.optimize import Bounds, OptimizeWarning, least_squares, leastsq, minimize
from scipy.stats import multivariate_normal, norm

from qspec import tools
from qspec.analyze._analyze_cpp import generate_collinear_points_cpp
from qspec.physics import mass_factor, me_u, me_u_d
from qspec.qtypes import (
    Any,
    Callable,
    Iterable,
    array_like,
    asarray,
    floating,
    int_like,
    integer,
    ndarray,
    scalar,
    scalar_nd,
)

__all__ = [
    "King",
    "const",
    "covariance_matrix",
    "curve_fit",
    "draw_sigma2d",
    "draw_straight_unc_area",
    "ellipse2d",
    "generate_collinear_points_py",
    "linear_alpha_fit",
    "linear_fit",
    "linear_monte_carlo",
    "linear_nd_fit",
    "linear_nd_monte_carlo",
    "odr_fit",
    "poly",
    "straight",
    "straight_direction",
    "straight_std",
    "straight_x_std",
    "weight",
    "york_fit",
]


def _wrap_func_pars(func: Callable, p0: ndarray, p0_fixed: ndarray) -> Callable:
    def func_wrapped(x: Any, *params: Any) -> ndarray:
        _params = p0
        _params[~p0_fixed] = np.asarray(params)
        return func(x, *_params)

    return func_wrapped


def _wrap_func_sigma(sigma: Callable, p0: ndarray, p0_fixed: ndarray) -> Callable:
    def transform(xdata: ndarray, ydata: ndarray, yfunc: ndarray, *params: Any) -> ndarray:
        _params = p0
        _params[~p0_fixed] = np.asarray(params)
        return 1 / sigma(xdata, ydata, yfunc, *_params)

    return transform


def _wrap_func(func: Callable, xdata: Any, ydata: ndarray, transform: Callable | ndarray | None) -> Callable:
    """
    The residual function of `scipy.optimize.curve_fit`, extended for callable transforms.
    """
    if transform is None:

        def func_wrapped(params: ndarray) -> ndarray:
            return func(xdata, *params) - ydata

    elif callable(transform):

        def func_wrapped(params: ndarray) -> ndarray:
            yfunc = func(xdata, *params)
            return transform(xdata, ydata, yfunc, *params) * (yfunc - ydata)

    elif transform.ndim == 1:

        def func_wrapped(params: ndarray) -> ndarray:
            return transform * (func(xdata, *params) - ydata)

    else:

        def func_wrapped(params: ndarray) -> ndarray:
            return solve_triangular(transform, func(xdata, *params) - ydata, lower=True)

    return func_wrapped


def _wrap_jac(jac: Callable, xdata: Any, transform: ndarray | None) -> Callable[[ndarray], ndarray]:
    """
    The Jacobian of the residual function of `scipy.optimize.curve_fit`.
    """
    if transform is None:

        def jac_wrapped(params: ndarray) -> ndarray:
            return np.asarray(jac(xdata, *params))

    elif transform.ndim == 1:

        def jac_wrapped(params: ndarray) -> ndarray:
            return transform[:, np.newaxis] * np.asarray(jac(xdata, *params))

    else:

        def jac_wrapped(params: ndarray) -> ndarray:
            return solve_triangular(transform, np.asarray(jac(xdata, *params)), lower=True)

    return jac_wrapped


def _prepare_bounds(bounds: tuple[array_like, array_like], n: int) -> tuple[ndarray, ndarray]:
    """
    Broadcast scalar bounds to `n` parameters as in `scipy.optimize.curve_fit`.
    """
    lb, ub = (np.asarray(b, dtype=float) for b in bounds)
    if lb.ndim == 0:
        lb = np.resize(lb, n)
    if ub.ndim == 0:
        ub = np.resize(ub, n)
    return lb, ub


def _initialize_feasible(lb: ndarray, ub: ndarray) -> ndarray:
    """
    Initial parameters within the bounds as in `scipy.optimize.curve_fit`.
    """
    p0 = np.ones_like(lb)
    lb_finite = np.isfinite(lb)
    ub_finite = np.isfinite(ub)

    mask = lb_finite & ub_finite
    p0[mask] = 0.5 * (lb[mask] + ub[mask])

    mask = lb_finite & ~ub_finite
    p0[mask] = lb[mask] + 1

    mask = ~lb_finite & ub_finite
    p0[mask] = ub[mask] - 1

    return p0


def poly(x: array_like, *args: scalar) -> ndarray:
    r"""
    A polynomial

    $$
    f(x) = \sum\limits_{k=0}^n \frac{a_k}{k!}x^k
    $$

    of order `n = len(args) - 1` with coefficients `args`.

    :param x: The $x$ values.
    :param args: The coefficients of the polynomial $a_k$.
    :returns: (y) The function value $f(x)$.
    """
    x = np.asarray(x)
    return np.sum([args[n] / tools.factorial(n) * x**n for n in range(len(args))], axis=0)


def const(x: array_like, a: array_like) -> ndarray:
    r"""
    A constant

    $$
    f(x) = a.
    $$

    :param x: The $x$ values.
    :param a: The y-intercept $a$.
    :returns: (y) The function value $f(x)$.
    """
    return np.full_like(x, a)


def straight(x: array_like, a: array_like, b: array_like) -> ndarray:
    r"""
    A straight line

    $$
    f(x) = a + bx.
    $$

    :param x: The $x$ values.
    :param a: The y-intercept $a$.
    :param b: The slope $b$.
    :returns: (y) The function value $f(x)$.
    """
    x, a, b = np.asarray(x), np.asarray(a), np.asarray(b)
    return a + b * x


def straight_std(x: array_like, sigma_a: array_like, sigma_b: array_like, corr_ab: array_like) -> ndarray:
    r"""
    The standard deviation of a straight line [`straight`]

    $$
    \sigma_f(x) = \sqrt{\sigma_a^2 + (x\sigma_b)^2 + 2x\sigma_a\sigma_b\rho_{ab}}.
    $$

    :param x: The $x$ values.
    :param sigma_a: The standard deviation $\sigma_a$ of the y-intercept.
    :param sigma_b: The standard deviation $\sigma_b$ of the slope.
    :param corr_ab: The correlation coefficient $\rho_{ab}$ between the slope and the y-intercept.
    :returns: (sigma_f) The standard deviation $\sigma_f$ of a straight line.
    """
    return straight_x_std(x, 0.0, 0.0, sigma_a, sigma_b, corr_ab)


def straight_x_std(
    x: array_like, b: array_like, sigma_x: array_like, sigma_a: array_like, sigma_b: array_like, corr_ab: array_like
) -> ndarray:
    r"""
    The standard deviation of a straight line [`straight`] including the uncertainty of the $x$ values

    $$
    \sigma_f(x) = \sqrt{\sigma_a^2 + (x\sigma_b)^2 + 2x\sigma_a\sigma_b\rho_{ab}
    + (b\sigma_x)^2 + (\sigma_b\sigma_x)^2}.
    $$

    :param x: The $x$ values.
    :param b: The slope $b$.
    :param sigma_x: The standard deviation $\sigma_x$ of the `x` values.
    :param sigma_a: The standard deviation $\sigma_a$ of the y-intercept.
    :param sigma_b: The standard deviation $\sigma_b$ of the slope `b`.
    :param corr_ab: The correlation coefficient $\rho_{ab}$ between the slope `b` and the y-intercept.
    :returns: (sigma_f) The standard deviation $\sigma_f$ of a straight line.
    """
    x, sigma_x = np.asarray(x), np.asarray(sigma_x)
    b, sigma_a, sigma_b, corr_ab = np.asarray(b), np.asarray(sigma_a), np.asarray(sigma_b), np.asarray(corr_ab)
    return np.sqrt(
        sigma_a**2
        + (x * sigma_b) ** 2
        + 2 * x * sigma_a * sigma_b * corr_ab
        + (b * sigma_x) ** 2
        + (sigma_b * sigma_x) ** 2
    )


def straight_direction(p_0: array_like, p_1: array_like, axis: int = -1) -> ndarray:
    r"""
    The normalized directional vector of a straight line through two points

    $$
    \vec{r} = \frac{\vec{p}_1 - \vec{p}_0}{|\vec{p}_1 - \vec{p}_0|}.
    $$

    :param p_0: The first point(s) $\vec{p}_0$.
    :param p_1: The second point(s) $\vec{p}_1$.
    :param axis: The axis along which the vector components are aligned.
    :returns: (r) The direction $\vec{r}$ of a straight line.
    """
    p_0, p_1 = np.asarray(p_0), np.asarray(p_1)
    dr = p_1 - p_0
    return dr / np.expand_dims(tools.absolute(dr, axis=axis), axis=axis)


def draw_straight_unc_area(
    x: array_like, a: array_like, b: array_like, sigma_a: array_like, sigma_b: array_like, corr_ab: array_like, **kwargs
) -> None:
    r"""
    Draw the $1\sigma_f$ standard deviation area of a straight line $f$ [`straight`] enclosed by

    $$
    [f(x) - \sigma_f(x), f(x) + \sigma_f(x)],
    $$

    see [`straight_std`].

    :param x: The $x$ values.
    :param a: The y-intercept $a$.
    :param b: The slope $b$.
    :param sigma_a: The standard deviation $\sigma_a$ of the y-intercept `a`.
    :param sigma_b: The standard deviation $\sigma_b$ of the slope `b`.
    :param corr_ab: The correlation coefficient $\rho_{ab}$ between the y-intercept `a` and the slope `b`.
    :param kwargs: Additional keyword arguments are passed to `matplotlib.pyplot.fill_between`.
    """
    y = straight(x, a, b)
    unc = straight_std(x, sigma_a, sigma_b, corr_ab)
    plt.fill_between(x, y - unc, y + unc, **kwargs)


def ellipse2d(
    x: array_like, y: array_like, scale_x: array_like, scale_y: array_like, phi: array_like, corr: array_like
) -> tuple[ndarray, ndarray]:
    r"""
    A point on an ellipse in 2d-space

    $$
    \vec{p}(x, y) = \begin{pmatrix}
    x + \sigma_x\cos(\phi) \\
    y + \sigma_y\left[\rho_{xy}\cos(\phi) + \sqrt{1 - \rho_{xy}^2}\sin(\phi)\right]
    \end{pmatrix}.
    $$

    :param x: The $x$-component of the position of the ellipse.
    :param y: The $y$-component of the position of the ellipse.
    :param scale_x: The amplitude $\sigma_x$ of the $x$-component.
    :param scale_y: The amplitude $\sigma_y$ of the $y$-component.
    :param phi: The parameter angle $\phi$ of the ellipse (rad).
    :param corr: The correlation coefficient $\rho_{xy}$ between the `x` and `y` data.
    :returns: (p_x, p_y) A point $\vec{p}(x, y)$ on an ellipse.
    """
    x, y, scale_x, scale_y, corr = asarray(x, y, scale_x, scale_y, corr)
    return x + scale_x * np.cos(phi), y + scale_y * (corr * np.cos(phi) + np.sqrt(1 - corr**2) * np.sin(phi))


def draw_sigma2d(
    x: array_like, y: array_like, sigma_x: array_like, sigma_y: array_like, corr: array_like, n: int = 1, **kwargs
) -> None:
    r"""
    Draw the $n\sigma$ bounds of the given data points $(x, y)$ with standard deviations $(\sigma_x, \sigma_y)$
    and correlation $\rho_{xy}$, see [`ellipse2d`].

    :param x: The $x$ data.
    :param y: The $y$ data.
    :param sigma_x: The standard deviation $\sigma_x$ of the `x` data.
    :param sigma_y: The standard deviation $\sigma_y$ of the `y` data.
    :param corr: The correlation coefficients $\rho_{xy}$ between the `x` and `y` data.
    :param n: The maximum sigma region $n$ to draw.
    :param kwargs: Additional keyword arguments are passed to `matplotlib.pyplot.plot`.
     Use the key `fmt` to specify the format string (third argument) of `matplotlib.pyplot.plot`.
    """
    x, y, sigma_x, sigma_y, corr = asarray(x, y, sigma_x, sigma_y, corr)

    fmt = "-k"
    if "fmt" in list(kwargs.keys()):
        fmt = kwargs.pop("fmt")

    phi = np.linspace(0.0, 2 * np.pi, 361)

    for x_i, y_i, s_x, s_y, r in zip(x, y, sigma_x, sigma_y, corr):
        for i in range(1, n + 1, 1):
            plt.plot(*ellipse2d(x_i, y_i, i * s_x, i * s_y, phi, r), fmt, **kwargs)


def weight(sigma: array_like) -> ndarray:
    r"""
    The weight

    $$
    w = \frac{1}{\sigma^2},
    $$

    corresponding to the $1\sigma$ uncertainty `sigma`.

    :param sigma: The $1\sigma$ uncertainty.
    :returns: (w) The weight $w$.
    """
    sigma = np.asarray(sigma)
    return 1.0 / sigma**2


def york_fit(
    x: array_like,
    y: array_like,
    sigma_x: array_like | None = None,
    sigma_y: array_like | None = None,
    corr: array_like | None = None,
    iter_max: int = 200,
    report: bool = False,
    show: bool = False,
    **kwargs,
) -> tuple[ndarray, ndarray]:
    r"""
    A linear regression algorithm to find the best straight line through points $\vec{\mu}_i\in\mathbb{R}^2$
    with covariances $\mathbf{\Sigma}_i\in\mathbb{R}^{2\times 2}$,
    assuming $2$-dimensional multivariate normal distributions $\mathcal{N}(\vec{\mu}_i, \mathbf{\Sigma}_i)$.
    The data points and covariances are given by

    $$
    \vec{\mu} = \begin{pmatrix}x\\y\end{pmatrix},\quad
    \mathbf{\Sigma} = \begin{pmatrix}
    \sigma_x^2 & \rho_{xy}\sigma_x\sigma_y \\
    \rho_{xy}\sigma_x\sigma_y & \sigma_y^2
    \end{pmatrix}.
    $$

    The algorithm is described in
    [<a href="https://doi.org/10.1119/1.1632486">York et al., Am. J. Phys. 72, 367 (2004)</a>].

    :param x: The $x$ data.
    :param y: The $y$ data.
    :param sigma_x: The standard deviation $\sigma_x$ of the `x` data. If `None`, it is 1.
    :param sigma_y: The standard deviation $\sigma_y$ of the `y` data. If `None`, it is 1.
    :param corr: The correlation coefficients $\rho_{xy}$ between the `x` and `y` data. If `None`, it is 0.
    :param iter_max: The maximum number of iterations to find the best slope.
    :param report: Whether to print the result of the fit.
    :param show: Whether to plot the fit result.
    :param kwargs: Additional keyword arguments are accepted for compatibility with the other linear regression
     algorithms and ignored.
    :returns: (popt, pcov) The best y-intercept and slope and their covariance matrix.
    :raises ValueError: If the optimization fails.
    """
    x, y = np.asarray(x), np.asarray(y)

    if sigma_x is None:
        sigma_x = np.full_like(x, 1.0)
        if report:
            print("\nNo uncertainties for `x` were given. Assuming `sigma_x` = 1.")

    if sigma_y is None:
        sigma_y = np.full_like(y, 1.0)
        if report:
            print("\nNo uncertainties for `y` were given. Assuming `sigma_y` = 1.")

    sigma_2d = True
    if corr is None:
        sigma_2d = False
        corr = 0.0

    sigma_x, sigma_y, corr = np.asarray(sigma_x), np.asarray(sigma_y), np.asarray(corr)

    p_opt = curve_fit(straight, x, y, p0=[0.0, 1.0], sigma=sigma_y)[0]
    b_init = p_opt[1]
    b = b_init

    w_x, w_y = weight(sigma_x), weight(sigma_y)
    alpha = np.sqrt(w_x * w_y)

    n = 0
    r_tol = 1e-15
    tol = 1.0
    mod_w = None
    x_bar, y_bar = None, None
    beta = None
    while tol > r_tol and n < iter_max:
        b_init = b
        mod_w = w_x * w_y / (w_x + b_init**2 * w_y - 2.0 * b_init * corr * alpha)
        x_bar, y_bar = np.average(x, weights=mod_w), np.average(y, weights=mod_w)
        u = x - x_bar
        v = y - y_bar
        beta = mod_w * (u / w_y + v * b_init / w_x - straight(u, v, b_init) * corr / alpha)
        b = np.sum(mod_w * beta * v) / np.sum(mod_w * beta * u)
        tol = abs((b - b_init) / b)
        n += 1

    if mod_w is None or x_bar is None or y_bar is None or beta is None:
        raise ValueError(
            f"Optimization failed because conditions `tol > r_tol`: {tol} > {r_tol}"
            f" or `n < iter_max`: {n} > {iter_max} are not fulfilled."
        )

    a = y_bar - b * x_bar
    x_i = x_bar + beta
    x_bar = np.average(x_i, weights=mod_w)
    u = x_i - x_bar
    sigma_b = np.sqrt(1.0 / np.sum(mod_w * u**2))
    sigma_a = np.sqrt(1.0 / np.sum(mod_w) + x_bar**2 * sigma_b**2)
    corr_ab = -x_bar * sigma_b / sigma_a
    chi2 = np.sum(mod_w * (y - b * x - a) ** 2) / (x.size - 2)

    if report:
        if n == iter_max:
            print(f"\nMaximum number of iterations ({iter_max}) was reached.")
        tools.printh("york_fit result:")
        print(f"a: {a} +/- {sigma_a}\nb: {b} +/- {sigma_b}\ncorr_ab: {corr_ab}\nred. chi2: {chi2}")
    if show:
        x_cont = np.linspace(np.min(x) - 0.1 * (np.max(x) - np.min(x)), np.max(x) + 0.1 * (np.max(x) - np.min(x)), 1001)
        plt.xlabel("x")
        plt.ylabel("y")
        if sigma_2d:
            plt.plot(x, y, "k.", label="Data")
            draw_sigma2d(x, y, sigma_x, sigma_y, corr, n=1)
        else:
            plt.errorbar(x, y, xerr=sigma_x, yerr=sigma_y, fmt="k.", label="Data")
        plt.plot(x_cont, straight(x_cont, a, b), "b-", label="Fit")
        y_min = straight(x_cont, a, b) - straight_std(x_cont, sigma_a, sigma_b, corr_ab)
        y_max = straight(x_cont, a, b) + straight_std(x_cont, sigma_a, sigma_b, corr_ab)
        plt.fill_between(x_cont, y_min, y_max, color="b", alpha=0.3, antialiased=True)
        plt.legend(loc="best", numpoints=1)
        plt.show()

    popt = np.array([a, b], dtype=float)
    pcov = np.array([[sigma_a**2, sigma_a * sigma_b * corr_ab], [sigma_a * sigma_b * corr_ab, sigma_b**2]], dtype=float)
    return popt, pcov


def covariance_matrix(
    cov: array_like | None = None,
    sigma: array_like | None = None,
    corr: array_like | None = None,
    k: int | None = None,
    n: int | None = None,
) -> ndarray:
    r"""
    Construct covariance matrices $\mathbf{\Sigma}_i\in\mathbb{R}^{n\times n}$
    from standard deviations $\vec{\sigma}_i\in\mathbb{R}^n$
    and correlation matrices $\mathbf{\rho}_i\in\mathbb{R}^{n\times n}$

    $$
    (\mathbf{\Sigma}_i)_{jk} = (\vec{\sigma}_i)_j(\vec{\sigma}_i)_k(\mathbf{\rho}_i)_{jk}.
    $$

    :param cov: The covariance matrices $\mathbf{\Sigma}_i$. Must have shape `(k, n, n)`,
     where `k` is the number of data points and `n` the number of dimensions of each data point.
     If `None`, the other parameters are used. Else, all other parameters are ignored.
    :param sigma: The standard deviations $\vec{\sigma}_i$ of the data vectors. Must have shape `(k, n)`.
     If `None`, all diagonal elements of the covariance matrix equal 1.
    :param corr: The correlation matrices $\mathbf{\rho}_i$ of the data vectors. Must have shape `(k, n, n)`.
     If `n == 2` it can have shape `(k, )`. If `None`, all off-diagonal elements of the covariance matrix are 0.
    :param k: The number of data points. It is ignored if `sigma` or `corr` is specified.
    :param n: The number of dimensions $n$ of each data point. It is ignored if `sigma` or `corr` is specified.
    :returns: (cov) Covariance matrices $\mathbf{\Sigma}_i$ constructed from `sigma` and `corr`.
    """
    _cov = tools.asarray_optional(cov, dtype=float)
    sigma, corr = tools.asarray_optional(sigma, dtype=float), tools.asarray_optional(corr, dtype=float)

    if _cov is None:
        if k is None:
            k = 1
        _k = k

        if n is None:
            n = 2
        _n = n

        if sigma is not None and corr is not None:
            _k, _n = sigma.shape
            if len(corr.shape) < 2 and _n == 2:
                corr = np.array([[[1, corr[__k]], [corr[__k], 1]] for __k in range(_k)], dtype=float)

        elif sigma is not None and corr is None:
            _k, _n = sigma.shape
            corr = np.array([np.identity(_n) for _ in range(_k)], dtype=float)

        elif sigma is None and corr is not None:
            _k = corr.shape[0]

            if len(corr.shape) < 2:
                _n = 2
                corr = np.array([[[1, corr[__k]], [corr[__k], 1]] for __k in range(_k)], dtype=float)
            else:
                _n = corr.shape[1]

            sigma = np.ones((_k, _n), dtype=float)

        else:
            sigma = np.ones((_k, _n), dtype=float)
            corr = np.array([np.identity(_n) for _ in range(_k)], dtype=float)

        _cov = sigma[:, :, None] * sigma[:, None, :] * corr

    return _cov


def _x0_r_from_p(p: ndarray) -> tuple[ndarray, ndarray]:
    dim = p.shape[0] // 2
    return p[:dim], p[dim:]


def _linear_nd_t0(p: ndarray, x: ndarray, cov_inv: ndarray) -> ndarray:
    x0, r = _x0_r_from_p(p)
    xi = x0[None, :] - x
    ax = np.sum(cov_inv * r[None, None, :], axis=-1)

    var_t = 1 / np.sum(r[None, :] * ax, axis=-1)
    return -var_t * np.sum(xi * ax, axis=-1)


def _linear_nd_func(p: ndarray, x: ndarray, cov_inv: ndarray) -> ndarray:
    x0, r = _x0_r_from_p(p)
    xi = x0[None, :] - x
    ax = np.sum(cov_inv * r[None, None, :], axis=-1)
    ar = np.sum(cov_inv * xi[:, None, :], axis=-1)

    var_t = 1 / np.sum(r[None, :] * ax, axis=-1)
    t0 = -var_t * np.sum(xi * ax, axis=-1)
    tx = np.sum(xi * ar, axis=-1)
    return 0.5 * np.sum(tx - t0**2 / var_t, axis=-1)


def _linear_nd_jac(p: ndarray, x: ndarray, cov_inv: ndarray) -> ndarray:
    r"""
    The gradient vector / Jacobian.

    :param p: The optimized parameters vector $\vec{p}$.
    :param x: The data vector $\vec{x}$.
    :param cov_inv: The inverse covariance matrix $\mathbf{\Sigma}_i^{-1}$.
    :returns: The gradient vector / Jacobian $\vec{J}$.
    """
    x0, r = _x0_r_from_p(p)
    xi = x0[None, :] - x
    ax = np.sum(cov_inv * r[None, None, :], axis=-1)
    ar = np.sum(cov_inv * xi[:, None, :], axis=-1)
    a = np.sum(xi * ax, axis=-1)[:, None]
    b = np.sum(r[None, :] * ax, axis=-1)[:, None]

    xx = np.sum(ar - a / b * ax, axis=0)
    rr = -np.sum(a / b * ar - a**2 / b**2 * ax, axis=0)

    return np.concatenate([xx, rr], axis=0)


def _linear_nd_hess(p: ndarray, x: ndarray, cov_inv: ndarray) -> ndarray:
    r"""
    The Hesse matrix.

    :param p: The optimized parameters vector $\vec{p}$.
    :param x: The data vector $\vec{x}$.
    :param cov_inv: The inverse covariance matrix $\mathbf{\Sigma}_i^{-1}$.
    :returns: The Hesse matrix $\mathbf{H}$.
    """
    x0, r = _x0_r_from_p(p)
    xi = x0[None, :] - x
    ax = np.sum(cov_inv * r[None, None, :], axis=-1)
    ar = np.sum(cov_inv * xi[:, None, :], axis=-1)
    a = np.sum(xi * ax, axis=-1)[:, None, None]
    b = np.sum(r[None, :] * ax, axis=-1)[:, None, None]

    xx = np.sum(cov_inv - ax[:, :, None] * ax[:, None, :] / b, axis=0)
    rr = -np.sum(
        (ar[:, :, None] / b - 2 * a / b**2 * ax[:, :, None]) * ar[:, None, :]
        + 2 * (2 * a**2 / b**3 * ax[:, :, None] - a / b**2 * ar[:, :, None]) * ax[:, None, :]
        - a**2 / b**2 * cov_inv,
        axis=0,
    )
    xr = -np.sum((ar[:, :, None] / b - 2 * a / b**2 * ax[:, :, None]) * ax[:, None, :] + a / b * cov_inv, axis=0)

    return np.block([[xx, xr], [xr.T, rr]])


def _get_linear_nd_reduced(x: ndarray, cov_inv: ndarray, p0: ndarray, mask: ndarray, method: str) -> Callable:
    m = eval(f"_linear_nd_{method}")
    dim = p0.shape[0] // 2
    axis = list(mask).index(False)
    i = {"func": None, "jac": mask, "hess": np.ix_(mask, mask)}[method]

    def func(p: ndarray) -> ndarray:
        _p = np.insert(p, [axis, axis + dim - 1], [p0[axis], p0[axis + dim]])
        return m(_p, x, cov_inv) if i is None else m(_p, x, cov_inv)[i]

    return func


def linear_nd_fit(
    x: array_like,
    cov: array_like | None = None,
    p0: array_like | None = None,
    axis: int | None = None,
    optimize_cov: bool = False,
    **kwargs: Any,
) -> tuple[ndarray, ndarray]:
    r"""
    Maximum likelihood fit for a straight line through points $\vec{\mu}_i\in\mathbb{R}^n$
    with covariances $\mathbf{\Sigma}_i\in\mathbb{R}^{n\times n}$,
    assuming $n$-dimensional multivariate normal distributions $\mathcal{N}(\vec{\mu}_i, \mathbf{\Sigma}_i)$.

    The algorithm is described in <a href="https://doi.org/10.1016/j.cpc.2025.109550">
    qspec's publication</a>.

    :param x: The data vectors $\vec{\mu}_i$. Must have shape `(k, n)`, where `k` is the number of data points
     and `n` is the number of dimensions of each point.
    :param cov: The covariance matrices $\mathbf{\Sigma}_i$ of the data vectors. Must have shape `(k, n, n)`.
     Use [`covariance_matrix`] to construct covariance matrices.
    :param p0: The start parameters for the linear fit. Must have shape `(2 * n, )`.
     The first `n` elements specify the origin vector of the straight line,
     the second `n` elements specify the direction of the straight line.
    :param axis: The index of the vector component of the n-dimensional vectors that are fixed for fitting.
     This is required since a straight line in `n` dimensions is fully described by `2 * (n - 1)` parameters.
     If `None`, the best axis is determined from the data,
     and the direction vector of the straight line is normalized.
    :param optimize_cov: If `True`, the origin vector of the straight line is optimized
     to yield the smallest covariances.
    :param kwargs: Additional keyword arguments are accepted for compatibility with [`linear_nd_monte_carlo`]
     and ignored.
    :returns: (popt, pcov) The optimized parameters and their covariances.
     The resulting shapes are `(2 * n, )` and `(2 * n, 2 * n)`.
    :raises numpy.linalg.LinAlgError: If a covariance matrix cannot be inverted.
    """
    x_temp = np.array(x, dtype=float)
    size, dim = x_temp.shape

    axis_was_none = False
    if axis is None:
        axis_was_none = True
        axis = int(np.argmax(np.abs(np.std(x_temp, ddof=1, axis=0) / np.mean(x_temp, axis=0))))
    elif axis < 0:
        axis += dim

    cov = covariance_matrix(cov, k=size, n=dim)

    x_off = x_temp[np.argmin(x_temp[:, axis], axis=0)].copy()
    x_temp -= x_off[None, :]
    i_fac = np.argmax(np.abs(x_temp), axis=0)
    x_fac = x_temp[i_fac, np.arange(dim)]
    x_temp /= x_fac[None, :]
    cov_temp = cov / (x_fac[None, None, :] * x_fac[None, :, None])

    try:
        cov_inv = np.linalg.inv(cov_temp)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(f"Unable to invert covariance matrix ('{e}')")

    if p0 is None:
        popt = np.zeros(2 * dim, dtype=float)
        popt[dim:] = np.ones(dim, dtype=float)
    else:
        popt = np.asarray(p0, dtype=float)
        tools.check_dimension(2 * dim, 0, popt)
        popt[:dim] -= x_off
        popt[:dim] /= x_fac
        popt[dim:] /= x_fac / x_fac[axis]

    i = np.arange(2 * dim, dtype=int)
    mask = i == axis
    mask += i == dim + axis
    mask = ~mask
    p0_red = popt[mask]

    res = minimize(
        _get_linear_nd_reduced(x_temp, cov_inv, popt, mask, "func"),
        p0_red,
        method="Newton-CG",
        jac=_get_linear_nd_reduced(x_temp, cov_inv, popt, mask, "jac"),
        hess=_get_linear_nd_reduced(x_temp, cov_inv, popt, mask, "hess"),
        options={"xtol": 1e-15},
    )

    popt[mask] = res.x

    def get_cov(_popt: ndarray[floating]) -> ndarray[floating]:
        h2 = _linear_nd_hess(_popt, x_temp, cov_inv)[np.ix_(mask, mask)]
        return np.linalg.inv(h2)

    if optimize_cov:

        def get_cov_t(t: scalar, _popt: ndarray[floating]) -> ndarray[floating]:
            _popt_cov = _popt.copy()
            _popt_cov[:dim] += t * _popt_cov[dim:]
            return get_cov(_popt_cov)

        def cost_cov(t: scalar, _popt: ndarray[floating]) -> floating:
            _pcov = get_cov_t(t, _popt)
            return np.sum(np.triu(_pcov, k=1) ** 2)

        def get_cost_cov(_popt: ndarray[floating]) -> Callable[[scalar], floating]:
            return lambda t: cost_cov(t, _popt)

        tp = _linear_nd_t0(popt, x_temp, cov_inv)
        popt[:dim] += np.mean(tp, axis=-1) * popt[dim:]
        tp = 0.1 * (np.max(tp, axis=-1) - np.min(tp, axis=-1))

        res = minimize(get_cost_cov(popt), np.full(1, tp), method="BFGS")
        popt[:dim] += res.x[0] * popt[dim:]

    if axis_was_none:
        popt[dim:] /= tools.absolute(popt[dim:])

    popt[:dim] *= x_fac
    popt[dim:] *= x_fac / x_fac[axis]
    x_temp *= x_fac[None, :]
    cov_temp *= x_fac[None, None, :] * x_fac[None, :, None]

    popt[:dim] += x_off
    if not optimize_cov:
        popt[:dim] -= popt[dim:] * popt[axis] / popt[dim + axis]
    x_temp += x_off[None, :]
    cov_inv = np.linalg.inv(cov_temp)

    pcov = np.zeros((2 * dim, 2 * dim), dtype=float)
    pcov[np.ix_(mask, mask)] = get_cov(popt)

    return popt, pcov


def linear_fit(
    x: array_like,
    y: array_like,
    sigma_x: array_like | None = None,
    sigma_y: array_like | None = None,
    corr: array_like | None = None,
    report: bool = False,
    **kwargs: Any,
) -> tuple[ndarray, ndarray]:
    r"""
    Maximum likelihood fit for a straight line through points $\vec{\mu}_i\in\mathbb{R}^2$
    with covariances $\mathbf{\Sigma}_i\in\mathbb{R}^{2\times 2}$,
    assuming $2$-dimensional multivariate normal distributions $\mathcal{N}(\vec{\mu}_i, \mathbf{\Sigma}_i)$.
    The data points and covariances are given by

    $$
    \vec{\mu} = \begin{pmatrix}x\\y\end{pmatrix},\quad
    \mathbf{\Sigma} = \begin{pmatrix}
    \sigma_x^2 & \rho_{xy}\sigma_x\sigma_y \\
    \rho_{xy}\sigma_x\sigma_y & \sigma_y^2
    \end{pmatrix}.
    $$

    This is a wrapper for the more general [`linear_nd_fit`] function.
    The algorithm is described in <a href="https://doi.org/10.1016/j.cpc.2025.109550">
    qspec's publication</a>.

    :param x: The $x$ data.
    :param y: The $y$ data.
    :param sigma_x: The standard deviation $\sigma_x$ of the `x` data. If `None`, it is 1.
    :param sigma_y: The standard deviation $\sigma_y$ of the `y` data. If `None`, it is 1.
    :param corr: The correlation coefficients $\rho_{xy}$ between the `x` and `y` data. If `None`, it is 0.
    :param report: Whether to print the result of the fit.
    :param kwargs: Additional keyword arguments are accepted for compatibility with the other linear regression
     algorithms and ignored.
    :returns: (popt, pcov) The best y-intercept and slope and their covariance matrix.
    :raises numpy.linalg.LinAlgError: If a covariance matrix cannot be inverted.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    sigma_x, sigma_y = tools.asarray_optional(sigma_x, dtype=float), tools.asarray_optional(sigma_y, dtype=float)

    mean = np.concatenate([x[:, None], y[:, None]], axis=1)

    if sigma_x is None:
        sigma_x = np.ones_like(x)
    if sigma_y is None:
        sigma_y = np.ones_like(y)
    sigma = np.concatenate([sigma_x[:, None], sigma_y[:, None]], axis=1)

    cov = covariance_matrix(sigma=sigma, corr=corr)

    popt, pcov = linear_nd_fit(mean, cov=cov, axis=0, optimize_cov=False)
    i = np.array([1, 3], dtype=int)
    popt, pcov = popt[i], pcov[np.ix_(i, i)]

    if report:
        tools.printh("linear_fit result:")
        print(
            f"a: {popt[0]} +/- {np.sqrt(pcov[0, 0])}\nb: {popt[1]} +/- {np.sqrt(pcov[1, 1])}\n"
            f"corr_ab: {pcov[0, 1] / np.sqrt(pcov[0, 0] * pcov[1, 1])}"
        )

    return popt, pcov


def _test_order_linear_nd_monte_carlo(
    x: ndarray,
    cov: ndarray,
    n_samples: int = 100000,
    n_accepted: int = 100000,
    method: str = "py",
    report: bool = False,
    **kwargs: Any,
) -> ndarray:
    r"""
    Test which data points should be used as boundary conditions for the generated lines.

    :param x: The data vectors $\vec{\mu}_i$. Must have shape `(k, n)`, where `k` is the number of data points
     and `n` is the number of dimensions of each point.
    :param cov: The covariance matrices $\mathbf{\Sigma}_i$ of the data vectors. Must have shape `(k, n, n)`.
     Use [`covariance_matrix`] to construct covariance matrices.
    :param n_samples: Maximum number of generated samples.
     If `None` and `method == "cpp"`, samples are generated until `n_accepted` samples get accepted,
     see [`generate_collinear_points_cpp`].
    :param n_accepted: The number of samples to be accepted for each data point. Only available if `method == "cpp"`.
    :param method: The method to generate the collinear points. Can be one of `{"py", "cpp"}`.
     The `"py"` version is faster but only allows to specify `n_samples`.
     The `"cpp"` version is slower but allows to specify both `n_accepted` and `n_samples`.
    :param report: Whether to report the number of samples.
    :param kwargs: Additional keyword arguments to be passed to the chosen method. `"py": {}`, `"cpp": {seed: None}`.
    :returns: (order) The order of `x`, with respect to axis 0, that yields the most accepted samples.
    """
    indices = [(i, j) for i in range(x.shape[0]) for j in range(x.shape[0]) if j > i]
    best_n = 0
    best_order = np.arange(x.shape[0], dtype=int)

    for i, j in indices:
        order = np.array(
            [
                i,
            ]
            + [k for k in range(x.shape[0]) if k != i and k != j]
            + [
                j,
            ]
        )
        _x, _cov = np.ascontiguousarray(x[order]), np.ascontiguousarray(cov[order])
        _, n_accepted, n_samples = generate_collinear_points(
            _x, _cov, n_samples=n_samples, n_accepted=n_accepted, method=method, report=report, **kwargs
        )
        if n_accepted > best_n:
            best_n = n_accepted
            best_order = order
    return best_order


def generate_collinear_points_py(
    mean: ndarray, cov: ndarray, n_samples: int_like | None = 100000, report: bool = False, **kwargs: Any
) -> tuple[ndarray, int, int]:
    r"""
    Randomly generate points $\vec{p}_i$ according to the given data vectors $\vec{\mu}_i\in\mathbb{R}^n$
    and covariance matrices $\mathbf{\Sigma}_i\in\mathbb{R}^{n\times n}$,
    under the condition that they are aligned on a straight line. This function uses pure Python.

    :param mean: The data vectors $\vec{\mu}_i$. Must have shape `(k, n)`, where `k` is the number of data points
     and `n` is the number of dimensions of each point.
    :param cov: The covariance matrices $\mathbf{\Sigma}_i$ of the data vectors. Must have shape `(k, n, n)`.
     Use [`covariance_matrix`] to construct covariance matrices.
    :param n_samples: The number of samples generated for each data point. If `None`, 100,000 samples are generated.
    :param report: Whether to report the number of samples.
    :param kwargs: Additional keyword arguments are accepted for compatibility with
     [`generate_collinear_points_cpp`] and ignored.
    :returns: (p, n_accepted, n_samples) The generated data vectors $\vec{p}_i$ with shape `(n_accepted, k, n)`
     and the number of accepted and generated samples.
    """
    if n_samples is None:
        n_samples = 100000
    n_samples = int(n_samples)

    p_0 = multivariate_normal.rvs(mean=mean[0], cov=cov[0], size=n_samples)
    p_1 = multivariate_normal.rvs(mean=mean[-1], cov=cov[-1], size=n_samples)
    dr = straight_direction(p_0, p_1, axis=-1)

    cov_inv = [np.linalg.inv(cov_i)[None, :, :] for cov_i in cov[1:-1]]
    a = [tools.transform(cov_i, dr, axis=-1) for cov_i in cov_inv]
    sigma = [1 / np.sqrt(np.sum(dr * a_i, axis=-1)) for a_i in a]
    t_0 = [-np.sum((p_0 - x_i) * a_i, axis=-1) * sigma_i**2 for x_i, a_i, sigma_i in zip(mean[1:-1], a, sigma)]
    t = [norm.rvs(loc=t_0_i, scale=sigma_i) for t_0_i, sigma_i in zip(t_0, sigma)]

    p_new = [p_0 + np.expand_dims(t_i, axis=-1) * dr for t_i in t]
    f = np.prod(
        [multivariate_normal.pdf(p_i, mean=x_i, cov=cov_i) for p_i, x_i, cov_i in zip(p_new, mean[1:-1], cov[1:-1])],
        axis=0,
    )
    g = np.prod([norm.pdf(t_i, loc=t_0_i, scale=sigma_i) for t_i, t_0_i, sigma_i in zip(t, t_0, sigma)], axis=0)

    if p_new:
        u = f / g
        u /= np.max(u)
        accepted = np.random.default_rng().random(size=n_samples) < u
        p = np.concatenate((np.expand_dims(p_0, 0), p_new, np.expand_dims(p_1, 0)), axis=0)
    else:
        accepted = np.full(n_samples, True, dtype=bool)
        p = np.concatenate((np.expand_dims(p_0, 0), np.expand_dims(p_1, 0)), axis=0)

    n_accepted = int(np.sum(accepted))
    if report:
        tools.print_colored("OKGREEN", f"Accepted samples: {n_accepted} / {n_samples}")

    p = np.transpose(p, axes=[1, 0, 2])
    return p[accepted], n_accepted, n_samples


def generate_collinear_points(
    x: ndarray,
    cov: ndarray,
    n_samples: int_like | None = None,
    n_accepted: int_like | None = None,
    method: str = "py",
    report: bool = True,
    **kwargs: Any,
) -> tuple[ndarray, int, int]:
    r"""
    Randomly generate points $\vec{p}_i$ according to the given data vectors $\vec{\mu}_i\in\mathbb{R}^n$
    and covariance matrices $\mathbf{\Sigma}_i\in\mathbb{R}^{n\times n}$,
    under the condition that they are aligned on a straight line. Specify `method` to switch between Python and C++.

    :param x: The data vectors $\vec{\mu}_i$. Must have shape `(k, n)`, where `k` is the number of data points
     and `n` is the number of dimensions of each point.
    :param cov: The covariance matrices $\mathbf{\Sigma}_i$ of the data vectors. Must have shape `(k, n, n)`.
     Use [`covariance_matrix`] to construct covariance matrices.
    :param n_samples: The number of samples generated for each data point.
     If `None` and `method == "cpp"`, samples are generated until `n_accepted` samples get accepted.
    :param n_accepted: The number of samples to be accepted for each data point. Only available if `method == "cpp"`.
    :param method: The method to generate the collinear points. Can be one of `{"py", "cpp"}`.
     The `"py"` version is faster but only allows to specify `n_samples`.
     The `"cpp"` version is slower but allows to specify both `n_accepted` and `n_samples`.
    :param report: Whether to report the number of samples.
    :param kwargs: Additional keyword arguments to be passed to the chosen method. `"py": {}`, `"cpp": {seed: None}`.
    :returns: (p, n_accepted, n_samples) The generated data vectors $\vec{p}_i$ with shape `(n_accepted, k, n)`
     and the number of accepted and generated samples.
    :raises ValueError: `method` must be in `{"py", "cpp"}`.
    """
    m = {"py", "cpp"}
    if method.lower() not in m:
        raise ValueError(f"`method` ({method}) must be in {m}.")
    if method == "py":
        return generate_collinear_points_py(x, cov, n_samples=n_samples, report=report)

    return generate_collinear_points_cpp(x, cov, n_samples=n_samples, n_accepted=n_accepted, report=report, **kwargs)


def linear_nd_monte_carlo(
    x: array_like,
    cov: array_like | None = None,
    axis: int_like | None = None,
    optimize_cov: bool = False,
    n_samples: int_like | None = None,
    n_accepted: int_like | None = None,
    optimize_sampling: bool = True,
    return_samples: bool = False,
    method: str = "py",
    report: bool = False,
    **kwargs: Any,
) -> Any:
    r"""
    Maximum likelihood Monte-Carlo sampling of a straight line through points $\vec{\mu}_i\in\mathbb{R}^n$
    with covariances $\mathbf{\Sigma}_i\in\mathbb{R}^{n\times n}$,
    assuming $n$-dimensional multivariate normal distributions $\mathcal{N}(\vec{\mu}_i, \mathbf{\Sigma}_i)$.

    The algorithm is described in the supplementary material of
    [<a href="https://doi.org/10.1103/PhysRevLett.115.053003">Gebert et al., Phys. Rev. Lett. 115, 053003 (2015)</a>].

    :param x: The data vectors $\vec{\mu}_i$. Must have shape `(k, n)`, where `k` is the number of data points
     and `n` is the number of dimensions of each point.
    :param cov: The covariance matrices $\mathbf{\Sigma}_i$ of the data vectors. Must have shape `(k, n, n)`.
     Use [`covariance_matrix`] to construct covariance matrices.
     If `None`, all covariance matrices are identity matrices.
    :param axis: The index of the vector component of the n-dimensional vectors that are fixed for fitting.
     This is required since a straight line in `n` dimensions is fully described by `2 * (n - 1)` parameters.
     If `None`, the best axis is determined from the data,
     and the direction vector of the straight line is normalized.
    :param optimize_cov: If `True`, the origin vector of the straight line is optimized
     to yield the smallest covariances.
    :param n_samples: The number of samples generated for each data point.
     If `None` and `method == "cpp"`, samples are generated until `n_accepted` samples get accepted.
    :param n_accepted: The number of samples to be accepted for each data point. Only available if `method == "cpp"`.
    :param optimize_sampling: Whether to optimize the data sampling for acceptance efficiency.
    :param return_samples: Whether to also return the generated points $\vec{p}_i$ with shape `(n_accepted, k, n)`.
    :param method: The method to generate the collinear points. Can be one of `{"py", "cpp"}`.
     The `"py"` version is faster but only allows to specify `n_samples`.
     The `"cpp"` version is slower but allows to specify both `n_accepted` and `n_samples`.
    :param report: Whether to print the result of the fit.
    :param kwargs: Additional keyword arguments to be passed to the chosen method. `"py": {}`, `"cpp": {seed: None}`.
    :returns: (popt, pcov, p) The optimized parameters and their covariances.
     If `return_samples` is `True`, also the generated points $\vec{p}_i$ are returned.
     The resulting shapes are `(2 * n, )`, `(2 * n, 2 * n)` and `(n_accepted, k, n)`.
    :raises ValueError: If `method` is not supported.
    """
    x = np.asarray(x, dtype=float)
    size, dim = x.shape

    cov = covariance_matrix(cov, k=size, n=dim)

    axis_was_none = False
    if axis is None:
        axis_was_none = True
        axis = int(np.argmax(np.abs(np.std(x, ddof=1, axis=0) / np.mean(x, axis=0)), axis=0))
    elif axis < 0:
        axis = int(axis + dim)
    axis = int(axis)

    if optimize_sampling:
        if report:
            tools.printh("\nOptimizing sampling:")
        order = _test_order_linear_nd_monte_carlo(x, cov, n_samples=100000, method=method, report=report)
    else:
        order = np.arange(size, dtype=int)

    inverse_order = np.array(
        [next(index for index, j in enumerate(order) if j == i) for i in range(order.size)], dtype=int
    )

    if report:
        tools.printh("\nGenerating samples:")
    _x, _cov = np.ascontiguousarray(x[order]), np.ascontiguousarray(cov[order])
    p, n_accepted, n_samples = generate_collinear_points(
        _x, _cov, n_accepted=n_accepted, n_samples=n_samples, method=method, report=report, **kwargs
    )

    p0 = p[:, 0, :]
    dr = p[:, -1, :] - p0
    p = p[:, inverse_order, :]

    s = 0.0
    if optimize_cov:

        def get_cov_t(t_opt: ndarray[floating], _p0: ndarray[floating], _dr: ndarray[floating]) -> ndarray[floating]:
            p0_opt = _p0.copy()
            _t = (t_opt - p0[:, [axis]]) / dr[:, [axis]]
            p0_opt += _t * _dr
            return np.cov(p0_opt.T, _dr.T)

        def cost_cov(t_opt: ndarray[floating], _p0: ndarray[floating], _dr: ndarray[floating]) -> floating:
            _pcov = get_cov_t(t_opt, _p0, _dr)
            return np.sum(np.triu(_pcov, k=1) ** 2)

        def get_cost_cov(_p0: ndarray[floating], _dr: ndarray[floating]) -> Callable[[ndarray[floating]], floating]:
            return lambda t_opt: cost_cov(t_opt, _p0, _dr)

        res = minimize(get_cost_cov(p0, dr), np.full(1, 0.1), method="BFGS")
        s = res.x[0]

    t = (s - p0[:, [axis]]) / dr[:, [axis]]
    p0 += t * dr

    if axis_was_none:
        dr /= tools.absolute(dr, axis=1)[:, None]
    else:
        dr /= dr[:, [axis]]

    popt = np.concatenate([np.mean(p0, axis=0), np.mean(dr, axis=0)], axis=0)
    pcov = np.cov(p0.T, dr.T)

    if return_samples:
        return popt, pcov, p
    return popt, pcov


def linear_monte_carlo(
    x: array_like,
    y: array_like,
    sigma_x: array_like | None = None,
    sigma_y: array_like | None = None,
    corr: array_like | None = None,
    optimize_cov: bool = True,
    n_samples: int_like | None = None,
    n_accepted: int_like | None = None,
    optimize_sampling: bool = True,
    return_samples: bool = False,
    method: str = "py",
    report: bool = True,
    **kwargs: Any,
) -> Any:
    r"""
    Maximum likelihood Monte-Carlo sampling of a straight line through points $\vec{\mu}_i\in\mathbb{R}^2$
    with covariances $\mathbf{\Sigma}_i\in\mathbb{R}^{2\times 2}$,
    assuming $2$-dimensional multivariate normal distributions $\mathcal{N}(\vec{\mu}_i, \mathbf{\Sigma}_i)$.

    This is a wrapper for the more general
    [`linear_nd_monte_carlo`] function. The algorithm is described in the supplementary material of
    [<a href="https://doi.org/10.1103/PhysRevLett.115.053003">Gebert et al., Phys. Rev. Lett. 115, 053003 (2015)</a>].

    :param x: The $x$ data.
    :param y: The $y$ data.
    :param sigma_x: The standard deviation $\sigma_x$ of the `x` data. If `None`, it is 1.
    :param sigma_y: The standard deviation $\sigma_y$ of the `y` data. If `None`, it is 1.
    :param corr: The correlation coefficients $\rho_{xy}$ between the `x` and `y` data. If `None`, it is 0.
    :param optimize_cov: If `True`, the origin vector of the straight line is optimized
     to yield the smallest covariances.
    :param n_samples: The number of samples generated for each data point.
     If `None` and `method == "cpp"`, samples are generated until `n_accepted` samples get accepted.
    :param n_accepted: The number of samples to be accepted for each data point. Only available if `method == "cpp"`.
    :param optimize_sampling: Whether to optimize the data sampling for acceptance efficiency.
    :param return_samples: Whether to also return the generated points $\vec{p}_i$ with shape `(n_accepted, k, 2)`.
    :param method: The method to generate the collinear points. Can be one of `{"py", "cpp"}`.
     The `"py"` version is faster but only allows to specify `n_samples`.
     The `"cpp"` version is slower but allows to specify both `n_accepted` and `n_samples`.
    :param report: Whether to print the result of the fit.
    :param kwargs: Additional keyword arguments are accepted for compatibility with the other linear regression
     algorithms and ignored.
    :returns: (popt, pcov, p) The best y-intercept and slope and their covariance matrix.
     If `return_samples` is `True`, also the generated points $\vec{p}_i$ are returned.
     The resulting shapes are `(2, )`, `(2, 2)` and `(n_accepted, k, 2)`.
    :raises ValueError: If `method` is not supported.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    sigma_x, sigma_y = tools.asarray_optional(sigma_x, dtype=float), tools.asarray_optional(sigma_y, dtype=float)

    mean = np.concatenate([x[:, None], y[:, None]], axis=1)

    if sigma_x is None:
        sigma_x = np.ones_like(x)
    if sigma_y is None:
        sigma_y = np.ones_like(y)
    sigma = np.concatenate([sigma_x[:, None], sigma_y[:, None]], axis=1)

    cov = covariance_matrix(sigma=sigma, corr=corr)

    result = linear_nd_monte_carlo(
        mean,
        cov=cov,
        axis=0,
        optimize_cov=optimize_cov,
        n_accepted=n_accepted,
        n_samples=n_samples,
        optimize_sampling=optimize_sampling,
        return_samples=True,
        method=method,
        report=report,
    )
    popt, pcov = result[:2]
    i = np.array([1, 3], dtype=int)
    popt, pcov = popt[i], pcov[np.ix_(i, i)]

    if return_samples:
        assert len(result) == 3
        return popt, pcov, result[2]
    return popt, pcov


def linear_alpha_fit(
    x: array_like,
    y: array_like,
    sigma_x: array_like | None = None,
    sigma_y: array_like | None = None,
    corr: array_like | None = None,
    func: Callable | str = york_fit,
    alpha: scalar = 0,
    find_alpha: bool = True,
    report: bool = False,
    show: bool = False,
    **kwargs,
) -> tuple[ndarray, ndarray, float]:
    r"""
    Wrapper for the $2$-dimensional linear regression algorithms [`york_fit`], [`linear_fit`]
    and [`linear_monte_carlo`] that optimizes an additional $x$-axis shift $\alpha$
    for the minimum correlation coefficient between the $y$-intercept and the slope of the straight line

    $$
    f(x) = a + b(x - \alpha).
    $$

    :param x: The $x$ data.
    :param y: The $y$ data.
    :param sigma_x: The standard deviation $\sigma_x$ of the `x` data.
    :param sigma_y: The standard deviation $\sigma_y$ of the `y` data.
    :param corr: The correlation coefficients $\rho_{xy}$ between the `x` and `y` data.
    :param func: The fitting routine. Supports any of `{"york_fit", "linear_fit", "linear_monte_carlo"}`.
    :param alpha: An $x$-axis offset $\alpha$ to reduce the correlation coefficient
     between the $y$-intercept and the slope.
    :param find_alpha: Whether to search for the best `alpha`. Uses the given `alpha` as a starting point.
     May not give the desired result if `alpha` was initialized too far from its optimal value.
    :param report: Whether to print the result of the fit.
    :param show: Whether to plot the fit result.
    :param kwargs: Additional keyword arguments are passed to the fitting routine.
    :returns: (popt, pcov, alpha) The best y-intercept and slope, their covariance matrix and the final `alpha`.
    :raises ValueError: If `func` is not supported.
    """
    n = tools.floor_log10(alpha)
    x = np.asarray(x, dtype=float)

    funcs = {"york_fit", "linear_fit", "linear_monte_carlo"}
    if isinstance(func, str) and func in funcs:
        _func: Callable = eval(func)
    elif callable(func) and func.__name__ in funcs:
        _func = func
    else:
        raise ValueError(f"`func` must be in {funcs} but is {func}.")

    def cost(x0: ndarray) -> float:
        _, c = _func(x - x0[0], y, sigma_x=sigma_x, sigma_y=sigma_y, corr=corr, show=False, **kwargs)
        return c[0, 1] ** 2 * 10.0 ** (n + 1)

    if find_alpha:
        alpha = minimize(
            cost,
            np.array([alpha]),
            method="Nelder-Mead",
            options={"xatol": 1.0, "fatol": float(0.01**2 * 10.0 ** (n + 1))},
        ).x[0]

    popt, pcov = _func(x - alpha, y, sigma_x=sigma_x, sigma_y=sigma_y, corr=corr, report=report, show=show, **kwargs)
    if report:
        print(f"alpha: {alpha}")
    return popt, pcov, float(alpha)


def odr_fit(
    f: Callable,
    x: scalar_nd,
    y: scalar_nd,
    sigma_x: scalar_nd | None = None,
    sigma_y: scalar_nd | None = None,
    p0: scalar_nd | None = None,
    delta0: scalar_nd | None = None,
    p0_fixed: scalar_nd | None = None,
    report: bool = False,
    **kwargs,
) -> tuple[ndarray, ndarray]:
    r"""
    Fit a function to data with uncertainties in both $x$ and $y$ using the orthogonal distance regression (ODR)
    routine <a href="https://hugomvale.github.io/odrpack-python/">`odrpack.odr_fit`</a> with the syntax of
    <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html">
    `scipy.optimize.curve_fit`</a>. Requires the optional dependency
    <a href="https://pypi.org/project/odrpack/">`odrpack`</a>, which can be installed with `pip install qspec[odr]`.

    :param f: The model function $f$ to fit to the data.
    :param x: The $x$ data.
    :param y: The $y$ data.
    :param sigma_x: The $1\sigma$ uncertainty of the `x` data.
    :param sigma_y: The $1\sigma$ uncertainty of the `y` data.
    :param p0: A numpy array or an Iterable of the initial guesses for the parameters.
     Must have at least the same length as the minimum number of parameters required by the function `f`.
     If `p0` is `None`, 1 is taken as an initial guess for all non-keyword parameters.
    :param delta0: A numpy array or an Iterable of the initial guesses for the errors of the `x` data.
     Must have the same shape as `x`. If `None`, the errors are initialized with 0.
    :param p0_fixed: A numpy array or an Iterable of `bool` values specifying whether to fix a parameter.
     Must have the same length as `p0`.
    :param report: Whether to print the result of the fit.
    :param kwargs: Additional keyword arguments are passed to
     <a href="https://hugomvale.github.io/odrpack-python/">`odrpack.odr_fit`</a>.
    :returns: (popt, pcov) The optimal parameters and their covariance matrix.
    :raises ImportError: If `odrpack` is not installed.
    :raises ValueError: If `f` has no parameters to optimize or if `p0_fixed` and `p0` have different shapes.
    """
    try:
        import odrpack
    except ImportError as e:
        raise ImportError(
            "`odr_fit` requires the optional dependency 'odrpack'. Install it with `pip install qspec[odr]`."
        ) from e

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    weight_x = None if sigma_x is None else 1.0 / np.asarray(sigma_x, dtype=float) ** 2
    weight_y = None if sigma_y is None else 1.0 / np.asarray(sigma_y, dtype=float) ** 2
    if p0 is None:
        arg_spec = inspect.getfullargspec(f)
        n = len(arg_spec.args) - 1
        if (n < 1 and arg_spec.defaults is None) or (arg_spec.defaults is not None and n - len(arg_spec.defaults) < 1):
            raise ValueError("`f` must have at least one parameter that can be optimized.")
        p0 = [
            1.0,
        ] * n
        if arg_spec.defaults is not None:
            p0 = [
                1.0,
            ] * (n - len(arg_spec.defaults))
    p0 = np.asarray(p0, dtype=float)

    fix_beta = None
    if p0_fixed is not None:
        fix_beta = np.asarray(p0_fixed, dtype=bool)
        if fix_beta.shape != p0.shape:
            raise ValueError("`p0_fixed` must have the same shape as `p0`.")

    out = odrpack.odr_fit(
        lambda x_i, beta: f(x_i, *beta),
        x,
        y,
        p0,
        weight_x=weight_x,
        weight_y=weight_y,
        fix_beta=fix_beta,
        delta0=None if delta0 is None else np.asarray(delta0, dtype=float),
        **kwargs,
    )

    if report:
        tools.printh("odr_fit result:")
        for i, (val, err) in enumerate(zip(out.beta, np.sqrt(np.diag(out.cov_beta)))):
            print(f"{i}: {val} +/- {err}")
    return out.beta, out.cov_beta


def curve_fit(
    f: Callable,
    x: array_like | object,
    y: array_like,
    p0: array_like | None = None,
    p0_fixed: array_like | None = None,
    sigma: array_like | Callable | None = None,
    absolute_sigma: bool = False,
    check_finite: bool = True,
    bounds: tuple[array_like, array_like] | Bounds = (-np.inf, np.inf),
    method: Literal["trf", "dogbox", "lm"] | None = None,
    jac: Callable | Literal["2-point", "3-point", "cs"] | None = None,
    full_output: bool = False,
    report: bool = False,
    **kwargs,
) -> Any:
    r"""
    Use non-linear least squares to fit a function $f$ to data. Assumes `y = f(x, *params) + eps`.
    This is a reimplementation of
    <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html">
    `scipy.optimize.curve_fit`</a> with the additional features of fixing parameters with `p0_fixed`
    and the option to use the parameter `sigma` dynamically as a function `g(x, y, f(x, *params), *params) -> sigma`.

    :param f: The model function $f$ to fit to the data.
    :param x: The $x$ data.
    :param y: The $y$ data.
    :param p0: A numpy array or an Iterable of the initial guesses for the parameters.
     Must have at least the same length as the minimum number of parameters required by the function `f`.
     If `p0` is `None`, 1 is taken as an initial guess for all non-keyword parameters.
    :param p0_fixed: A numpy array or an Iterable of `bool` values specifying whether to fix a parameter.
     Must have the same length as `p0`.
    :param sigma: The $1\sigma$ uncertainty of the `y` data.
     This can also be a function $g$ such that `g(x, y, f(x, *params), *params) -> sigma`.
    :param absolute_sigma: Whether `sigma` is used in an absolute sense (`True`) and the covariance matrix
     reflects these absolute values, or only the relative magnitudes of `sigma` matter (`False`)
     and the covariance matrix is scaled by the reduced $\chi^2$.
    :param check_finite: See
     <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html">
     `scipy.optimize.curve_fit`</a>.
    :param bounds: See
     <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html">
     `scipy.optimize.curve_fit`</a>.
    :param method: See
     <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html">
     `scipy.optimize.curve_fit`</a>.
    :param jac: See
     <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html">
     `scipy.optimize.curve_fit`</a>. Must not be callable if `sigma` is callable.
    :param full_output: See
     <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html">
     `scipy.optimize.curve_fit`</a>.
    :param report: Whether to print the result of the fit.
    :param kwargs: See
     <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html">
     `scipy.optimize.curve_fit`</a>.
    :returns: (popt, pcov) The optimal parameters and their covariance matrix.
     Additional output if `full_output` is `True`.
     See <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html">
     `scipy.optimize.curve_fit`</a>. An `OptimizeWarning` is issued
     if the covariance of the parameters cannot be estimated.
    :raises ValueError: If either `x` or `y` contain NaNs, if `sigma` has an invalid shape
     or if incompatible options are used.
    :raises RuntimeError: If the least-squares minimization fails.
    :raises TypeError: If the number of parameters exceeds the number of data points for the method `"lm"`.
    """
    if p0 is None:
        args = [
            p
            for p in inspect.signature(f).parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(args) < 2:
            raise ValueError("Unable to determine number of fit parameters.")
        n = len(args) - 1
    else:
        p0 = np.atleast_1d(np.asarray(p0, dtype=float))
        n = p0.size

    p0_fixed = np.zeros(n, dtype=bool) if p0_fixed is None else np.atleast_1d(p0_fixed)

    if isinstance(bounds, Bounds):
        lb, ub = _prepare_bounds((bounds.lb, bounds.ub), n)
    else:
        lb, ub = _prepare_bounds(bounds, n)
    if p0 is None:
        p0 = _initialize_feasible(lb, ub)

    p0_free = p0[~p0_fixed]
    lb, ub = lb[~p0_fixed], ub[~p0_fixed]
    func = _wrap_func_pars(f, p0, p0_fixed)

    bounded_problem = np.any((lb > -np.inf) | (ub < np.inf))
    if method is None:
        method = "trf" if bounded_problem else "lm"

    if method == "lm" and bounded_problem:
        raise ValueError("Method 'lm' only works for unconstrained problems. Use 'trf' or 'dogbox' instead.")

    y = np.asarray_chkfinite(y, float) if check_finite else np.asarray(y, float)

    if isinstance(x, (list, tuple, np.ndarray)):
        x = np.asarray_chkfinite(x, float) if check_finite else np.asarray(x, float)

    if y.size == 0:
        raise ValueError("`y` must not be empty!")

    transform: Callable | ndarray | None = None
    if sigma is not None:
        if callable(sigma):
            transform = _wrap_func_sigma(sigma, p0, p0_fixed)
        else:
            sigma = np.asarray(sigma)

            if sigma.shape == (y.size,):
                transform = 1.0 / sigma
            elif sigma.shape == (y.size, y.size):
                try:
                    transform = cholesky(sigma, lower=True)
                except LinAlgError as e:
                    raise ValueError("`sigma` must be positive definite.") from e
            else:
                raise ValueError("`sigma` has incorrect shape.")

    func = _wrap_func(func, x, y, transform)
    if callable(jac):
        if callable(transform):
            raise ValueError("`jac` must not be callable if `sigma` is callable.")
        jac = _wrap_jac(jac, x, transform)

    if "args" in kwargs:
        raise ValueError("'args' is not a supported keyword argument.")

    ier: int
    if method == "lm":
        if y.size != 1 and n > y.size:
            raise TypeError(f"The number of func parameters={n} must not exceed the number of data points={y.size}")
        if isinstance(jac, str):
            raise ValueError("Method 'lm' does not support string values for `jac`.")

        res = leastsq(func, p0_free, Dfun=jac, full_output=True, **kwargs)
        assert len(res) == 5
        popt, pcov, _infodict, errmsg, ier = res
        infodict: dict[str, Any] = dict(_infodict)

        ysize = len(infodict["fvec"])
        cost = np.sum(infodict["fvec"] ** 2)
        if ier not in [1, 2, 3, 4]:
            raise RuntimeError("Optimal parameters not found: " + errmsg)
    else:
        if "max_nfev" not in kwargs:
            kwargs["max_nfev"] = kwargs.pop("maxfev", None)

        if jac is None:
            jac = "2-point"
        res_ls = least_squares(func, p0_free, jac=jac, bounds=(lb, ub), method=method, **kwargs)

        if not res_ls.success:
            raise RuntimeError("Optimal parameters not found: " + res_ls.message)

        infodict = dict(nfev=res_ls.nfev, fvec=res_ls.fun)
        ier = res_ls.status
        errmsg = res_ls.message

        ysize = len(res_ls.fun)
        cost = 2 * res_ls.cost
        popt = res_ls.x

        _, s, vt = svd(np.asarray(res_ls.jac), full_matrices=False)
        threshold = np.finfo(float).eps * max(res_ls.jac.shape) * s[0]
        s = s[s > threshold]
        vt = vt[: s.size]
        pcov = np.dot(vt.T / s**2, vt)

    warn_cov = False
    s_sq = cost / (ysize - p0_free.size) if ysize > p0_free.size else np.inf
    if pcov is None:
        pcov = np.zeros((len(popt), len(popt)), dtype=float)
        pcov.fill(np.inf)
        warn_cov = True
    elif not absolute_sigma:
        if not np.isinf(s_sq):
            pcov = pcov * s_sq
        else:
            pcov.fill(np.inf)
            warn_cov = True

    popt_ret, pcov_ret = p0.copy(), np.zeros((p0.size, p0.size), dtype=float)
    popt_ret[~p0_fixed] = popt
    pcov_mask = ~(p0_fixed[np.newaxis, :] + p0_fixed[:, np.newaxis])
    pcov_ret[pcov_mask] = pcov.flatten()

    if report:
        digits = int(np.floor(np.log10(np.abs(p0.size)))) + 1
        tools.printh("curve_fit result:")
        for i, (val, err, fix) in enumerate(zip(popt_ret, np.sqrt(np.diag(pcov_ret)), p0_fixed)):
            print(f"{str(i).zfill(digits)}: {val} +/- {err} ({'fixed' if fix else 'free'})")
        print(f"red. chi2: {s_sq}")

    if warn_cov:
        warnings.warn("Covariance of the parameters could not be estimated", category=OptimizeWarning)

    if full_output:
        return popt_ret, pcov_ret, infodict, errmsg, ier

    return popt_ret, pcov_ret


def _mass_factor_array(m0: ndarray, m1: ndarray, m0_d: ndarray, m1_d: ndarray) -> tuple[ndarray, ndarray]:
    m_rel = m0 - m1
    mask = m_rel == 0
    m_rel[mask] = 1

    mu = m0 * m1 / m_rel
    mu_d = ((mu / m0 - mu / m_rel) * m0_d) ** 2
    mu_d += ((mu / m1 + mu / m_rel) * m1_d) ** 2
    mu_d = np.sqrt(mu_d)

    mu[mask] = np.inf
    mu_d[mask] = 0
    return mu, mu_d


class King:
    FIT_NOT_DONE_ERROR = "No fit was performed yet. Use one of the fitting options {'King.fit', 'King.fit_nd'}."
    """The error message if a method requires a fit that was not performed yet."""
    NOT_ENOUGH_DATA_POINTS_ERROR = (
        "Cannot perform a linear fit with less than 3 data points."
        " Only {} isotopes and {} reference isotopes were specified."
    )
    """The error message if less than 3 data points are given."""
    NO_X_ABS_ERROR = "To fit, either specify `{}` or `King.x_abs`. Both are `None`."
    """The error message if neither the data nor `King.x_abs` are given."""

    a: ndarray[integer]
    """The mass numbers of the considered isotopes."""
    m: ndarray[floating]
    """The masses and their uncertainties of the isotopes (u)."""
    m_sub: ndarray[floating]
    """The masses and their uncertainties of the isotopes after subtracting `King.subtract_electrons` electrons (u)."""
    m_mod: ndarray[floating]
    """The mass factors and their uncertainties of all isotope pairs, see [`qspec.physics.mass_factor`]."""
    x_abs: ndarray[floating] | None
    """The absolute values of the data vectors and their standard deviations."""
    subtract_electrons: float
    """The number of electron masses subtracted from the isotope masses."""
    n_samples: int
    """The number of generated samples for the Monte-Carlo routines."""
    element_label: str
    """The label of the element."""
    n: int
    """The number of dimensions of the King plot."""
    nd: bool
    """Whether the last fit was performed with [`King.fit_nd`]."""
    a_fit: ndarray[integer] | None
    """The mass numbers of the isotopes of the last fit."""
    a_ref: ndarray[integer] | None
    """The mass numbers of the reference isotopes of the last fit."""
    x: ndarray[floating] | None
    """The data vectors and their standard deviations of the last fit."""
    x_mod: ndarray[floating] | None
    """The modified data vectors and their standard deviations of the last fit."""
    corr: ndarray[floating] | None
    """The correlation coefficients of the modified $x$ and $y$ data of the last 2-dimensional fit."""
    cov: ndarray[floating] | None
    """The covariance matrices of the modified data vectors of the last $n$-dimensional fit."""
    popt: ndarray[floating] | None
    """The optimized parameters of the last fit."""
    pcov: ndarray[floating] | None
    """The covariance matrix of the optimized parameters of the last fit."""
    alpha: float | None
    """The $x$-axis offset $\\alpha$ of the last fit."""
    axis: int
    """The vector component used for the parameterization in the last $n$-dimensional fit."""

    def __init__(
        self,
        a: array_like,
        m: array_like,
        x_abs: array_like | None = None,
        subtract_electrons: scalar = 0.0,
        n_samples: int = 100000,
        element_label: str | None = None,
    ) -> None:
        r"""
        A class for $n$-dimensional King plots, encapsulating the linear regression algorithms of the
        [`qspec.analyze`] module. Call [`King.fit`] or [`King.fit_nd`] for $2$- or $n$-dimensional fitting.

        :param a: An iterable of the mass numbers $A$ of the considered isotopes with shape `(k, )`,
         where `k` is the number of isotopes.
        :param m: The masses and their uncertainties of the isotopes `a` (u). `m` must have shape `(k, 2)`.
        :param x_abs: The absolute values of the data vectors and their standard deviations
         corresponding to the mass numbers `a`. If given, the `x` and `y` parameters can be omitted in
         [`King.fit`] and [`King.fit_nd`], and are determined automatically as the shift between the mass numbers `a`
         and `a_ref`. Must have shape `(k, n, 2)` where `n` is the number of dimensions of the King plot.
        :param subtract_electrons: The number of electron masses that should be subtracted from the specified isotope
         masses. `subtract_electrons` can be a `float` if the ionization energy shall be considered.
        :param n_samples: The number of generated samples for the Monte-Carlo routines.
        :param element_label: The label of the element to enhance the printed and plotted information.
        :raises ValueError: If the mass numbers `a` are not unique.
        """
        self.a = np.asarray(a, dtype=int)
        if len(set(self.a)) < self.a.size:
            raise ValueError("The given mass numbers must be unique.")

        self.m = np.asarray(m, dtype=float)
        self.x_abs = tools.asarray_optional(x_abs, dtype=float)
        self.subtract_electrons = float(subtract_electrons)
        self.n_samples = int(n_samples)
        self.element_label = "" if element_label is None else str(element_label)

        self.n = 2 if self.x_abs is None else self.x_abs.shape[1]

        self.m_sub = self.m.copy()
        self.m_sub[:, 0] -= (self.subtract_electrons - 1) * me_u
        self.m_sub[:, 1] = np.sqrt(self.m[:, 1] ** 2 + ((self.subtract_electrons - 1) * me_u_d) ** 2)

        m_mod = mass_factor(
            self.m_sub[None, :, 0], self.m_sub[:, None, 0], self.m_sub[None, :, 1], self.m_sub[:, None, 1]
        )
        self.m_mod = np.transpose(m_mod, axes=[1, 2, 0])

        self.nd = False
        self.a_fit = None
        self.a_ref = None
        self.x = None
        self.x_mod = None
        self.corr = None
        self.cov = None
        self.popt = None
        self.pcov = None
        self.alpha = None
        self.axis = 0

    def _get_i(self, a: Iterable[int_like], a_ref: Iterable[int_like]) -> tuple[ndarray[integer], ndarray[integer]]:
        i = np.array([self.a.tolist().index(_a) for _a in a])
        i_ref = np.array([self.a.tolist().index(_a) for _a in a_ref])
        return i, i_ref

    def _sample_xmod(self) -> ndarray[floating]:
        if self.a_fit is None or self.a_ref is None or self.x is None:
            raise ValueError(self.FIT_NOT_DONE_ERROR)

        i, i_ref = self._get_i(self.a_fit, self.a_ref)

        m = norm.rvs(loc=self.m_sub[:, 0], scale=self.m_sub[:, 1], size=(self.n_samples, self.m_sub.shape[0]))
        m = np.asarray(m, dtype=float).T

        x = norm.rvs(
            loc=self.x[:, :, 0], scale=self.x[:, :, 1], size=(self.n_samples, self.x.shape[0], self.x.shape[1])
        )
        x = np.transpose(x, axes=[1, 2, 0])

        m_mod = mass_factor(m[i], m[i_ref], 0.0, 0.0)[0]
        return m_mod[:, None, :] * x

    def _cov_mc(self) -> ndarray[floating]:
        """
        :returns: The covariance matrices of the data vectors by sampling the data.
        """
        x_mod = self._sample_xmod()
        return np.array([np.cov(x_mod_i) for x_mod_i in x_mod], dtype=float)

    def _corr_mc(self) -> ndarray[floating]:
        """
        :returns: The correlation coefficients of the modified x- and y-data points by sampling the data.
        """
        x_mod = self._sample_xmod()
        return np.array([np.corrcoef(x_mod_i) for x_mod_i in x_mod], dtype=float)

    def _cov(self) -> ndarray[floating]:
        if self.a_fit is None or self.a_ref is None or self.x is None:
            raise ValueError(self.FIT_NOT_DONE_ERROR)

        i, i_ref = self._get_i(self.a_fit, self.a_ref)
        dim = self.x.shape[1]
        m_mod = self.m_mod[i_ref, i, :]
        cov_x = [np.diag(np.insert(_x[:, 1], 0, _m_mod[1])) ** 2 for _m_mod, _x in zip(m_mod, self.x)]
        jac = [
            np.concatenate([_x[:, 0, None], np.diag(np.full(dim, _m_mod[0]))], axis=1)
            for _m_mod, _x in zip(m_mod, self.x)
        ]
        return np.array([_jac @ (_cov_x @ _jac.T) for _jac, _cov_x in zip(jac, cov_x)])

    def _corr(self) -> ndarray[floating]:
        return np.array([_cov / np.sqrt(np.diag(_cov)[:, None] * np.diag(_cov)[None, :]) for _cov in self._cov()])

    def fit(
        self,
        a: array_like,
        a_ref: array_like,
        x: array_like | None = None,
        y: array_like | None = None,
        xy: tuple[int, int] = (0, 1),
        func: str | Callable = york_fit,
        alpha: scalar = 0,
        find_alpha: bool = False,
        show: bool = True,
        **kwargs,
    ) -> tuple[ndarray, ndarray]:
        r"""
        Perform $2$-dimensional linear regression to create a King plot, see [`linear_alpha_fit`].
        The data are modified with the mass factors $\mu$, see [`qspec.physics.mass_factor`].
        Choose between `{"york_fit" (default), "linear_fit", "linear_monte_carlo"}` for the fit routine.
        Use a parameter `alpha` to optimize the correlation coefficient between the $y$-intercept and the slope.

        :param a: An iterable of the mass numbers $A$ of the used isotopes.
        :param a_ref: An iterable of the mass numbers $A_\mathrm{ref}$ of the used reference isotopes.
        :param x: The $x$ data and their standard deviations as shape `(len(a), 2)` arrays.
         Either the differences of mean-square nuclear charge radii $\delta\langle r^2\rangle^{A,A_\mathrm{ref}}$
         or $\Lambda^{A,A_\mathrm{ref}}$ (fm$^2$), plotted with `mode="radii"`,
         or isotope shifts $\delta\nu_x^{A,A_\mathrm{ref}}$ (MHz), plotted with `mode="shifts"`, see [`King.plot`].
         If `None`, `x` is inherited from `King.x_abs`.
        :param y: The isotope shifts $\delta\nu_y^{A,A_\mathrm{ref}}$ and their standard deviations
         as shape `(len(a), 2)` arrays (MHz).
         If `None`, `y` is inherited from `King.x_abs`.
        :param xy: A 2-tuple of indices `(ix, iy)`, used to select the two plot axes from `King.x_abs`.
         Only used if `x` or `y` is not specified. The default value is `(0, 1)`,
         fitting the second against the first axis.
        :param func: The fitting routine. Must be one of `{"york_fit" (default), "linear_fit", "linear_monte_carlo"}`.
        :param alpha: An $x$-axis offset $\alpha$ to reduce the correlation coefficient
         between the $y$-intercept and the slope. Expected unit: (u fm$^2$) or (u MHz).
        :param find_alpha: Whether to search for the best `alpha`. Uses the given `alpha` as a starting point.
         May not give the desired result if `alpha` was initialized too far from its optimal value.
        :param show: Whether to plot the fit result.
        :param kwargs: Additional keyword arguments are passed to [`King.plot`].
        :returns: (popt, pcov) The best y-intercept and slope and their covariance matrix.
         The final `alpha` can be accessed through `King.alpha`.
        :raises ValueError: If less than 3 data points are given, if neither `x` (`y`) nor `King.x_abs` are given,
         or if `func` is not supported.
        """
        self.nd = False
        self.a_fit, self.a_ref = np.asarray(a, dtype=int).flatten(), np.asarray(a_ref, dtype=int).flatten()

        if self.a_fit.size < 3 or self.a_ref.size < 3:
            raise ValueError(self.NOT_ENOUGH_DATA_POINTS_ERROR.format(self.a_fit.size, self.a_ref.size))

        i, i_ref = self._get_i(self.a_fit, self.a_ref)

        if xy is None:
            xy = (0, 1)

        self.x = np.zeros((self.a_fit.size, 2, 2), dtype=float)
        if x is None:
            if self.x_abs is None:
                raise ValueError(self.NO_X_ABS_ERROR.format("x"))

            self.x[:, 0, 0] = self.x_abs[i, xy[0], 0] - self.x_abs[i_ref, xy[0], 0]
            self.x[:, 0, 1] = np.sqrt(self.x_abs[i, xy[0], 1] ** 2 + self.x_abs[i_ref, xy[0], 1] ** 2)
        else:
            self.x[:, 0, :] = np.asarray(x, dtype=float)
        if y is None:
            if self.x_abs is None:
                raise ValueError(self.NO_X_ABS_ERROR.format("y"))

            self.x[:, 1, 0] = self.x_abs[i, xy[1], 0] - self.x_abs[i_ref, xy[1], 0]
            self.x[:, 1, 1] = np.sqrt(self.x_abs[i, xy[1], 1] ** 2 + self.x_abs[i_ref, xy[1], 1] ** 2)
        else:
            self.x[:, 1, :] = np.asarray(y, dtype=float)

        m_mod = self.m_mod[i_ref, i, :]
        m_mod = m_mod[:, None, :]

        self.x_mod = np.array(
            [
                self.x[:, :, 0] * m_mod[:, :, 0],
                straight_x_std(self.x[:, :, 0], m_mod[:, :, 0], self.x[:, :, 1], 0.0, m_mod[:, :, 1], 0.0),
            ]
        )
        self.x_mod = np.transpose(self.x_mod, axes=[1, 2, 0])

        self.corr = self._corr()[:, 0, 1]
        self.popt, self.pcov, self.alpha = linear_alpha_fit(
            self.x_mod[:, 0, 0],
            self.x_mod[:, 1, 0],
            sigma_x=self.x_mod[:, 0, 1],
            sigma_y=self.x_mod[:, 1, 1],
            corr=self.corr,
            func=func,
            alpha=alpha,
            find_alpha=find_alpha,
            show=False,
        )

        if show:
            self.plot(**kwargs)

        return self.popt, self.pcov

    def fit_nd(
        self,
        a: scalar_nd,
        a_ref: scalar_nd,
        x: scalar_nd | None = None,
        axis: int = 0,
        optimize_cov: bool = False,
        func: Callable | str = linear_nd_fit,
        show: bool = True,
        **kwargs,
    ) -> tuple[ndarray, ndarray]:
        r"""
        Perform $n$-dimensional linear regression to create a King plot.
        The data are modified with the mass factors $\mu$, see [`qspec.physics.mass_factor`].
        Choose between `{"linear_nd_fit" (default), "linear_nd_monte_carlo"}` for the fit routine.

        :param a: An iterable of the mass numbers $A$ of the used isotopes.
        :param a_ref: An iterable of the mass numbers $A_\mathrm{ref}$ of the used reference isotopes.
        :param x: The data as an iterable of vectors with standard deviations of shape `(k, n, 2)`, where `k` is the
         number of data points/isotopes and `n` is the dimension of each vector.
         If `None`, `x` is inherited from `King.x_abs`.
        :param axis: The vector component to use for the parameterization.
         For example, a King plot with the isotope shifts of two transitions `["D1", "D2"]`
         yields the slope $F_\mathrm{D2} / F_\mathrm{D1}$ if `axis == 0`.
        :param optimize_cov: If `True`, the origin vector of the straight line is optimized
         to yield the smallest covariances.
        :param func: The fitting routine. Must be one of `{"linear_nd_fit" (default), "linear_nd_monte_carlo"}`.
        :param show: Whether to plot the fit result.
        :param kwargs: Additional keyword arguments are passed to `func` and [`King.plot`].
        :returns: (popt, pcov) The optimized parameters and their covariances.
         The resulting shapes are `(2 * n, )` and `(2 * n, 2 * n)`.
        :raises ValueError: If neither `x` nor `King.x_abs` are given or if `func` is not supported.
        """
        self.nd = True
        self.a_fit, self.a_ref = np.asarray(a, dtype=int), np.asarray(a_ref, dtype=int)
        i, i_ref = self._get_i(self.a_fit, self.a_ref)

        if x is None:
            if self.x_abs is None:
                raise ValueError(self.NO_X_ABS_ERROR.format("x"))

            self.x = np.zeros((self.a_fit.size, self.x_abs.shape[1], 2), dtype=float)
            self.x[:, :, 0] = self.x_abs[i, :, 0] - self.x_abs[i_ref, :, 0]
            self.x[:, :, 1] = np.sqrt(self.x_abs[i, :, 1] ** 2 + self.x_abs[i_ref, :, 1] ** 2)
        else:
            self.x = np.asarray(x, dtype=float)

        m_mod = self.m_mod[i_ref, i, :]
        m_mod = m_mod[:, None, :]

        self.x_mod = np.array(
            [
                self.x[:, :, 0] * m_mod[:, :, 0],
                straight_x_std(self.x[:, :, 0], m_mod[:, :, 0], self.x[:, :, 1], 0.0, m_mod[:, :, 1], 0.0),
            ]
        )
        self.x_mod = np.transpose(self.x_mod, axes=[1, 2, 0])

        funcs = {"linear_nd_fit", "linear_nd_monte_carlo"}
        if isinstance(func, str) and func in funcs:
            _func: Callable = eval(func)
        elif callable(func) and func.__name__ in funcs:
            _func = func
        else:
            raise ValueError(f"`func` must be in {funcs} but is {func}.")

        self.cov = self._cov()
        self.popt, self.pcov, *_ = _func(
            self.x_mod[:, :, 0], cov=self.cov, axis=axis, optimize_cov=optimize_cov, **kwargs
        )
        assert isinstance(self.popt, np.ndarray)
        assert isinstance(self.pcov, np.ndarray)

        self.alpha = self.popt[axis]
        self.axis = axis

        if show:
            self.plot(**kwargs)

        return self.popt, self.pcov

    def _get_gradient(self, x: ndarray, m_mod: ndarray, index: int, axis: int) -> ndarray[floating]:
        if self.popt is None:
            raise ValueError(self.FIT_NOT_DONE_ERROR)

        dim = self.popt.size // 2
        _a, _b = self.popt[:dim], self.popt[dim:]
        br = _b[index] / _b[axis]
        t = (x - _a[axis] / m_mod) / _b[axis]

        yi = [br]
        ai = [
            0 if index == axis or index != d != axis else (1 / m_mod if d == index else -br / m_mod) for d in range(dim)
        ]
        bi = [0 if index == axis or index != d != axis else (t if d == index else -br * t) for d in range(dim)]
        mu = [0 if index == axis else (br * _a[axis] - _a[index]) / m_mod**2]
        return np.array(ai + bi + mu + yi, dtype=float)

    def _get_gradient_mod(self, x: ndarray, index: int, axis: int) -> ndarray[floating]:
        if self.popt is None:
            raise ValueError(self.FIT_NOT_DONE_ERROR)

        dim = self.popt.size // 2
        _a, _b = self.popt[:dim], self.popt[dim:]
        br = _b[index] / _b[axis]
        t = (x - _a[axis]) / _b[axis]

        yi = [br]
        ai = [0 if index == axis or index != d != axis else (1 if d == index else -br) for d in range(dim)]
        bi = [_ai * t for _ai in ai]
        return np.array(ai + bi + yi, dtype=float)

    def get_unmodified(
        self, a: scalar_nd, a_ref: scalar_nd, x: scalar_nd, axis: int = 0, show: bool = False, **kwargs
    ) -> tuple[ndarray, ndarray, ndarray]:
        r"""
        Calculate unknown isotope shifts/charge radii by using the King plot results.

        :param a: An iterable of mass numbers $A$, corresponding to the isotopes of the given `x` values.
        :param a_ref: An iterable of mass numbers $A_\mathrm{ref}$,
         corresponding to reference isotopes of the given `x` values.
        :param x: The unmodified input values for the given mass numbers `a` and `a_ref`.
         These could be the unmodified isotope shifts $\delta\nu_x^{A,A_\mathrm{ref}}$. Must have shape `(len(a), 2)`.
        :param axis: The vector component corresponding to the given input values.
         For example, in a $2$-dimensional fit with components `["D1", "D2"]`,
         set `axis=0` to use the isotope shifts of the D1-line to calculate those of the D2-line.
        :param show: Whether to draw the calculated values in the King plot.
        :param kwargs: Additional keyword arguments are passed to [`King.plot`].
        :returns: (y, cov, cov_stat) The unmodified $y$ values calculated with the fit results and the given `x` values.
         `y` has shape `(k, n)` and includes the given `x` values, where `k` is the number of data points/isotopes
         and `n` is the dimension of the King plot.
         Also returns the total and the statistical covariance matrices of the results.
        :raises ValueError: If no fit was performed yet.
        """
        if self.popt is None or self.pcov is None or self.alpha is None:
            raise ValueError(self.FIT_NOT_DONE_ERROR)

        mask = None
        if not self.nd:
            indexes = np.arange(4, dtype=int)
            mask = indexes == self.axis
            mask += indexes == 2 + self.axis
            mask = ~mask

            pcov = self.pcov.copy()
            self.popt = np.insert(self.popt, [0, 1], [self.alpha, 1.0])
            self.pcov = np.zeros((4, 4), dtype=float)
            self.pcov[np.ix_(mask, mask)] = pcov

        a, a_ref, x = np.asarray(a), np.asarray(a_ref), np.asarray(x)
        i, i_ref = self._get_i(a, a_ref)

        dim = self.popt.size // 2

        _a, _b = self.popt[:dim], self.popt[dim:]

        m_mod = self.m_mod[i_ref, i, :]
        x_mod = x[:, 0] * m_mod[:, 0]
        x_mod_d = np.sqrt((x[:, 1] * m_mod[:, 0]) ** 2 + (x[:, 0] * m_mod[:, 1]) ** 2)

        jac = [np.array([self._get_gradient_mod(_x_mod, d, axis) for d in range(dim)]) for _x_mod in x_mod]
        cov_x = [
            np.block([[self.pcov, np.zeros((2 * dim, 1))], [np.zeros((1, 2 * dim)), _x_mod_d**2]])
            for _x_mod_d in x_mod_d
        ]
        cov_mod = [_jac @ (_cov_x @ _jac.T) for _jac, _cov_x in zip(jac, cov_x)]
        corr_mod = [_cov / np.sqrt(np.diag(_cov)[:, None] * np.diag(_cov)[None, :]) for _cov in cov_mod]

        y_mod = _a[None, :] + _b[None, :] / _b[axis] * (x_mod[:, None] - _a[axis])

        jac = [
            np.array([self._get_gradient(_x, _m_mod, d, axis) for d in range(dim)])
            for _m_mod, _x in zip(m_mod[:, 0], x[:, 0])
        ]
        cov_x = [
            np.block([[self.pcov, np.zeros((2 * dim, 2))], [np.zeros((2, 2 * dim)), np.diag([_m_mod_d**2, _x_d**2])]])
            for _m_mod_d, _x_d in zip(m_mod[:, 1], x[:, 1])
        ]
        cov_x_stat = [
            np.block(
                [
                    [np.zeros((2 * dim, 2 * dim)), np.zeros((2 * dim, 2))],
                    [np.zeros((2, 2 * dim)), np.diag([_m_mod_d**2, _x_d**2])],
                ]
            )
            for _m_mod_d, _x_d in zip(m_mod[:, 1], x[:, 1])
        ]

        cov = np.array([_jac @ (_cov_x @ _jac.T) for _jac, _cov_x in zip(jac, cov_x)])
        cov_stat = np.array([_jac @ (_cov_stat @ _jac.T) for _jac, _cov_stat in zip(jac, cov_x_stat)])

        y = y_mod / m_mod[:, 0, None]

        if show:
            add_xy = np.array(
                [
                    [
                        [
                            _y_mod[self.axis],
                            np.sqrt(_cov[self.axis, self.axis]),
                            _y_mod[d],
                            np.sqrt(_cov[d, d]),
                            _corr[self.axis, d],
                        ]
                        for _y_mod, _cov, _corr in zip(y_mod, cov_mod, corr_mod)
                    ]
                    for d in range(dim)
                    if d != self.axis
                ]
            )
            add_a = np.array([a, a_ref]).T
            if not self.nd:
                assert mask is not None
                self.popt = self.popt[mask]
                self.pcov = self.pcov[np.ix_(mask, mask)]
                add_xy = add_xy[0]

            self.plot(add_xy=add_xy, add_a=add_a, **kwargs)

        return y, cov, cov_stat

    def _plot_2d(
        self,
        mode: str = "",
        sigma2d: int = 1,
        scale: tuple | None = None,
        add_xy: array_like | None = None,
        add_a: array_like | None = None,
        show: bool = True,
        **kwargs,
    ) -> None:
        r"""
        Plot the 2-dimensional King plot based on `King.x_mod` and the fit results `King.popt` and `King.pcov`.
        See [`King.plot`] for the parameters.
        """
        if (
            self.popt is None
            or self.pcov is None
            or self.alpha is None
            or self.x_mod is None
            or self.corr is None
            or self.a_fit is None
            or self.a_ref is None
        ):
            raise ValueError(self.FIT_NOT_DONE_ERROR)

        add_xy = tools.asarray_optional(add_xy)
        add_a = tools.asarray_optional(add_a)

        a, b = self.popt
        sigma_a, sigma_b = np.sqrt(np.diag(self.pcov))
        corr = self.pcov[0, 1] / (sigma_a * sigma_b)

        scale_x = 1e-3 if scale is None else scale[0]
        scale_y = 1e-3 if scale is None else scale[1]

        offset_str = f" - {round(self.alpha * scale_x, 1)}" if self.alpha != 0 else ""
        xlabel = f"x{offset_str} (mode='shifts' or ='radii' for right x-label)"
        if mode == "radii":
            if scale is None:
                scale_x = 1.0
            offset_str = f" - {round(self.alpha * scale_x, 1)}" if self.alpha != 0 else ""
            xlabel = r"$\mu\,\delta\langle r^2\rangle^{A, A^\prime}" + offset_str + r"\quad(\mathrm{u}\,\mathrm{fm}^2)$"

        elif mode == "shifts":
            if scale is None:
                scale_x = 1e-3
            offset_str = f" - {round(self.alpha * scale_x, 1)}" if self.alpha != 0 else ""
            xlabel = r"$\mu\,\delta\nu_x^{A, A^\prime}" + offset_str + r"\quad(\mathrm{u\,GHz})$"

        if sigma2d:
            plt.plot((self.x_mod[:, 0, 0] - self.alpha) * scale_x, self.x_mod[:, 1, 0] * scale_y, "k.", label="Data")
            draw_sigma2d(
                (self.x_mod[:, 0, 0] - self.alpha) * scale_x,
                self.x_mod[:, 1, 0] * scale_y,
                self.x_mod[:, 0, 1] * scale_x,
                self.x_mod[:, 1, 1] * scale_y,
                self.corr,
                n=sigma2d,
            )
            if add_xy is not None:
                plt.plot((add_xy[:, 0] - self.alpha) * scale_x, add_xy[:, 2] * scale_y, "r.", label="Data")
                draw_sigma2d(
                    (add_xy[:, 0] - self.alpha) * scale_x,
                    add_xy[:, 2] * scale_y,
                    add_xy[:, 1] * scale_x,
                    add_xy[:, 3] * scale_y,
                    add_xy[:, 4],
                    n=sigma2d,
                    **{"fmt": "-r"},
                )
        else:
            plt.errorbar(
                (self.x_mod[:, 0, 0] - self.alpha) * scale_x,
                self.x_mod[:, 1, 0] * scale_y,
                xerr=self.x_mod[:, 0, 1] * scale_x,
                yerr=self.x_mod[:, 1, 1] * scale_y,
                fmt="k.",
                label="Data",
            )
            if add_xy is not None:
                plt.errorbar(
                    (add_xy[:, 0] - self.alpha) * scale_x,
                    add_xy[:, 2] * scale_y,
                    xerr=add_xy[:, 1] * scale_x,
                    yerr=add_xy[:, 3] * scale_y,
                    fmt="r.",
                    label="Add. Data",
                )

        for j in range(len(self.a_fit)):
            pm = -1.0 if b > 0 else 1.0
            plt.text(
                (self.x_mod[j, 0, 0] + 1.5 * self.x_mod[j, 0, 1] - self.alpha) * scale_x,
                (self.x_mod[j, 1, 0] + pm * 1.5 * self.x_mod[j, 1, 1]) * scale_y,
                f"{self.a_fit[j]} - {self.a_ref[j]}",
                ha="left",
                va="top" if b > 0 else "bottom",
            )

        if add_a is not None and add_xy is not None:
            for j in range(add_xy.shape[0]):
                pm = 1.0 if b > 0 else -1.0
                plt.text(
                    (add_xy[j, 0] - 1.5 * add_xy[j, 1] - self.alpha) * scale_x,
                    (add_xy[j, 2] + pm * 1.5 * add_xy[j, 3]) * scale_y,
                    "{} - {}".format(*add_a[j]),
                    ha="right",
                    va="bottom" if b > 0 else "top",
                )

        min_x, max_x = np.min(self.x_mod[:, 0, 0]), np.max(self.x_mod[:, 0, 0])
        if add_xy is not None:
            min_x, max_x = np.min([min_x, np.min(add_xy[:, 0])]), np.max([max_x, np.max(add_xy[:, 0])])
        x_cont = np.linspace(min_x - 0.1 * (max_x - min_x), max_x + 0.1 * (max_x - min_x), 1001) - self.alpha
        plt.plot(x_cont * scale_x, straight(x_cont, a, b) * scale_y, "b-", label="Fit")
        y_min = straight(x_cont, a, b) - straight_std(x_cont, sigma_a, sigma_b, corr)
        y_max = straight(x_cont, a, b) + straight_std(x_cont, sigma_a, sigma_b, corr)
        plt.fill_between(x_cont * scale_x, y_min * scale_y, y_max * scale_y, color="b", alpha=0.3, antialiased=True)

        plt.xlabel(xlabel)
        plt.ylabel(r"$\mu\,\delta\nu_y^{A, A^\prime}\quad(\mathrm{u\,GHz})$")

        plt.legend(numpoints=1, loc="best")
        plt.margins(0.1)
        plt.tight_layout()
        if show:
            plt.show()
            plt.close()

    def plot(
        self,
        mode: str = "",
        sigma2d: int = 1,
        scale: tuple | None = None,
        add_xy: array_like | None = None,
        add_a: array_like | None = None,
        font_dict: dict | None = None,
        show: bool = True,
        **kwargs: Any,
    ) -> None:
        r"""
        Show the King plot using <a href="https://matplotlib.org/">`matplotlib`</a>.
        The King plot is based on the modified data `King.x_mod` and the fit results `King.popt` and `King.pcov`.

        :param mode: The mode of the King plot, which determines the label of the $x$-axis.
         If `mode == "radii"`, the differences of mean-square nuclear charge radii
         $\delta\langle r^2\rangle^{A,A_\mathrm{ref}}$ or $\Lambda^{A,A_\mathrm{ref}}$ are expected for the $x$-axis,
         if `mode == "shifts"`, isotope shifts $\delta\nu_x^{A,A_\mathrm{ref}}$ are expected.
        :param sigma2d: Whether to draw actual 2-dimensional uncertainty bounds or classical errorbars.
         An integer number corresponds to the number of drawn sigma regions.
        :param scale: Factors to scale the $x$ and the $y$-axis.
        :param add_xy: Additional $x$ and $y$ data to plot. Must have shape `(n - 1, k, 5)`
         (`(k, 5)` for 2-dimensional fits), where `n` is the dimension of the data vectors,
         `k` is the number of additional data points and the five entries
         in the last axis correspond to arrays of the form `[x, x_d, y, y_d, corr_xy]`.
        :param add_a: Additional mass numbers for the additional data. Must have shape `(k, 2)`,
         where each row is a tuple `(A, A_ref)`.
        :param font_dict: The font_dict passed to `matplotlib.rc("font", font_dict)`.
        :param show: Whether to call `matplotlib.pyplot.show`.
        :param kwargs: Additional keyword arguments are accepted for compatibility with the fit methods and ignored.
        :raises ValueError: If no fit was performed yet.
        """
        if font_dict is None:
            font_dict = {}
        font_dict = {**{"size": 12, "family": "arial"}, **font_dict}
        mpl.rc("font", **font_dict)

        if not self.nd:
            return self._plot_2d(
                mode=mode, sigma2d=sigma2d, scale=scale, add_xy=add_xy, add_a=add_a, show=show, **kwargs
            )

        if (
            self.popt is None
            or self.pcov is None
            or self.alpha is None
            or self.x_mod is None
            or self.corr is None
            or self.cov is None
            or self.a_fit is None
            or self.a_ref is None
        ):
            raise ValueError(self.FIT_NOT_DONE_ERROR)

        add_xy = tools.asarray_optional(add_xy)
        add_a = tools.asarray_optional(add_a)

        dim = int(self.popt.size / 2)
        a, b = self.popt[:dim], self.popt[dim:]
        sigma_a, sigma_b = np.sqrt(np.diag(self.pcov)[:dim]), np.sqrt(np.diag(self.pcov)[dim:])

        scale_x = 1e-3 if scale is None else scale[0]
        scale_y = 1e-3 if scale is None else scale[1]
        offset_str = f" - {round(self.alpha * scale_x, 1)}" if self.alpha != 0 else ""
        xlabel = f"x{offset_str} (mode='shifts' or ='radii' for right x-label)"
        if mode == "radii":
            if scale is None:
                scale_x = 1.0
            offset_str = f" - {round(self.alpha * scale_x, 1)}" if self.alpha != 0 else ""
            xlabel = r"$\mu\,\delta\langle r^2\rangle^{A, A^\prime}" + offset_str + r"\quad(\mathrm{u}\,\mathrm{fm}^2)$"
        elif mode == "shifts":
            if scale is None:
                scale_x = 1e-3
            offset_str = f" - {round(self.alpha * scale_x, 1)}" if self.alpha != 0 else ""
            xlabel = r"$\mu\,\delta\nu_x^{A, A^\prime}" + offset_str + r"\quad(\mathrm{u\,GHz})$"

        n_plots = self.x_mod.shape[1] - 1
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))
        plot_indexes = [i for i in range(n_plots + 1) if i != self.axis]
        for i, iy in enumerate(plot_indexes):
            plt.subplot(n_rows, n_cols, i + 1)

            if sigma2d:
                plt.plot(
                    (self.x_mod[:, self.axis, 0] - self.alpha) * scale_x,
                    self.x_mod[:, iy, 0] * scale_y,
                    "k.",
                    label="Data",
                )
                draw_sigma2d(
                    (self.x_mod[:, self.axis, 0] - self.alpha) * scale_x,
                    self.x_mod[:, iy, 0] * scale_y,
                    np.sqrt(self.cov[:, self.axis, self.axis]) * scale_x,
                    np.sqrt(self.cov[:, iy, iy]) * scale_y,
                    self.cov[:, self.axis, iy] / np.sqrt(self.cov[:, self.axis, self.axis] * self.cov[:, iy, iy]),
                    n=sigma2d,
                )
                if add_xy is not None:
                    plt.plot((add_xy[i, :, 0] - self.alpha) * scale_x, add_xy[i, :, 2] * scale_y, "r.", label="Data")
                    draw_sigma2d(
                        (add_xy[i, :, 0] - self.alpha) * scale_x,
                        add_xy[i, :, 2] * scale_y,
                        add_xy[i, :, 1] * scale_x,
                        add_xy[i, :, 3] * scale_y,
                        add_xy[i, :, 4],
                        n=sigma2d,
                        **{"fmt": "-r"},
                    )
            else:
                plt.errorbar(
                    (self.x_mod[:, self.axis, 0] - self.alpha) * scale_x,
                    self.x_mod[:, iy, 0] * scale_y,
                    xerr=np.sqrt(self.cov[:, self.axis, self.axis]) * scale_x,
                    yerr=np.sqrt(self.cov[:, iy, iy]) * scale_y,
                    fmt="k.",
                    label="Data",
                )

            for j in range(len(self.a_fit)):
                pm = -1.0 if b[iy] > 0 else 1.0
                plt.text(
                    (self.x_mod[j, self.axis, 0] + 1.5 * np.sqrt(self.cov[j, self.axis, self.axis]) - self.alpha)
                    * scale_x,
                    (self.x_mod[j, iy, 0] + pm * 1.5 * np.sqrt(self.cov[j, iy, iy])) * scale_y,
                    f"{self.a_fit[j]} - {self.a_ref[j]}",
                    ha="left",
                    va="top" if b[iy] > 0 else "bottom",
                )

            if add_a is not None and add_xy is not None:
                for j in range(add_xy.shape[1]):
                    pm = 1.0 if b[iy] > 0 else -1.0
                    plt.text(
                        (add_xy[i, j, 0] - 1.5 * add_xy[i, j, 1] - self.alpha) * scale_x,
                        (add_xy[i, j, 2] + pm * 1.5 * add_xy[i, j, 3]) * scale_y,
                        "{} - {}".format(*add_a[j]),
                        ha="right",
                        va="bottom" if b[iy] > 0 else "top",
                    )

            min_x, max_x = np.min(self.x_mod[:, self.axis, 0]), np.max(self.x_mod[:, self.axis, 0])
            if add_xy is not None:
                min_x, max_x = np.min([min_x, np.min(add_xy[i, :, 0])]), np.max([max_x, np.max(add_xy[i, :, 0])])

            x_cont = np.linspace(min_x - 0.1 * (max_x - min_x), max_x + 0.1 * (max_x - min_x), 1001) - self.alpha

            plt.plot(x_cont * scale_x, straight(x_cont, a[iy], b[iy]) * scale_y, "b-", label="Fit")

            corr = self.pcov[iy, dim + iy] / (sigma_a[iy] * sigma_b[iy])
            y_min = straight(x_cont, a[iy], b[iy]) - straight_std(x_cont, sigma_a[iy], sigma_b[iy], corr)
            y_max = straight(x_cont, a[iy], b[iy]) + straight_std(x_cont, sigma_a[iy], sigma_b[iy], corr)
            plt.fill_between(x_cont * scale_x, y_min * scale_y, y_max * scale_y, color="b", alpha=0.3, antialiased=True)

            plt.xlabel(xlabel)
            plt.ylabel(r"$\mu\,\delta\nu_{y" + str(i) + r"}^{A, A^\prime}\quad(\mathrm{u\,GHz})$")
            plt.legend(numpoints=1, loc="best")
            plt.margins(0.1)

        if show:
            plt.show()
            plt.close()

        return None
