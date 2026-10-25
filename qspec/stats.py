"""
qspec.stats
===========

Module for statistical analysis and uncertainty propagation.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as si
import scipy.stats as st
from scipy.optimize import curve_fit, minimize, root

from qspec import tools
from qspec.qtypes import (
    Any,
    Callable,
    Iterable,
    SupportsFloat,
    SupportsIndex,
    array_like,
    asarray,
    cast,
    int_like,
    ndarray,
    scalar,
    scalar_nd,
)

__all__ = [
    "Observable",
    "add",
    "average",
    "combined_pdf",
    "estimate_skewnorm",
    "info",
    "median",
    "mode_lognormal",
    "mul",
    "propagate",
    "propagate_fit",
    "relevant_interval",
    "uniform",
    "uniform_pumped",
]


class Observable(float):
    """
    A float with a `label` and either a symmetric or a left- and right-sided uncertainty.
    The operators `+`, `*` and `**` propagate the uncertainties with [`propagate`].
    """

    label: str | None
    """The label of the observable."""
    std: float
    """The standard deviation, or the left-sided 1-sigma percentile relative to the value if `std_2` is not 0."""
    std_2: float
    """The right-sided 1-sigma percentile relative to the value. If 0, the uncertainty is symmetric."""
    popt: ndarray | None
    """The parameters of a skew normal distribution fitted to the observable, see [`Observable.set_popt`]."""
    est: ndarray | None
    """The parameters of a skew normal distribution estimated from the asymmetric uncertainties."""

    def __new__(
        cls,
        x: SupportsFloat | SupportsIndex | str | bytes | bytearray,
        std: scalar = 0.0,
        std_2: scalar = 0.0,
        label: str | None = None,
    ) -> "Observable":
        return super().__new__(cls, x)

    def __init__(
        self,
        x: SupportsFloat | SupportsIndex | str | bytes | bytearray,
        std: scalar = 0.0,
        std_2: scalar = 0.0,
        label: str | None = None,
    ) -> None:
        """
        :param x: The value of the observable.
        :param std: The standard deviation of the observable if `std_2` is 0,
         else the left-sided 1-sigma percentile relative to `x`.
        :param std_2: The right-sided 1-sigma percentile relative to `x`. If 0, the uncertainty is symmetric.
        :param label: The label of the observable.
        """
        float.__init__(float(x))
        self.label = label
        self.std = float(std)
        self.std_2 = float(std_2)
        self.popt = None
        self.est = None
        if std_2 is not None:
            try:
                self.est = estimate_skewnorm(float(self), self.std, self.std_2)
            except ValueError as e:
                tools.printw(repr(e))
                tools.printw("The real PDF might not be sufficiently described by a skew normal distribution.")

    def __repr__(self) -> str:
        return f"Observable({super().__repr__()}, {self.std}, {self.std_2}, '{self.label}')"

    def __str__(self) -> str:
        if self.std == 0.0:
            return f"{super().__repr__()}"

        if self.std_2 == 0.0:
            return f"{super().__repr__()}(+-{self.std})"

        return f"{super().__repr__()}(-{self.std}+{self.std_2})"

    def __add__(self, other: Any) -> "Observable":
        return _propagate(add, [self, other])[0]

    def __radd__(self, other: Any) -> "Observable":
        return _propagate(add, [other, self])[0]

    def __mul__(self, other: Any) -> "Observable":
        return _propagate(mul, [self, other])[0]

    def __rmul__(self, other: Any) -> "Observable":
        return _propagate(mul, [other, self])[0]

    def __pow__(self, power: Any, mod: None = None) -> "Observable":
        return _propagate(pow, [self, power])[0]

    def __rpow__(self, power: Any, mod: None = None) -> "Observable":
        return _propagate(pow, [power, self])[0]

    def set_popt(self, popt: array_like | None = None) -> None:
        """
        Set the parameters `Observable.popt` of a skew normal distribution, e.g., fitted to a sample of the observable.

        :param popt: The parameters `(a, loc, scale)` of `scipy.stats.skewnorm` or `None`.
        """
        if popt is None:
            self.popt = popt
        else:
            self.popt = np.asarray(popt)

    def rvs(self, size: int = 1) -> ndarray:
        """
        Random variates of the observable. If asymmetric uncertainties are specified,
        a skew normal distribution is assumed. However, the ratio between the left- and right-sided uncertainty
        must not exceed 1.5.

        :param size: The number of random variates. The default is 1.
        :returns: (x) Random variates of the observable with the given `size`.
        """
        if self.std_2 == 0.0:
            return np.asarray(st.norm.rvs(loc=self, scale=self.std, size=size), dtype=float)

        if self.popt is not None:
            return np.asarray(st.skewnorm.rvs(*self.popt, size=size), dtype=float)

        if self.est is not None:
            return np.asarray(st.skewnorm.rvs(*self.est, size=size), dtype=float)

        return np.full(size, np.nan)

    def pdf(self) -> Callable:
        """
        The probability density function (pdf) of the observable. If asymmetric uncertainties are specified,
        a skew normal distribution is assumed.

        :returns: (pdf) The pdf as a function of the quantiles `x`.
        """
        if self.std_2 == 0.0:
            return lambda x: st.norm.pdf(x, loc=self, scale=self.std)

        if self.popt is not None:
            return lambda x: st.skewnorm.pdf(x, *np.asarray(self.popt))

        if self.est is not None:
            return lambda x: st.skewnorm.pdf(x, *np.asarray(self.est))

        return lambda x: np.full_like(x, np.nan)

    def hist(self, size: int_like = 1000000, n_bins: int_like = 200) -> None:
        """
        Plot a histogram of random variates of the observable together with its pdf.

        :param size: The number of random variates. The default is 1,000,000.
        :param n_bins: The number of bins.
        """
        y = self.rvs(int(size))
        _, bins, _ = plt.hist(y, bins=int(n_bins), density=True, label="Sample data", facecolor="lightgray")
        bins = bins[:-1] + 0.5 * (bins[-1] - bins[0]) / n_bins
        plt.plot(bins, self.pdf()(bins), "C0", label="Est. PDF")
        plt.legend(loc=1)
        plt.show()


def add(a: Any, b: Any) -> Any:
    """
    The sum `a + b`, used by [`propagate`] to add observables.

    :param a: The first summand.
    :param b: The second summand.
    :returns: (sum) The sum of the two summands.
    """
    return a + b


def mul(a: Any, b: Any) -> Any:
    """
    The product `a * b`, used by [`propagate`] to multiply observables.

    :param a: The first factor.
    :param b: The second factor.
    :returns: (product) The product of the two factors.
    """
    return a * b


def average(
    a: array_like, std: array_like | None = None, cov: scalar_nd | None = None, axis: int_like | None = None
) -> tuple[ndarray, ndarray]:
    r"""
    The (weighted) average of a sample and its standard error

    $$\begin{aligned}
    \bar{a} &= \frac{\sum_i w_ia_i}{\sum_i w_i},\quad w_i = \frac{1}{\Sigma_{ii}}\\[1ex]
    \sigma_{\bar{a}}^2 &= \frac{\sum_{ij} w_iw_j\Sigma_{ij}}{\left(\sum_i w_i\right)^2},
    \end{aligned}$$

    with the covariance matrix $\Sigma$ given by `cov` or by $\Sigma_{ij} = \delta_{ij}\sigma_i^2$ if `std` is given.
    If neither is given, the arithmetic mean and the standard error of the mean are returned.

    :param a: The sample data.
    :param std: An array of standard deviations associated with the values in `a`.
     If specified, the weighted average of the uncorrelated sample data and its standard error is returned.
    :param cov: The covariance matrix associated with the values in `a`.
     If specified, it overrides `std` and the weighted average of the correlated sample data
     and its standard error is returned. If no axis is specified, `cov` must have shape `(a.size, a.size)`
     with elements associated with the values in the flattened `a`,
     otherwise `cov` must have either shape `(a.shape[axis], a.shape[axis])`
     or `(..., a.shape[axis - 1], a.shape[axis], a.shape[axis], a.shape[axis + 1], ...)`.
    :param axis: The axis along which the average is computed. If `None`, `a` is flattened.
    :returns: (av, av_d) The average and its standard error for a given sample `a` along the specified `axis`.
    :raises ValueError: If the shapes of `a` and `cov` do not match.
    """
    a = np.asarray(a, dtype=float)
    axis = None if axis is None else int(axis)

    if std is None and cov is None:
        av = np.average(a, axis=axis)
        av_d = np.std(a, axis=axis, ddof=1)

        if axis is None:
            av_d /= np.sqrt(a.size)
        else:
            av_d /= np.sqrt(a.shape[axis])

    elif std is not None and cov is None:
        std = np.asarray(std, dtype=float)
        av, sum_of_weights = np.average(a, axis=axis, weights=1.0 / (std**2), returned=True)
        av_d = np.sqrt(1.0 / sum_of_weights)

    else:
        cov = np.asarray(cov, dtype=float)
        if axis is None:
            if cov.shape != (a.size, a.size):
                raise ValueError(f"Shape mismatch between `a` {a.shape} and `cov` {cov.shape}.")
            av, sum_of_weights = np.average(a, weights=1.0 / np.diag(cov), returned=True)
            av_d = np.sum(
                [
                    1.0 / cov[i, i]
                    + 2.0 * np.sum([cov[i, j] / (cov[i, i] * cov[j, j]) for j in range(a.size) if i < j])
                    for i in range(a.size)
                ]
            )

        else:
            if a.size != a.shape[0] and cov.shape == (a.size, a.size):
                raise ValueError(
                    f"Shape mismatch between `a` {a.shape} and `cov` {cov.shape} for specified axis {axis}."
                )

            if cov.shape == (a.shape[axis], a.shape[axis]):
                av, sum_of_weights = np.average(a, axis=axis, weights=1.0 / np.diag(cov), returned=True)
                av_d = np.sum(
                    [
                        1.0 / cov[i, i]
                        + 2.0 * np.sum([cov[i, j] / (cov[i, i] * cov[j, j]) for j in range(a.shape[axis]) if i < j])
                        for i in range(a.shape[axis])
                    ]
                )

            else:
                var = np.diagonal(cov, axis1=axis, axis2=axis + 1)
                axes = [ax for ax in range(len(var.shape))]
                i = axes.pop(-1)
                axes.insert(axis, i)
                var = np.transpose(var, axes=axes)
                sum_of_weights = np.sum(1.0 / var, axis=axis)
                av = np.sum(a / var, axis=axis) / sum_of_weights
                av_d = np.sum(
                    cov / (np.expand_dims(var, axis=axis + 1) * np.expand_dims(var, axis=axis)), axis=(axis + 1, axis)
                )

        av_d = np.sqrt(av_d) / sum_of_weights

    return np.asarray(av, dtype=float), np.asarray(av_d, dtype=float)


def median(a: array_like, axis: int_like | None = None) -> tuple[ndarray, ndarray, ndarray]:
    """
    The median and the 1-sigma percentiles of a sample, ignoring `nan` values.

    :param a: The sample data.
    :param axis: The axis along which the three percentiles are computed.
    :returns: (med, per_0, per_1) The median (0.5-percentile) as well as the left- (~0.1587)
     and right-sided (~0.8413) 1-sigma percentile of a given sample `a` along the specified `axis`.
    """
    a = np.asarray(a, dtype=float)
    med = np.nanmedian(a, axis=axis)
    neg = np.nanpercentile(a, 15.8655254, axis=axis)
    pos = np.nanpercentile(a, 84.1344746, axis=axis)
    return med, neg, pos


def estimate_skewnorm(med: scalar, per_0: scalar, per_1: scalar) -> ndarray | None:
    """
    Estimate the parameters of a skew normal distribution from its median and 1-sigma percentiles.

    :param med: The median (0.5-percentile) of a random variable.
    :param per_0: The left-sided 1-sigma percentile (~0.1587-percentile) relative to `med`.
    :param per_1: The right-sided 1-sigma percentile (~0.8413-percentile) relative to `med`.
    :returns: (popt) The estimated parameters `(a, loc, scale)` of `scipy.stats.skewnorm`
     that match the given percentiles, or `None` if `per_0` or `per_1` is 0.
    :raises ValueError: If the ratio between the left-(right-) and right-(left-)sided uncertainty exceeds 1.5.
    """
    if per_0 == 0 or per_1 == 0:
        return None
    if per_0 / per_1 > 1.5 or per_1 / per_0 > 1.5:
        raise ValueError("The ratio between the left-(right-) and right-(left-)sided uncertainty must not exceed 1.5.")
    per = np.array([med, med - per_0, med + per_1])
    y = np.array([0.5, 0.158655254, 0.841344746])

    def f(x: ndarray) -> ndarray:
        return st.skewnorm.cdf(per, *x) - y

    def df(x: ndarray) -> ndarray:
        xi0 = -np.exp(-0.5 * ((per - x[1]) / x[2]) ** 2 * (1.0 + x[0] ** 2)) / (np.pi * (1.0 + x[0] ** 2))
        xi1 = -st.skewnorm.pdf(per, *x)
        xi2 = -st.skewnorm.pdf(per, *x) * (per - x[1]) / x[2]
        return np.array([xi0, xi1, xi2]).T

    return root(f, x0=np.array([0.0, med, per_0]), jac=df).x


def lognormal(
    x: array_like, s: array_like = 0.0, loc: array_like = 0.0, scale: array_like = 1.0, a: array_like = 1.0
) -> ndarray:
    x, s, loc, scale, a = asarray(x, s, loc, scale, a, dtype=float)
    return a * st.lognorm.pdf(x, s=s, loc=loc, scale=scale)


def mode_lognormal(x: array_like, bins: array_like = 100) -> float:
    """
    Estimate the mode of a sample by fitting a lognormal distribution to its histogram.
    The histogram and the fitted distribution are plotted.

    :param x: The sample data.
    :param bins: The number of bins or the bin edges of the histogram, see `numpy.histogram`.
    :returns: (mode) The mode of the lognormal distribution.
    """
    y, edges = np.histogram(np.asarray(x), bins=bins)
    delta = np.mean(edges[1:] - edges[:-1])
    x = edges[:-1] + 0.5 * delta

    loc = np.min(x) - 0.1 * (np.max(x) - np.min(x))
    scale = np.mean(x - loc)
    s = np.std(np.log(x - loc))
    a = scale * np.max(y)
    p0 = [s, loc, scale, a]

    popt, _ = curve_fit(lognormal, x, y, p0=p0)
    mode = popt[1] + popt[2] * np.exp(-(popt[0] ** 2))

    plt.bar(x, y, width=delta)
    plt.plot(x, lognormal(x, *p0), "-C1")
    plt.plot(x, lognormal(x, *popt), "-C2")
    plt.vlines(mode, 0, np.max(y), colors="k")
    plt.show()
    return mode


def propagate(
    f: Callable,
    x: Iterable[scalar],
    x_d: array_like | None = None,
    cov: scalar_nd | None = None,
    unc_places: int_like | None = None,
    sample_size: int_like = 1000000,
    rtol: float = 1e-3,
    atol: float | None = None,
    force_sym: bool = False,
    full_output: bool = False,
    show: bool = False,
) -> Observable | tuple[Observable, ndarray, ndarray]:
    """
    Propagate the uncertainties of the input values `x` through the function `f` using Monte-Carlo sampling.

    If the resulting uncertainties are asymmetric, the value and the two uncertainties of the returned `Observable`
    are the median and the left- (~0.1587) and right-sided (~0.8413) 1-sigma percentiles
    relative to the median, respectively, and the parameters of a skew normal distribution
    are fitted to the sample and stored in the observable. If the uncertainties are symmetric,
    the observable is assumed to be normally distributed and its value and uncertainty are
    the mean and the standard deviation of the sample.

    :param f: The function to compute. `f` needs to be vectorized.
    :param x: The input values. If `x_d` is `None`, the sample data is generated with [`Observable.rvs`],
     which considers asymmetric uncertainties. If an element is not an `Observable`, its uncertainty is assumed to be 0.
    :param x_d: The uncertainties of the input values.
    :param cov: The covariance matrix of the input values. If not `None`, `x` is assumed to be distributed according
     to a multivariate normal distribution with covariance `cov`.
    :param unc_places: The number of significant decimal places the result will be rounded to.
     If `None`, the result is not rounded.
    :param sample_size: The number of random variates used for the calculation. The default is 1,000,000.
    :param rtol: The relative tolerance, with respect to the median of the resulting sample,
     with which the left- and right-sided uncertainties can deviate before asymmetric uncertainties are used.
    :param atol: The absolute tolerance with which the left- and right-sided uncertainties can deviate,
     before asymmetric uncertainties are used. Overrides `rtol`.
    :param force_sym: Whether to force symmetric uncertainties. If so, a normal distribution is assumed.
    :param full_output: Whether to return the randomly generated data samples.
    :param show: Whether to show a histogram and estimated pdfs of the computed sample data.
    :returns: (ob, x_samples, y_samples) An `Observable` whose uncertainties result from the propagation
     of the uncertainties of the input values `x` by the function `f`.
     If `full_output` is `True`, the input samples and the output sample are returned as well.
    :raises ValueError: If `x` and `x_d` have different sizes.
    """
    ob, rand_x, rand_y = _propagate(
        f,
        x,
        x_d=x_d,
        cov=cov,
        unc_places=unc_places,
        sample_size=sample_size,
        rtol=rtol,
        atol=atol,
        force_sym=force_sym,
        show=show,
    )
    if full_output:
        return ob, np.array([np.broadcast_to(np.asarray(r, dtype=float), rand_y.shape) for r in rand_x]), rand_y
    return ob


def _propagate(
    f: Callable,
    x: Iterable[scalar],
    x_d: array_like | None = None,
    cov: scalar_nd | None = None,
    unc_places: int_like | None = None,
    sample_size: int_like = 1000000,
    rtol: float = 1e-3,
    atol: float | None = None,
    force_sym: bool = False,
    show: bool = False,
) -> tuple[Observable, list, ndarray]:
    sample_size = int(sample_size)

    label = f.__name__
    n_bins = 200
    if cov is None:
        if x_d is None:
            rand_x = [x_i.rvs(sample_size) if isinstance(x_i, Observable) else x_i for x_i in x]
        else:
            x = np.asarray(x, dtype=float)
            x_d = np.asarray(x_d, dtype=float)
            if x.size != x_d.size:
                raise ValueError(f"x and x_d must have the same size but have sizes {x.size} and {x_d.size}.")
            rand_x = [
                x_i if x_d_i == 0.0 else st.norm.rvs(loc=x_i, scale=x_d_i, size=sample_size)
                for x_i, x_d_i in zip(x, x_d)
            ]
    else:
        cov = np.asarray(cov)
        rand_x = list(
            np.asarray(st.multivariate_normal.rvs(mean=[float(x_i) for x_i in x], cov=cov, size=sample_size)).T
        )

    rand_y = np.asarray(f(*rand_x))

    med, per_0, per_1 = median(rand_y)
    mean = np.average(rand_y)
    std = np.std(rand_y, ddof=1)

    plt.figure(num=1, dpi=96)
    n, bins, _ = plt.hist(rand_y, bins=n_bins, density=True, label="Sample data", facecolor="lightgray")
    bins = bins[:-1] + 0.5 * (bins[-1] - bins[0]) / n_bins

    if (
        force_sym
        or (atol is None and abs(per_1 - per_0) / med <= rtol)
        or (atol is not None and abs(per_1 - per_0) <= atol)
    ):
        if unc_places is not None:
            std, dec = tools.round_to_n(std, unc_places)
            mean = np.around(mean, decimals=dec)
        ob = Observable(mean, label=label, std=float(std))
    else:
        if unc_places is not None:
            per_0, dec_0 = tools.round_to_n(per_0, unc_places)
            per_1, dec_1 = tools.round_to_n(per_1, unc_places)
            dec = np.max([dec_0, dec_1])
            med = np.around(med, decimals=dec)
        ob = Observable(med, label=label, std=float(med - per_0), std_2=float(per_1 - med))
        try:
            popt, _ = curve_fit(st.skewnorm.pdf, bins, np.asarray(n), p0=[0.0, mean, std])
            ob.set_popt(popt)
        except RuntimeError:
            print(
                "Optimal parameters for a skew normal distribution could not be found through fitting."
                " Using an estimation if possible."
            )

    if show:
        print(f"Median(sample data): {med} (-{med - per_0} +{per_1 - med})")
        plt.plot(bins, st.norm.pdf(bins, loc=mean, scale=std), "C0", label="Est. normal")
        if ob.popt is not None:
            med, per_0, per_1 = median(st.skewnorm.rvs(*ob.popt, size=sample_size))
            print(f"Median(Fit. skew normal): {med} (-{med - per_0} +{per_1 - med})")
            plt.plot(bins, st.skewnorm.pdf(bins, *ob.popt), "C1", label="Fit. skew normal")
        if ob.est is not None:
            med, per_0, per_1 = median(st.skewnorm.rvs(*ob.est, size=sample_size))
            print(f"Median(Est. skew normal): {med} (-{med - per_0} +{per_1 - med})")
            plt.plot(bins, st.skewnorm.pdf(bins, *ob.est), "C2", label="Est. skew normal")
        plt.legend(loc=1)
        plt.show()
    plt.close()

    return ob, rand_x, rand_y


def propagate_fit(
    f: Callable, x: array_like, popt: array_like, pcov: array_like, sample_size: int = 1000000
) -> tuple[ndarray, ndarray, ndarray]:
    """
    Propagate the uncertainties of fitted parameters through the function `f` using Monte-Carlo sampling.

    :param f: The function `f(x, *popt)` to compute. `f` needs to be vectorized.
    :param x: The input values.
    :param popt: An array of the fitted parameters, e.g., from `scipy.optimize.curve_fit`.
    :param pcov: A 2d-array of the estimated covariances of `popt`.
    :param sample_size: The number of random variates used for the calculation. The default is 1,000,000.
    :returns: (med, per_0, per_1) The median and the left- and right-sided 1-sigma percentiles
     of the sampled function values at `x`.
    """
    x, popt, pcov = asarray(x, popt, pcov, dtype=float)
    _x = np.expand_dims(x, axis=-1)
    _popt = np.asarray(st.multivariate_normal.rvs(mean=popt, cov=pcov, size=sample_size)).T
    _popt = [np.expand_dims(p, axis=tuple(range(len(x.shape)))) for p in _popt]
    med, per_0, per_1 = median(f(_x, *_popt), axis=-1)
    return med, per_0, per_1


def combined_pdf(
    z: array_like,
    pdf_1: Callable = st.norm.pdf,
    pdf_2: Callable = st.norm.pdf,
    loc_1: float = 0.0,
    scale_1: float = 1.0,
    loc_2: float = 0.0,
    scale_2: float = 1.0,
    operator: str = "+",
    n: int = 10,
) -> ndarray:
    r"""
    The probability density function (pdf) of the sum or the product of two random variates $X$ and $Y$

    $$
    \rho_Z(z) = \begin{cases}\int\rho_X(x)\,\rho_Y(z - x)\,\mathrm{d}x, & Z = X + Y\\
    \int\rho_X(x)\,\rho_Y(z / x)\,\frac{1}{|x|}\,\mathrm{d}x, & Z = XY\end{cases},
    $$

    calculated by numerical integration.

    :param z: The quantiles $z$ of the combined pdf.
    :param pdf_1: The pdf $\rho_X$ of the first random variate.
    :param pdf_2: The pdf $\rho_Y$ of the second random variate.
    :param loc_1: The `loc` parameter of the first pdf.
    :param scale_1: The `scale` parameter of the first pdf.
    :param loc_2: The `loc` parameter of the second pdf.
    :param scale_2: The `scale` parameter of the second pdf.
    :param operator: The operator that defines the new random variate $Z$. Can be either `"+"` or `"*"`.
    :param n: The precision of the numerical integration, which uses `2**n` intervals.
    :returns: (rho_z) The value of the pdf $\rho_Z$ at the given `z` quantiles.
    :raises ValueError: If `operator` is not supported.
    """
    arg = np.asarray(z)
    if arg.shape == ():
        arg = np.expand_dims(arg, axis=0)
    arg = np.expand_dims(arg, axis=-1)

    if operator == "*":
        left = loc_1 * loc_2 - 5.0 * (loc_1 * loc_2 - (loc_1 - scale_1) * (loc_2 - scale_2))
        right = loc_1 * loc_2 + 5.0 * (loc_1 * loc_2 - (loc_1 - scale_1) * (loc_2 - scale_2))

        def kernel(x: ndarray) -> ndarray:
            return pdf_1(x, loc=loc_1, scale=scale_1) * pdf_2(arg / x, loc=loc_2, scale=scale_2) / abs(x)

    elif operator == "+":
        left = loc_1 + loc_2 - 10.0 * np.sqrt(scale_1**2 + scale_2**2)
        right = loc_1 + loc_2 + 10.0 * np.sqrt(scale_1**2 + scale_2**2)

        def kernel(x: ndarray) -> ndarray:
            return pdf_1(x, loc=loc_1, scale=scale_1) * pdf_2(arg - x, loc=loc_2, scale=scale_2)

    else:
        raise ValueError(f"Operator '{operator}' not supported.")

    x_range = np.linspace(left, right, 2**n + 1)
    dx = x_range[1] - x_range[0]
    x_range = np.expand_dims(x_range, axis=0)
    y = kernel(x_range)

    return si.romb(y, dx, axis=1)


def relevant_interval(dist: Callable, *args: scalar, show: bool = False, **kwargs) -> tuple[float, float]:
    """
    Estimate the interval in which most of the probability of a probability density function (pdf) lies.

    :param dist: The pdf.
    :param args: Additional arguments for the pdf.
    :param show: Whether to show the pdf in the determined interval.
    :param kwargs: Additional keyword arguments for the pdf.
    :returns: (x_min, x_max) The estimated interval.
    """
    k = 10
    scale = 1.0

    def cost(_x: ndarray) -> float:
        if _x[0] > _x[1]:
            _y = dist(np.linspace(_x[1], _x[0], 2**k + 1), *args, **kwargs)
        else:
            _y = dist(np.linspace(_x[0], _x[1], 2**k + 1), *args, **kwargs)
        c = float((1.0 - si.romb(_y, abs(_x[1] - _x[0]) / (2**k))) * scale)
        if show:
            print(_x, c / scale)
        return c

    def d_cost(_x: ndarray) -> ndarray:
        _y = np.array([dist(_x[0], *args, **kwargs), -dist(_x[1], *args, **kwargs)])
        if _x[0] > _x[1]:
            _y = -_y
        return _y * scale

    x0 = np.array([-1.0, 1.0])
    while cost(x0) < 0.1:
        x0 *= 0.5

    while cost(x0) > 0.9:
        x0 *= 2.0

    scale = x0[1]
    x_min, x_max = minimize(cost, x0, jac=d_cost).x

    if show:
        x = np.linspace(x_min, x_max, 2**k + 1)
        y = dist(x, *args, **kwargs)
        plt.plot(x, y, label=dist.__name__)
        plt.legend()
        plt.show()

    return float(x_min), float(x_max)


""" Probability density functions """


def uniform(x: array_like, width: array_like) -> ndarray:
    r"""
    The probability density function of a uniform distribution centered at 0

    $$
    \rho(x) = \begin{cases}1/w, & |x| \leq w/2\\ 0, & \text{else}\end{cases}.
    $$

    :param x: The quantiles $x$.
    :param width: The width $w$ of the uniform distribution.
    :returns: (rho) The probability density at `x`.
    """
    x, width = asarray(x, width, dtype=float)
    return st.uniform.pdf(x, loc=-0.5 * width, scale=width)


def _uniform_pumped(x: array_like, width: scalar, gamma_u: scalar, depth: scalar) -> ndarray:
    """
    :param x: The quantiles.
    :param width: The width of the uniform distribution.
    :param gamma_u: The full width at half maximum (FWHM) of the pump transition.
    :param depth: The depth of the pump dip in parts of the height of the underlying uniform distribution.
    :returns: A not normalized uniform distribution with a Lorentz-shaped dip.
    """
    x = np.asarray(x, dtype=float)
    width, gamma_u, depth = cast(width, gamma_u, depth, dtype=float)
    scalar_true = tools.check_shape((), x, return_mode=True)
    if scalar_true:
        x = np.array([x])
    uni = uniform(x, width)
    nonzero = np.nonzero(uni)
    ret = np.zeros_like(x)
    ret[nonzero] = uni[nonzero] - depth / width * (np.pi * gamma_u / 2.0) * st.cauchy.pdf(
        x[nonzero], scale=gamma_u / 2.0
    )
    if scalar_true:
        return ret[0]
    return ret


def uniform_pumped(x: array_like, width: scalar, gamma_u: scalar, depth: scalar) -> ndarray:
    """
    The probability density function of a uniform distribution centered at 0 with a Lorentz-shaped dip at 0,
    e.g., a velocity distribution depleted by optical pumping.

    :param x: The quantiles.
    :param width: The width of the uniform distribution.
    :param gamma_u: The full width at half maximum (FWHM) of the pump transition.
    :param depth: The depth of the pump dip in parts of the height of the underlying uniform distribution.
    :returns: (rho) The normalized probability density at `x`.
    """
    x = np.asarray(x, dtype=float)
    width, gamma_u, depth = cast(width, gamma_u, depth, dtype=float)
    _x = np.linspace(-0.5 * width, 0.5 * width, 2**10 + 1)
    _y = _uniform_pumped(_x, width, gamma_u, depth)
    norm = si.romb(_y, float(width) / 1024.0)
    return _uniform_pumped(x, width, gamma_u, depth) / norm


def info() -> None:
    """
    Plot the 1-sigma percentiles (and their ratio) of a skew normal distribution relative to its median
    for different values of the shape parameter $\\alpha$.
    The plot shows that the ratio between the two 1-sigma percentiles cannot exceed 1.5.
    """
    neg = []
    pos = []
    a = np.linspace(-10, 10, 1001)
    for val in a:
        dist = st.skewnorm.rvs(val, loc=0, scale=1, size=100000)
        med = np.median(dist)
        neg.append(med - np.percentile(dist, 15.8655254))
        pos.append(np.percentile(dist, 84.1344746) - med)
    neg, pos = np.array(neg), np.array(pos)
    plt.title("Properties of the skew normal distribution")
    plt.xlabel(r"Parameter $\alpha$")
    plt.ylabel(r"Percentiles ($\sigma$) / Ratio of Percentiles")
    plt.ylim(0.0, 1.75)
    plt.plot(a, neg, label="left 1-sigma")
    plt.plot(a, pos, label="right 1-sigma")
    plt.plot(a, pos / neg, label="Ratio right/left")
    plt.plot(a, neg / pos, label="Ratio left/right")
    plt.legend(loc="lower center")
    plt.show()
