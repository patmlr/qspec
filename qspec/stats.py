# -*- coding: utf-8 -*-
"""
qspec.stats
===========

Created on 22.04.2020

@author: Patrick Mueller

Module for doing statistical analysis.
"""

import numpy as np
import scipy.stats as st
import scipy.integrate as si
from scipy.optimize import root, minimize
import matplotlib.pyplot as plt

from qspec._types import *
from qspec import tools
from qspec.analyze import curve_fit

__all__ = ['Observable', 'add', 'mul', 'average', 'median', 'estimate_skewnorm', 'propagate', 'propagate_fit',
           'combined_pdf', 'relevant_interval', 'uniform', 'uniform_pumped', 'info']


class Observable(float):
    """
    A float object which has a 'label' and optionally both a left- and right-sided or a symmetric uncertainty.
    """
    def __new__(cls, x: Union[SupportsFloat, SupportsIndex, str, bytes, bytearray],
                std: scalar = None, std_2: scalar = None, label: str = None):
        return super(Observable, cls).__new__(cls, x)

    def __init__(self, x: Union[SupportsFloat, SupportsIndex, str, bytes, bytearray],
                 std: scalar = None, std_2: scalar = None, label: str = None):
        """
        :param x: The value of the observable.
        :param label: The label of the observable.
        :param std: The left-sided 1-sigma percentile relative to 'x' if 'std_2' is specified,
         else the standard deviation of the observable.
        :param std_2: The right-sided 1-sigma percentile relative to 'x' if 'std' is specified,
         else the standard deviation of the observable.
        """
        float.__init__(float(x))
        self.label = label
        self.std = std if std is not None else std_2
        self.std_2 = std_2 if std is not None else std
        self.popt: Union[array_iter, None] = None
        self.est: Union[array_iter, None] = None
        if std_2 is not None:
            try:
                self.est = estimate_skewnorm(float(self), self.std, self.std_2)
            except ValueError as e:
                tools.printw(repr(e))
                tools.printw('The real PDF might not be sufficiently described by a skew normal distribution.')

    def __repr__(self):
        return 'Observable({}, {}, {}, \'{}\')'.format(super().__repr__(), self.std, self.std_2, self.label)

    def __str__(self):
        if self.std is None:
            return '{}'.format(super().__repr__())
        elif self.std_2 is None:
            return '{}(+-{})'.format(super().__repr__(), self.std)
        else:
            return '{}(-{}+{})'.format(super().__repr__(), self.std, self.std_2)

    def __add__(self, other):
        return propagate(add, [self, other])

    def __radd__(self, other):
        return propagate(add, [other, self])

    def __mul__(self, other):
        return propagate(mul, [self, other])

    def __rmul__(self, other):
        return propagate(mul, [other, self])

    def __pow__(self, power, modulo=None):
        return propagate(pow, [self, power])

    def __rpow__(self, power, modulo=None):
        return propagate(pow, [power, self])

    def set_popt(self, popt: array_like = None):
        """
        :param popt: The new value of 'popt'.
        :returns: None. Sets the 'popt' attribute.
        """
        if popt is None:
            self.popt = popt
        else:
            self.popt = np.asarray(popt)

    def rvs(self, size: int = 1) -> ndarray:
        """
        :param size: The defining number of random variates (default is 1).
        :returns: Random variates of the observable of given 'size'. If asymmetric uncertainties are specified,
         a skew normal distribution is assumed. However, the ratio between the left- and right-sided uncertainty
         must not exceed 1.5.
        """
        if self.std_2 is None:
            return st.norm.rvs(loc=self, scale=0. if self.std is None else self.std, size=size)
        elif self.popt is not None:
            return st.skewnorm.rvs(*self.popt, size=size)
        elif self.est is not None:
            return st.skewnorm.rvs(*self.est, size=size)
        else:
            return np.full(size, np.nan)

    def pdf(self) -> Callable:
        """
        :returns: The probability distribution function (pdf) of the observable.
        """
        if self.std_2 is None:
            return np.zeros_like if self.std is None else st.norm(loc=self, scale=self.std).pdf
        elif self.popt is not None:
            return st.skewnorm(*self.popt).pdf
        elif self.est is not None:
            return st.skewnorm(*self.est).pdf
        else:
            return lambda x: np.full_like(x, np.nan)

    def hist(self, size: int = 1000000, n_bins: int = 200):
        """
        :param size: The defining number of random variates (default is 1,000,000).
        :param n_bins: The number of bins.
        :returns: None. Plots a histogram of the observable.
        """
        y = self.rvs(size)
        n, bins, _ = plt.hist(y, bins=n_bins, density=True, label='Sample data', facecolor='lightgray')
        bins = bins[:-1] + 0.5 * (bins[-1] - bins[0]) / n_bins
        plt.plot(bins, self.pdf()(bins), 'C0', label='Est. PDF')
        plt.legend(loc=1)
        plt.show()


def add(a, b):
    """
    :param a: The first summand.
    :param b: The second summand.
    :returns: The sum of the two summands.
    """
    return a + b


def mul(a, b):
    """
    :param a: The first factor.
    :param b: The second factor.
    :returns: The product of the two factors.
    """
    return a * b


def average(a: array_like, std: array_like = None, cov: array_iter = None, axis: int = None) -> (ndarray, ndarray):
    """
    :param a: The sample data.
    :param axis: The axis along which the average is computed.
    :param std: An array of standard deviations associated with the values in 'a'.
     If specified, the weighted average of the uncorrelated sample data and its standard error is returned.
    :param cov: The covariance matrix associated with the values in 'a'.
     If specified, it overrides 'std' and the weighted average of the correlated sample data
     and its standard error is returned. If no axis is specified, 'cov' must have shape (a.size, a.size)
     with elements associated with the values in the flattened 'a',
     otherwise cov must have either shape (a.shape[axis], a.shape[axis])
     or (..., a.shape[axis - 1], a.shape[axis], a.shape[axis], a.shape[axis + 1], ...).
    :returns: The average and its standard error for a given sample 'a' along the specified 'axis'.
    """
    a = np.asarray(a, dtype=float)
    if std is None and cov is None:  # The average and its standard error.
        av = np.average(a, axis=axis)
        av_d = np.std(a, axis=axis, ddof=1)
        if axis is None:  # The shape of 'a' is ignored ('a' is flattened).
            av_d /= np.sqrt(a.size)
        else:
            av_d /= np.sqrt(a.shape[axis])
    elif std is not None and cov is None:  # The weighted average of uncorrelated data and its standard error.
        std = np.asarray(std, dtype=float)
        av, sum_of_weights = np.average(a, axis=axis, weights=1. / (std ** 2), returned=True)
        av_d = np.sqrt(1. / sum_of_weights)
    else:  # The weighted average of correlated data and its standard error.
        cov = np.asarray(cov, dtype=float)
        if axis is None:  # The shape of 'a' is ignored ('a' is flattened).
            if cov.shape != (a.size, a.size):
                raise ValueError('Shape mismatch between \'a\' {} and \'cov\' {}.'
                                 .format(a.shape, cov.shape))
            av, sum_of_weights = np.average(a, weights=1. / np.diag(cov), returned=True)
            av_d = np.sum([1. / cov[i, i] + 2. * np.sum([cov[i, j] / (cov[i, i] * cov[j, j])
                                                         for j in range(a.size) if i < j])
                           for i in range(a.size)])
        else:
            if a.size != a.shape[0] and cov.shape == (a.size, a.size):
                raise ValueError('Shape mismatch between \'a\' {} and \'cov\' {} for specified axis {}.'
                                 .format(a.shape, cov.shape, axis))
            elif cov.shape == (a.shape[axis], a.shape[axis]):
                av, sum_of_weights = np.average(a, axis=axis, weights=1. / np.diag(cov), returned=True)
                av_d = np.sum([1. / cov[i, i] + 2. * np.sum([cov[i, j] / (cov[i, i] * cov[j, j])
                                                             for j in range(a.shape[axis]) if i < j])
                               for i in range(a.shape[axis])])
            else:
                var = np.diagonal(cov, axis1=axis, axis2=axis + 1)
                axes = [ax for ax in range(len(var.shape))]
                i = axes.pop(-1)
                axes.insert(axis, i)
                var = np.transpose(var, axes=axes)
                sum_of_weights = np.sum(1. / var, axis=axis)
                av = np.sum(a / var, axis=axis) / sum_of_weights
                av_d = np.sum(cov / (np.expand_dims(var, axis=axis + 1) * np.expand_dims(var, axis=axis)),
                              axis=(axis + 1, axis))
        av_d = np.sqrt(av_d) / sum_of_weights
    return av, av_d


def median(a: array_like, axis: int = None) -> (ndarray, ndarray, ndarray):
    """
    :param a: The sample data.
    :param axis: The axis along which the three percentiles are computed.
    :returns: The median (0.5-percentile) as well as the left- (~0.1587) and right-sided (~0.8413) 1-sigma percentile
     of a given sample 'a' along the specified 'axis'.
    """
    med = np.median(a, axis=axis)
    neg = np.percentile(a, 15.8655254, axis=axis)
    pos = np.percentile(a, 84.1344746, axis=axis)
    return med, neg, pos


def estimate_skewnorm(med: scalar, per_0: scalar, per_1: scalar):
    """
    :param med: The median (0.5-percentile) of a random variable.
    :param per_0: The left-sided 1-sigma percentile (~0.1587-percentile) relative to 'med'.
    :param per_1: The right-sided 1-sigma percentile (~0.8413-percentile) relative to 'med'.
    :returns: A size-3-array of the estimated parameters (alpha, mean, std.)
     of a skew normal distribution that matches the given percentiles.
    :raises ValueError: If the ratio between the left-(right-) and right-(left-)sided uncertainty exceeds 1.5.
    """
    if per_0 == 0 or per_1 == 0:
        return None
    if per_0 / per_1 > 1.5 or per_1 / per_0 > 1.5:
        raise ValueError('The ratio between the left-(right-) and right-(left-)sided uncertainty must not exceed 1.5.')
    per = np.array([med, med - per_0, med + per_1])
    y = np.array([0.5, 0.158655254, 0.841344746])

    def f(x):
        return st.skewnorm.cdf(per, *x) - y

    def df(x):
        xi0 = -np.exp(-0.5 * ((per - x[1]) / x[2]) ** 2 * (1. + x[0] ** 2)) / (np.pi * (1. + x[0] ** 2))
        xi1 = -st.skewnorm.pdf(per, *x)
        xi2 = -st.skewnorm.pdf(per, *x) * (per - x[1]) / x[2]
        return np.array([xi0, xi1, xi2]).T

    return root(f, x0=np.array([0., med, per_0]), jac=df).x


def propagate(f: Callable, x: Union[array_like, Observable], x_d: array_like = None, cov: array_iter = None,
              unc_places: int = None, sample_size: int = 1000000, rtol: float = 1e-3, atol: float = None,
              force_sym: bool = False, full_output: bool = False, show: bool = False) -> (Observable, list, ndarray):
    """
    :param f: The function to compute. 'f' needs to be vectorized.
    :param x: The input values. If 'x_d' is None, the sample data will be generated with the 'Observable.rvs' function
     which considers asymmetric uncertainties. If an element is not an 'Observable', its uncertainty is assumed to be 0.
    :param x_d: The uncertainties of the input values.
    :param cov: The covariance matrix of the x values. If not None, 'x' are assumed to be distributed according to
     a multivariate normal distribution with covariance 'cov'.
    :param unc_places: The number of significant decimal places the result will be rounded to.
     If None, the result is not rounded.
    :param sample_size: The number of random variates used for the calculation. The default is 1,000,000.
    :param rtol: The relative tolerance, with respect to the median of the resulting sample,
     with which the left- and right-sided uncertainties can deviate before asymmetric uncertainties are used.
    :param atol: The absolute tolerance with which the left- and right-sided uncertainties can deviate,
     before asymmetric uncertainties are used. Overrides 'rtol'.
    :param force_sym: Whether to force symmetric uncertainties. If so, a normal distribution is assumed.
    :param full_output: Whether to return the randomly generated data samples.
    :param show: Whether to show a histogram and estimated PDFs of the computed sample data.
    :returns: An 'Observable' whose uncertainties result from the propagation of the uncertainties
     of the input values 'x' by function 'f'.
     If the uncertainties are asymmetric, the parameters of a skew normal distribution
     are estimated using least-square fitting and are stored to the observable. The value and the two uncertainties are
     the median and the left- (~0.1587) and right-sided (~0.8413) 1-sigma percentiles
     relative to the median, respectively. If the uncertainties are symmetric the observable
     is assumed to be normally distributed. The value and the single uncertainty is then calculated using
     the mean and the standard deviation of the sampled data. If 'full_output' is True, a list of the input samples
     as well as the output sample are returned along with the observable.
    """
    label = f.__name__
    n_bins = 200
    if cov is None:
        if x_d is None:
            rand_x = [x_i.rvs(sample_size) if isinstance(x_i, Observable) else x_i for x_i in x]
        else:
            x = np.asarray(x, dtype=float)
            x_d = np.asarray(x_d, dtype=float)
            if x.size != x_d.size:
                raise ValueError('x and x_d must have the same size but have sizes {} and {}.'.format(x.size, x_d.size))
            rand_x = [x_i if x_d_i == 0. else st.norm.rvs(loc=x_i, scale=x_d_i, size=sample_size)
                      for x_i, x_d_i in zip(x, x_d)]
    else:
        cov = np.asarray(cov)
        rand_x = st.multivariate_normal.rvs(mean=[float(x_i) for x_i in x], cov=cov, size=sample_size).T
    rand_y = np.asarray(f(*rand_x))

    med, per_0, per_1 = median(rand_y)
    mean = np.average(rand_y)
    std = np.std(rand_y, ddof=1)

    plt.figure(num=1, dpi=96)
    n, bins, _ = plt.hist(rand_y, bins=n_bins, density=True, label='Sample data', facecolor='lightgray')
    bins = bins[:-1] + 0.5 * (bins[-1] - bins[0]) / n_bins

    if force_sym or (atol is None and abs(per_1 - per_0) / med <= rtol) \
            or (atol is not None and abs(per_1 - per_0) <= atol):
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
            popt, _ = curve_fit(st.skewnorm.pdf, bins, n, p0=[0., mean, std])
            ob.set_popt(popt)
        except RuntimeError:
            print('Optimal parameters for a skew normal distribution could not be found through fitting.'
                  ' Using an estimation if possible.')
    
    if show:
        print('Median(sample data): {} (-{} +{})'.format(med, med - per_0, per_1 - med))
        plt.plot(bins, st.norm.pdf(bins, loc=mean, scale=std), 'C0', label='Est. normal')
        if ob.popt is not None:
            med, per_0, per_1 = median(st.skewnorm.rvs(*ob.popt, size=sample_size))
            print('Median(Fit. skew normal): {} (-{} +{})'.format(med, med - per_0, per_1 - med))
            plt.plot(bins, st.skewnorm.pdf(bins, *ob.popt), 'C1', label='Fit. skew normal')
        if ob.est is not None:
            med, per_0, per_1 = median(st.skewnorm.rvs(*ob.est, size=sample_size))
            print('Median(Est. skew normal): {} (-{} +{})'.format(med, med - per_0, per_1 - med))
            plt.plot(bins, st.skewnorm.pdf(bins, *ob.est), 'C2', label='Est. skew normal')
        plt.legend(loc=1)
        plt.show()
    plt.close()

    if full_output:
        return ob, rand_x, rand_y
    else:
        return ob


def propagate_fit(f: Callable, x: array_like, popt: array_like, pcov: array_like, sample_size: int = 1000000):
    """
    :param f: The function to compute. :math:`f` needs to be vectorized.
    :param x: The input values.
    :param popt: An array of the fitted parameters (compare curve_fit).
    :param pcov: A 2d-array of the estimated covariances (compare curve_fit).
    :param sample_size: The number of random variates used for the calculation. The default is 1,000,000.
    :returns: The median and the :math:`1\\sigma` percentiles of the sampled function :math:`f`.
    """
    x = np.asarray(x)
    _x = np.expand_dims(x, axis=-1)
    _popt = st.multivariate_normal.rvs(mean=popt, cov=pcov, size=sample_size).T
    _popt = [np.expand_dims(p, axis=tuple(range(len(x.shape)))) for p in _popt]
    med, per_0, per_1 = median(f(_x, *_popt), axis=-1)
    return med, per_0, per_1


def combined_pdf(z: array_like, pdf_1: Callable = st.norm.pdf, pdf_2: Callable = st.norm.pdf,
                 loc_1: float = 0., scale_1: float = 1., loc_2: float = 0., scale_2: float = 1.,
                 operator: str = '+', n: int = 10) -> ndarray:
    """
    :param z: The quantiles of the combined probability density function (pdf).
    :param pdf_1: The pdf of the first random variate.
    :param pdf_2: The pdf of the second random variate.
    :param loc_1: The 'loc' parameter of the first pdf.
    :param scale_1: The 'scale' parameter of the first pdf.
    :param loc_2: The 'loc' parameter of the second pdf.
    :param scale_2: The 'scale' parameter of the second pdf.
    :param operator: The operator that defines the new random variate given by Z = X <operator> Y.
     Currently supported operators are {'+', '*'}.
    :param n: The precision of the numerical integration.
     The integration uses 2 ** n intervals to evaluate the integral.
    :returns: The value of the pdf at the given 'z' quantiles.
    """
    arg = np.asarray(z)
    if arg.shape == ():
        arg = np.expand_dims(arg, axis=0)
    arg = np.expand_dims(arg, axis=-1)

    if operator == '*':
        left = loc_1 * loc_2 - 5. * (loc_1 * loc_2 - (loc_1 - scale_1) * (loc_2 - scale_2))
        right = loc_1 * loc_2 + 5. * (loc_1 * loc_2 - (loc_1 - scale_1) * (loc_2 - scale_2))

        def kernel(x):
            return pdf_1(x, loc=loc_1, scale=scale_1) * pdf_2(arg / x, loc=loc_2, scale=scale_2) / abs(x)

    elif operator == '+':
        left = loc_1 + loc_2 - 10. * np.sqrt(scale_1 ** 2 + scale_2 ** 2)
        right = loc_1 + loc_2 + 10. * np.sqrt(scale_1 ** 2 + scale_2 ** 2)

        def kernel(x):
            return pdf_1(x, loc=loc_1, scale=scale_1) * pdf_2(arg - x, loc=loc_2, scale=scale_2)

    else:
        raise ValueError('Operator \'{}\' not supported.'.format(operator))

    x_range = np.linspace(left, right, 2 ** n + 1)
    dx = x_range[1] - x_range[0]
    x_range = np.expand_dims(x_range, axis=0)
    y = kernel(x_range)
    result = si.romb(y, dx, axis=1)
    return result


def relevant_interval(dist: Callable, *args, show: bool = False, **kwargs):
    """
    :param dist: The probability distribution function (pdf).
    :param args: Additional arguments for the pdf.
    :param show: Whether to show the pdf in the determined interval.
    :param kwargs: Additional keyword arguments for the pdf.
    :returns: An estimation of the interval where most of the probability lies in.
    """
    k = 10
    scale = 1.

    def cost(_x):
        if _x[0] > _x[1]:
            _y = dist(np.linspace(_x[1], _x[0], 2 ** k + 1), *args, **kwargs)
        else:
            _y = dist(np.linspace(_x[0], _x[1], 2 ** k + 1), *args, **kwargs)
        c = (1. - si.romb(_y, abs(_x[1] - _x[0]) / (2 ** k))) * scale
        if show:
            print(_x, c / scale)
        return c

    def d_cost(_x):
        _y = np.array([dist(_x[0], *args, **kwargs), -dist(_x[1], *args, **kwargs)])
        if _x[0] > _x[1]:
            _y = -_y
        return _y * scale

    x0 = np.array([-1., 1.])
    while cost(x0) < 0.1:
        if show:
            print('x0: ', x0)
        x0 *= 0.5
    while cost(x0) > 0.9:
        if show:
            print('x0: ', x0)
        x0 *= 2.
    scale = x0[1]
    ret = minimize(cost, x0, jac=d_cost).x
    if show:
        x = np.linspace(*ret, 2 ** k + 1)
        y = dist(x, *args, **kwargs)
        plt.plot(x, y, label=dist.__name__)
        plt.legend()
        plt.show()
    return ret


""" Probability density functions """


def uniform(x: array_like, width: array_like) -> ndarray:
    """
    :param x: The x quantiles.
    :param width: The width of the uniform distribution.
    :returns: The probability density at 'x'.
    """
    return st.uniform.pdf(x, loc=-0.5 * width, scale=width)


def _uniform_pumped(x, width, gamma_u, depth) -> ndarray:
    """
    :param x: The x quantiles.
    :param width: The width of the uniform distribution.
    :param gamma_u: The fwhm of the pump transition.
    :param depth: The depth of the pump dip in parts of the height of the underlying uniform distribution.
    :returns: A not normalized uniform distribution with a Lorentz shaped dip.
    """
    x = np.asarray(x)
    scalar_true = tools.check_shape((), x, return_mode=True)
    if scalar_true:
        x = np.array([x])
    uni = uniform(x, width)
    nonzero = np.nonzero(uni)
    ret = np.zeros_like(x)
    ret[nonzero] = uni[nonzero] - depth / width * (np.pi * gamma_u / 2.) * st.cauchy.pdf(x[nonzero], scale=gamma_u / 2.)
    if scalar_true:
        return ret[0]
    return ret


def uniform_pumped(x, width, gamma_u, depth) -> ndarray:
    """
    :param x: The x quantiles.
    :param width: The width of the uniform distribution.
    :param gamma_u: The fwhm of the pump transition.
    :param depth: The depth of the pump dip in parts of the height of the underlying uniform distribution.
    :returns: A uniform distribution with a Lorentz-shaped dip.
    """
    _x = np.linspace(-0.5 * width, 0.5 * width, 2 ** 10 + 1)
    _y = _uniform_pumped(_x, width, gamma_u, depth)
    norm = si.romb(_y, width / (2 ** 10))
    return _uniform_pumped(x, width, gamma_u, depth) / norm


def info():
    """
    Plots the 1-sigma percentiles (and their ratio) of a skew normal distribution relative to its median
    for different values of the parameter alpha.
    The plot shows that the ratio between the two 1-sigma percentiles cannot exceed 1.5.

    :returns: None.
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
    plt.title('Properties of the skew normal distribution')
    plt.xlabel(r'Parameter $\alpha$')
    plt.ylabel(r'Percentiles ($\sigma$) / Ratio of Percentiles')
    plt.ylim(0., 1.75)
    plt.plot(a, neg, label='left 1-sigma')
    plt.plot(a, pos, label='right 1-sigma')
    plt.plot(a, pos / neg, label='Ratio right/left')
    plt.plot(a, neg / pos, label='Ratio left/right')
    plt.legend(loc='lower center')
    plt.show()
