# -*- coding: utf-8 -*-
"""
pycol.analyze

Created on 07.05.2020

@author: Patrick Mueller

Module for analysing/evaluating/fitting data.

linear regression algorithms:
    - york(); [York et al., Am. J. Phys. 72, 367 (2004)]
    - linear_monte_carlo(); based on [Gebert et al., Phys. Rev. Lett. 115, 053003 (2015), Suppl.]
    - linear_monte_carlo_nd(); based on [Gebert et al., Phys. Rev. Lett. 115, 053003 (2015), Suppl.]

curve fitting methods:
    - curve_fit(); Encapsulates the scipy.optimize.curve_fit method to allow fixing parameters.
    - odr_fit(); Encapsulates the scipy.odr.odr method to accept inputs similarly to curve_fit().

classes:
    (- Element; Holds spectroscopic information about a chemical element.)
    (- Radii; Holds information about nuclear charge radii of an chemical element.)
    - King; Creates a King plot with isotope shifts or nuclear charge radii.

"""

import inspect
import numpy as np
import scipy.stats as st
import scipy.odr as odr
import scipy.optimize as so
import matplotlib.pyplot as plt

from .types import *
from . import tools
from . import physics as ph
# from .stats import average
# from . import databases as dat


def _get_rtype2(ab_dict: dict, a: Union[Iterable, any], b: Union[Iterable, any], rtype: type = ndarray):
    """
    :param ab_dict: The nested dictionary of depth 2, where values are extracted from.
    :param a: The list of keys of the first level of 'ab_dict'.
    :param b: The list of keys of the second level of 'ab_dict'.
    :param rtype: The type in which the results will be returned. Currently supported special types are [dict, ].
     The standard 'rtype' is a numpy array.
    :returns: A subset of the dictionary 'ab_dict' as a dictionary or a numpy array.
    :raises: ValueError . 'a' and 'b' must be Non-Iterables or an Iterable of the corresponding types.
    :raises: KeyError . ab_dict[a][b] must exist for all a and b.
    """
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    a_dim, line_dim = len(a_arr.shape), len(b_arr.shape)
    if a_dim == 0 and line_dim == 0:
        if rtype == dict:
            return {a: {b: ab_dict[a][b]}}
        else:
            return ab_dict[a][b]
    elif a_dim == 0 and line_dim == 1:
        if rtype == dict:
            return {b_i: ab_dict[a][b_i] for b_i in b}
        else:
            return np.array([ab_dict[a][b_i] for b_i in b])
    elif a_dim == 1 and line_dim == 0:
        if rtype == dict:
            return {a_i: ab_dict[a_i][b] for a_i in a}
        else:
            return np.array([ab_dict[a_i][b] for a_i in a])
    elif a_dim == 1 and line_dim == 1:
        if rtype == dict:
            return {a_i: {b_i: ab_dict[a_i][b_i] for b_i in b} for a_i in a}
        else:
            return np.array([[ab_dict[a_i][b_i] for b_i in b] for a_i in a])
    else:
        raise ValueError('\'a\' and \'b\' must be Non-Iterables or an Iterable of the corresponding types.')


def straight(x: array_like, a: array_like, b: array_like) -> ndarray:
    """
    :param x: The x values.
    :param a: The y-intercept.
    :param b: The slope.
    :returns: The y values resulting from the 'x' values via the given linear relation.
    """
    x, a, b = np.asarray(x), np.asarray(a), np.asarray(b)
    return a + b * x


def straight_slope(p_0: array_iter, p_1: array_iter, axis=-1) -> ndarray:
    """
    :param p_0: The first point(s).
    :param p_1: The second point(s).
    :param axis: The axis along which the vector components are aligned.
    :returns: The parametrization of a straight line in n dimensions.
    """
    p_0, p_1 = np.asarray(p_0), np.asarray(p_1)
    dr = p_1 - p_0
    return dr / np.expand_dims(tools.absolute(dr, axis=axis), axis=axis)


def straight_std(x: array_like, sigma_a: array_like, sigma_b: array_like, corr_ab: array_like) -> ndarray:
    """
    :param x: The x values.
    :param sigma_a: The standard deviation of the y-intercept.
    :param sigma_b: The standard deviation of the slope.
    :param corr_ab: The correlation coefficient between the slope and y-intercept.
    :returns: The standard deviation of a straight line where the x values do not have uncertainties.
    """
    x = np.asarray(x)
    sigma_a, sigma_b, corr_ab = np.asarray(sigma_a), np.asarray(sigma_b), np.asarray(corr_ab)
    return np.sqrt(sigma_a ** 2 + (x * sigma_b) ** 2 + 2 * x * sigma_a * sigma_b * corr_ab)


def straight_x_std(x: array_like, b: array_like, sigma_x: array_like,
                   sigma_a: array_like, sigma_b: array_like, corr_ab: array_like) -> ndarray:
    """
    :param x: The x values.
    :param b: The slope.
    :param sigma_x: The standard deviation of the x values.
    :param sigma_a: The standard deviation of the y-intercept.
    :param sigma_b: The standard deviation of the slope.
    :param corr_ab: The correlation coefficient between the slope and y-intercept.
    :returns: The standard deviation of a straight line where all input values have uncertainties.
    """
    x, sigma_x = np.asarray(x), np.asarray(sigma_x)
    b, sigma_a, sigma_b, corr_ab = np.asarray(b), np.asarray(sigma_a), np.asarray(sigma_b), np.asarray(corr_ab)
    return np.sqrt(sigma_a ** 2 + (x * sigma_b) ** 2 + 2 * x * sigma_a * sigma_b * corr_ab
                   + (b * sigma_x) ** 2 + (sigma_b * sigma_x) ** 2)


def draw_straight_unc_area(x: array_like, y: array_like, sigma_a: array_like, sigma_b: array_like,
                           corr_ab: array_like, **kwargs):
    """
    :param x: The x values.
    :param y: The y values.
    :param sigma_a: The standard deviation of the y-intercept.
    :param sigma_b: The standard deviation of the slope.
    :param corr_ab: The correlation coefficient between the slope and y-intercept.
    :param kwargs: The keyword arguments for the fill_between function.
    :returns: The standard deviation of a straight line where the x values do not have uncertainties.
    """
    unc = straight_std(x, sigma_a, sigma_b, corr_ab)
    plt.fill_between(x, y - unc, y + unc, **kwargs)


def cubic(x, a, b, c, d):
    return a + b * x + 0.5 * c * x ** 2 + 1. / 6. * d * x ** 3


def ellipse2d(x: array_like, y: array_like, scale_x: array_like, scale_y: array_like,
              phi: array_like, corr: array_like) -> (ndarray, ndarray):
    """
    :param x: The x-component of the position of the ellipse.
    :param y: The y-component of the position of the ellipse.
    :param scale_x: The amplitude of the x-component.
    :param scale_y: The amplitude of the y-component.
    :param phi: The angle between the vector to the point on the ellipse and the x-axis.
    :param corr: The correlation coefficient between the x and y data.
    :returns: A point on an ellipse in 2d-space with amplitudes 'x', 'y'
     and correlation 'corr' between x- and y-component.
    """
    x, y, scale_x, scale_y, corr = np.asarray(x), np.asarray(y), \
        np.asarray(scale_x), np.asarray(scale_y), np.asarray(corr)
    return x + scale_x * np.cos(phi), y + scale_y * (corr * np.cos(phi) + np.sqrt(1 - corr ** 2) * np.sin(phi))


def draw_sigma2d(x: array_iter, y: array_iter, sigma_x: array_iter, sigma_y: array_iter,
                 corr: array_iter, n: int = 1, **kwargs):
    """
    :param x: The x data.
    :param y: The y data.
    :param sigma_x: The 1-sigma uncertainties of the x data.
    :param sigma_y: The 1-sigma uncertainties of the y data.
    :param corr: The correlation coefficients between the x and y data.
    :param n: The maximum sigma region to draw
    :param kwargs: Additional keyword arguments are passed to plt.plot().
    :returns: None. Draws the sigma-bounds of the given data points (x, y) until the n-sigma region.
    """
    phi = np.arange(0., 2 * np.pi, 0.001)
    for x_i, y_i, s_x, s_y, r in zip(x, y, sigma_x, sigma_y, corr):
        for i in range(1, n + 1, 1):
            plt.plot(*ellipse2d(x_i, y_i, i * s_x, i * s_y, phi, r), 'k-', **kwargs)


def weight(sigma):
    """
    :param sigma: The 1-sigma uncertainty.
    :returns: The weight corresponding to the 1-sigma uncertainty 'sigma'
    """
    return 1. / sigma ** 2


def york(x: array_iter, y: array_iter, sigma_x: array_iter = None, sigma_y: array_iter = None,
         corr: array_iter = None, iter_max: int = 200, report: bool = False, show: bool = False):
    """
    A linear regression algorithm to find the best straight line, given normally distributed errors for x and y
    and correlation coefficients between errors in x and y. The algorithm is described in
    ['Unified equations for the slope, intercept, and standard errors of the best straight line',
    York et al., American Journal of Physics 72, 367 (2004)]. See the comments to compare the individual steps.

    :param x: The x data.
    :param y: The y data.
    :param sigma_x: The 1-sigma uncertainties of the x data.
    :param sigma_y: The 1-sigma uncertainties of the y data.
    :param corr: The correlation coefficients between the x and y data.
    :param iter_max: The maximum number of iterations to find the best slope.
    :param report: Whether to print the result of the fit.
    :param show: Whether to plot the fit result.
    :returns: a, b, sigma_a, sigma_b, corr_ab. The best y-intercept and slope,
     their respective 1-sigma uncertainties and their correlation coefficient.
    """
    x, y = np.asarray(x), np.asarray(y)
    if sigma_x is None:
        sigma_x = np.full_like(x, 1.)
        if report:
            print('\nNo uncertainties for \'x\' were given. Assuming \'sigma_x\'=1.')
    if sigma_y is None:
        sigma_y = np.full_like(y, 1.)
        if report:
            print('\nNo uncertainties for \'y\' were given. Assuming \'sigma_y\'=1.')
    sigma_2d = True
    if corr is None:
        sigma_2d = False
        corr = 0.

    sigma_x, sigma_y, corr = np.asarray(sigma_x), np.asarray(sigma_y), np.asarray(corr)

    p_opt, _ = so.curve_fit(straight, x, y, p0=[0., 1.], sigma=sigma_y)  # (1)
    b_init = p_opt[1]
    b = b_init

    w_x, w_y = weight(sigma_x), weight(sigma_y)  # w(X_i), w(Y_i), (2)
    alpha = np.sqrt(w_x * w_y)

    n = 0
    r_tol = 1e-15
    tol = 1.
    mod_w = None
    x_bar, y_bar = None, None
    beta = None
    while tol > r_tol and n < iter_max:  # (6)
        b_init = b
        mod_w = w_x * w_y / (w_x + b_init ** 2 * w_y - 2. * b_init * corr * alpha)  # W_i, (3)
        x_bar, y_bar = np.average(x, weights=mod_w), np.average(y, weights=mod_w)  # (4)
        u = x - x_bar
        v = y - y_bar
        beta = mod_w * (u / w_y + v * b_init / w_x - straight(u, v, b_init) * corr / alpha)
        b = np.sum(mod_w * beta * v) / np.sum(mod_w * beta * u)  # (5)
        tol = abs((b - b_init) / b)
        n += 1  # (6)

    a = y_bar - b * x_bar  # (7)
    x_i = x_bar + beta  # (8)
    x_bar = np.average(x_i, weights=mod_w)  # (9)
    u = x_i - x_bar
    sigma_b = np.sqrt(1. / np.sum(mod_w * u ** 2))  # (10)
    sigma_a = np.sqrt(1. / np.sum(mod_w) + x_bar ** 2 * sigma_b ** 2)
    corr_ab = -x_bar * sigma_b / sigma_a
    chi2 = np.sum(mod_w * (y - b * x - a) ** 2) / (x.size - 2)

    if report:
        if n == iter_max:
            print('\nMaximum number of iterations ({}) was reached.'.format(iter_max))
        tools.printh('York-fit result:')
        print('a: {} +/- {}\nb: {} +/- {}\ncorr_ab: {}\nchi2: {}'
              .format(a, sigma_a, b, sigma_b, corr_ab, chi2))
    if show:
        x_cont = np.linspace(np.min(x) - 0.1 * (np.max(x) - np.min(x)), np.max(x) + 0.1 * (np.max(x) - np.min(x)), 1001)
        plt.xlabel('x')
        plt.ylabel('y')
        if sigma_2d:
            plt.plot(x, y, 'k.', label='Data')
            draw_sigma2d(x, y, sigma_x, sigma_y, corr, n=1)
        else:
            plt.errorbar(x, y, xerr=sigma_x, yerr=sigma_y, fmt='k.', label='Data')
        plt.plot(x_cont, straight(x_cont, a, b), 'b-', label='Fit')
        y_min = straight(x_cont, a, b) - straight_std(x_cont, sigma_a, sigma_b, corr_ab)
        y_max = straight(x_cont, a, b) + straight_std(x_cont, sigma_a, sigma_b, corr_ab)
        plt.fill_between(x_cont, y_min, y_max, color='b', alpha=0.3, antialiased=True)
        plt.legend(loc='best', numpoints=1)
        plt.show()

    return a, b, sigma_a, sigma_b, corr_ab


def linear_monte_carlo(x: array_iter, y: array_iter, sigma_x: array_iter = None, sigma_y: array_iter = None,
                       corr: array_iter = None, n: int = 1000000, report: bool = True, show: bool = False):
    """
    :param x: The x data.
    :param y: The y data.
    :param sigma_x: The 1-sigma uncertainties of the x data.
    :param sigma_y: The 1-sigma uncertainties of the y data.
    :param corr: The correlation coefficients between the x and y data.
    :param n: The number of samples generated for each data point.
    :param report: Whether to print the result of the fit.
    :param show: Whether to plot the fit result.
    :returns: a, b, sigma_a, sigma_b, corr_ab. The best y-intercept and slope,
     their respective 1-sigma uncertainties and their correlation coefficient.
    """
    if sigma_x is None:
        sigma_x = [1., ] * len(x)
        if report:
            print('\nNo uncertainties for \'x\' were given. Assuming \'sigma_x\'=1.')
    if sigma_y is None:
        sigma_y = [1., ] * len(y)
        if report:
            print('\nNo uncertainties for \'y\' were given. Assuming \'sigma_y\'=1.')
    x, y, sigma_x, sigma_y, corr = np.asarray(x), np.asarray(y),\
        np.asarray(sigma_x), np.asarray(sigma_y), np.asarray(corr)
    tools.check_shape_like(x, y, sigma_x, sigma_y, corr, allow_scalar=False)
    tools.check_dimension(x.size, 0, x, y, sigma_x, sigma_y, corr)
    mean = np.concatenate((np.expand_dims(x, axis=1), np.expand_dims(y, axis=1)), axis=1)
    cov = np.array([[[s_x ** 2, r * s_x * s_y],
                     [r * s_y * s_x, s_y ** 2]] for s_x, s_y, r in zip(sigma_x, sigma_y, corr)])
    a, b, sigma_a, sigma_b, corr_ab = linear_monte_carlo_nd(mean, cov=cov, axis=0, n=n, report=report, show=show)
    return a[0], b[0], sigma_a[0], sigma_b[0], corr_ab[0, 1]


def _test_order_linear_monte_carlo_nd(x: ndarray, cov: ndarray, n: int):
    """
    :param x: The x data. 'x' is a numpy array of vectors with arbitrary but fixed dimension.
    :param cov: The covariance matrices of the x data vectors. Must have shape (x.shape[0], x.shape[1], x.shape[1]).
    :param n: The number of samples generated for each data point.
    :returns: The order of axis 0 of 'x' which yields the most accepted samples.
    """
    indices = [(i, j) for i in range(x.shape[0]) for j in range(x.shape[0]) if j > i]
    best_n = 0
    best_order = np.arange(x.shape[0], dtype=int)
    for (i, j) in indices:
        order = np.array([i, ] + [k for k in range(x.shape[0]) if k != i and k != j] + [j, ])
        n_accepted = np.sum(_generate_collinear_points(x[order], cov[order], n)[2])
        # print(n_accepted)
        if n_accepted > best_n:
            best_n = n_accepted
            best_order = order
    # print(best_n, best_order)
    return best_order


def _generate_collinear_points(x: ndarray, cov: ndarray, n: int):
    """
    :param x: The x data. 'x' is a numpy array of vectors with arbitrary but fixed dimension.
    :param cov: The covariance matrices of the x data vectors. Must have shape (x.shape[0], x.shape[1], x.shape[1]).
    :param n: The number of samples generated for each data point.
    :returns: The randomly generated data vectors p [shape=(n, )+ x.shape] aligned along a straight line,
     the unit vectors dr [shape=(n, x.shape[1])] in the directions of the straight lines
     and a mask of the accepted samples. The accepted samples are p[accepted] and dr[accepted].
    """
    p_0 = st.multivariate_normal.rvs(mean=x[0], cov=cov[0], size=n)
    p_1 = st.multivariate_normal.rvs(mean=x[-1], cov=cov[-1], size=n)
    dr = straight_slope(p_0, p_1, axis=-1)

    t_0 = [np.einsum('ij,ij->i', np.expand_dims(x_i, axis=0) - p_0, dr) for x_i in x[1:-1]]
    t = [st.norm.rvs(loc=t_0_i, scale=np.sqrt(np.min(np.diag(cov_i)))) for t_0_i, cov_i in zip(t_0, cov[1:-1])]

    p_new = [p_0 + np.expand_dims(t_i, axis=-1) * dr for t_i in t]
    f = np.prod([st.multivariate_normal.pdf(p_i, mean=x_i, cov=cov_i)
                 for p_i, x_i, cov_i in zip(p_new, x[1:-1], cov[1:-1])], axis=0)

    g = np.prod([st.norm.pdf(t_i, loc=t_0_i, scale=np.sqrt(np.max(np.diag(cov_i))))
                 for t_i, t_0_i, cov_i in zip(t, t_0, cov[1:-1])], axis=0)
    if p_new:
        u = f / g
        u /= np.max(u)
        accepted = np.random.random(size=n) < u
        p = np.concatenate((np.expand_dims(p_0, 0), p_new, np.expand_dims(p_1, 0)), axis=0)
    else:
        accepted = np.full(n, True, dtype=bool)
        p = np.concatenate((np.expand_dims(p_0, 0), np.expand_dims(p_1, 0)), axis=0)
    p = np.transpose(p, axes=[1, 0, 2])
    return p, dr, accepted


def linear_monte_carlo_nd(x: array_iter, cov: array_iter = None, axis: int = None,
                          n: int = 1000000, full_output: bool = False, report: bool = True, show: bool = False):
    """
    A Monte-Carlo fitter that finds a straight line in n-dimensional space.

    :param x: The x data. 'x' is an iterable of vectors with arbitrary but fixed dimension.
    :param cov: The covariance matrices of the x data vectors. Must have shape (x.shape[0], x.shape[1], x.shape[1]).
    :param axis: The axis to use for the parametrization. If None, the directional vector is normalized.
    :param n: The number of samples generated for each data point.
    :param full_output: Whether to also return the generated points 'p' and a mask for the 'accepted' points.
     The accepted points are p[accepted]. p has shape (n, x.shape).
    :param report: Whether to print the result of the fit.
    :param show: Whether to plot the fit result.
    :returns: A list of results as defined by the routine 'linear_monte_carlo_nd':
     A tuple (a, b, sigma_a, sigma_b, corr_ab) of arrays. The best y-intercepts and slopes,
     their respective 1-sigma uncertainties and their correlation matrix.
     The y-intercepts and slopes are defined through 'axis' by a(x[axis] == 0)
     and b(dx[i != axis] / dx[axis]), respectively.
     Additionally, a second tuple with the accepted points, offsets, slopes and a mask
     for the accepted points is returned if full_output == True.
    """
    x = np.asarray(x)
    if len(x.shape) != 2:
        raise ValueError('x must be an iterable of vectors with arbitrary but fixed dimension.'
                         ' Hence, len(x.shape) =: 2, but is {}.'.format(len(x.shape)))
    if cov is None:
        cov = [np.identity(x.shape[1]), ] * x.shape[0]
    cov = np.asarray(cov)
    tools.check_shape((x.shape[0], x.shape[1], x.shape[1]), cov, allow_scalar=False)
    order = _test_order_linear_monte_carlo_nd(x, cov, 100000)
    inverse_order = np.array([int(np.where(order == i)[0])
                              for i in range(order.size)])  # Invert the order for later.
    p, dr, accepted = _generate_collinear_points(x[order], cov[order], n)
    p = p[:, inverse_order, :]
    n_accepted = np.sum(accepted)
    p_0 = p[accepted, 0, :]
    dr = dr[accepted]

    if show:
        for sample in range(0, int(n_accepted), int(n_accepted / 10)):
            x_plot = p[accepted][sample, :, 0]
            y_plot = p[accepted][sample, :, 1]
            plot_order = np.argsort(x_plot)
            x_plot = x_plot[plot_order]
            y_plot = y_plot[plot_order]
            plt.plot([x_plot[0], x_plot[-1]], [y_plot[0], y_plot[-1]], 'C0-')
            plt.plot(x_plot, y_plot, 'C1.')
        # plt.errorbar(x[:, 0], x[:, 1], xerr=np.sqrt(cov[:, 0, 0]), yerr=np.sqrt(cov[:, 1, 1]), fmt='k.')
        draw_sigma2d(x[:, 0], x[:, 1], np.sqrt(cov[:, 0, 0]), np.sqrt(cov[:, 1, 1]),
                     cov[:, 0, 1] / (cov[:, 0, 0] * cov[:, 1, 1]), n=2)
        # plt.gca().set_aspect('equal')
        plt.show()

    mask = [True, ] * x.shape[1]
    if axis is not None:
        if axis == -1:
            axis = x.shape[1] - 1
        mask = np.arange(x.shape[1]) != axis
        t = -p_0[:, axis] / dr[:, axis]
        p_0 = p_0 + np.expand_dims(t, axis=-1) * dr
        dr = dr / dr[:, [axis]]
    a = np.mean(p_0, axis=0)[mask]
    b = np.mean(dr, axis=0)[mask]
    sigma_a = np.std(p_0, axis=0, ddof=1)[mask]
    sigma_b = np.std(dr, axis=0, ddof=1)[mask]
    corr_ab = np.corrcoef(p_0.T[mask], dr.T[mask])
    # print(dr.shape)
    # plt.plot(p[accepted, :, 0], dr[:, 1], '.')
    # plt.show()
    if report:
        tools.printh('Linear Monte-Carlo-fit result:')
        print('Accepted samples: {} / {} ({}%)\na: {} +- {}\nb: {} +- {}\ncorr_ab:\n{}'
              .format(n_accepted, n, np.around(n_accepted / n * 100, decimals=2),
                      a, sigma_a, b, sigma_b, corr_ab))
    if full_output:
        return (a, b, sigma_a, sigma_b, corr_ab), (p[accepted], p_0, dr, accepted)
    return a, b, sigma_a, sigma_b, corr_ab


def linear_alpha(x: array_iter, y: array_iter, sigma_x: array_like = None, sigma_y: array_like = None,
                 corr: array_iter = None, func: Callable = york, alpha: scalar = 0, find_alpha: bool = True,
                 report: bool = True, show: bool = False, **kwargs):
    """
    :param x: The x data.
    :param y: The y data.
    :param sigma_x: The 1-sigma uncertainty of the x data.
    :param sigma_y: The 1-sigma uncertainty of the y data.
    :param corr: The correlation coefficients between the x and y data.
    :param func: The fitting routine.
    :param alpha: An x-axis offset to reduce the correlation coefficient between the y-intercept and the slope.
    :param find_alpha: Whether to search for the best 'alpha'. Uses the given 'alpha' as a starting point.
     May not give the desired result if 'alpha' was initialized to far from its optimal value.
    :param report: Whether to print the result of the fit.
    :param show: Whether to plot the fit result.
    :param kwargs: Additional keyword arguments are passed to the fitting routine.
    :returns: a, b, sigma_a, sigma_b, corr_ab, alpha. The best y-intercept and slope,
     their respective 1-sigma uncertainties, their correlation coefficient and the used alpha.
    """
    n = tools.floor_log10(alpha)

    def cost(x0):
        _, _, _, _, c = func(x - x0[0], y, sigma_x=sigma_x, sigma_y=sigma_y, corr=corr,
                             report=False, show=False, **kwargs)
        print('alpha: {}(1.), \tcost: {}({})'.format(x0[0], c ** 2 * 10. ** (n + 1), 10. ** (n - 3)))
        return c ** 2 * 10. ** (n + 1)

    if find_alpha:
        alpha = so.minimize(cost, np.array([alpha]), method='Nelder-Mead',
                            options={'xatol': 1., 'fatol': 0.01 ** 2 * 10. ** (n + 1)}).x[0]

    a, b, sigma_a, sigma_b, corr_ab =\
        func(x - alpha, y, sigma_x=sigma_x, sigma_y=sigma_y, corr=corr,
             report=report, show=show, **kwargs)
    if report:
        print('alpha: {}'.format(alpha))
    return a, b, sigma_a, sigma_b, corr_ab, alpha


def curve_fit(f: Callable, x: Union[array_like, object], y: array_like, p0: array_iter = None,
              p0_fixed: array_iter = None, sigma: array_iter = None, absolute_sigma: bool = False,
              check_finite: bool = True, bounds: (ndarray, ndarray) = (-np.inf, np.inf), method: str = None,
              jac: Union[Callable, str] = None, report: bool = False, **kwargs) -> (ndarray, ndarray):
    """
    :param f: The model function to fit to the data.
    :param x: The x data.
    :param y: The y data.
    :param p0: A numpy array or an Iterable of the initial guesses for the parameters.
     Must have at least the same length as the minimum number of parameters required by the function 'f'.
     If 'p0' is None, 1 is taken as an initial guess for all non-keyword parameters.
    :param p0_fixed: A numpy array or an Iterable of bool values specifying, whether to fix a parameter.
     Must have the same length as p0.
    :param sigma: The 1-sigma uncertainty of the y data.
    :param absolute_sigma: See scipy.optimize.curve_fit.
    :param check_finite: See scipy.optimize.curve_fit.
    :param bounds: See scipy.optimize.curve_fit.
    :param method: See scipy.optimize.curve_fit.
    :param jac: See scipy.optimize.curve_fit.
    :param report: Whether to print the result of the fit.
    :param kwargs: See scipy.optimize.curve_fit.
    :returns: popt, pcov. The optimal parameters and their covariance matrix.
    """
    if p0_fixed is None or all(not p for p in p0_fixed):
        popt, pcov = so.curve_fit(f, x, y, p0=p0, sigma=sigma, absolute_sigma=absolute_sigma,
                                  check_finite=check_finite, bounds=bounds, method=method, jac=jac, **kwargs)
    elif p0 is None:
        raise ValueError('Please specify the initial parameters when any of the parameters shall be fixed.')
    else:
        # noinspection PyTypeChecker
        p0, p0_fixed = np.asarray(p0), np.asarray(p0_fixed).astype(bool)
        if p0_fixed.shape != p0.shape:
            raise ValueError('\'p0_fixed\' must have the same shape as \'p0\'.')
        _p0 = p0[~p0_fixed]
        _bounds = (np.asarray(bounds[0]), np.asarray(bounds[1]))
        if len(_bounds[0].shape) > 0 and _bounds[0].size == p0.size:
            _bounds = (_bounds[0][~p0_fixed], _bounds[1][~p0_fixed])

        def _f(_x, *args):
            _args = p0
            _args[~p0_fixed] = np.asarray(args)
            return f(_x, *_args)

        popt, pcov = np.ones_like(p0, dtype=float), np.zeros((p0.size, p0.size), dtype=float)
        pcov_mask = ~(np.expand_dims(p0_fixed, axis=1) + np.expand_dims(p0_fixed, axis=0))
        popt[p0_fixed] = p0[p0_fixed]
        _popt, _pcov = so.curve_fit(_f, x, y, p0=_p0, sigma=sigma, absolute_sigma=absolute_sigma,
                                    check_finite=check_finite, bounds=_bounds, method=method, jac=jac, **kwargs)
        popt[~p0_fixed] = _popt
        # noinspection PyUnresolvedReferences
        pcov[pcov_mask] = _pcov.flatten()

    if report:
        tools.printh('curve_fit result:')
        for i, (val, err) in enumerate(zip(popt, np.sqrt(np.diag(pcov)))):
            print('{}: {} +/- {}'.format(i, val, err))
    return popt, pcov


def odr_fit(f: Callable, x: array_iter, y: array_iter, sigma_x: array_iter = None, sigma_y: array_iter = None,
            p0: array_iter = None, p0_d: array_iter = None, p0_fixed: array_iter = None,
            report: bool = False, **kwargs) -> (ndarray, ndarray):
    """
    :param f: The model function to fit to the data.
    :param x: The x data.
    :param y: The y data.
    :param sigma_x: The 1-sigma uncertainty of the x data.
    :param sigma_y: The 1-sigma uncertainty of the y data.
    :param p0: A numpy array or an Iterable of the initial guesses for the parameters.
     Must have at least the same length as the minimum number of parameters required by the function 'f'.
     If 'p0' is None, 1 is taken as an initial guess for all non-keyword parameters.
    :param p0_d: A numpy array or an Iterable of the uncertainties of the initial guesses for the parameters.
     Must have the same length as p0.
    :param p0_fixed: A numpy array or an Iterable of bool values specifying, whether to fix a parameter.
     Must have the same length as p0.
    :param report: Whether to print the result of the fit.
    :param kwargs: Keyword arguments passed to odr.ODR.
    :returns: popt, pcov. The optimal parameters and their covariance matrix.
    """
    x, y = np.asarray(x), np.asarray(y)
    sx, sy, covx, covy = None, None, None, None
    if sigma_x is not None:
        sx = np.asarray(sigma_x)
        # if sigma_x.shape == (x.size, x.size):
        #     covx = sigma_x
        #     sy = None
    if sigma_y is not None:
        sy = np.asarray(sigma_y)
        # if sigma_y.shape == (y.size, y.size):
        #     covy = sigma_y
        #     sy = None
    if p0 is None:
        arg_spec = inspect.getfullargspec(f)
        n = len(arg_spec.args) - 1
        if (n < 1 and arg_spec.defaults is None) or (arg_spec.defaults is not None and n - len(arg_spec.defaults) < 1):
            raise ValueError('\'f\' must have at least one parameter that can be optimized.')
        p0 = [1., ] * n
        if arg_spec.defaults is not None:
            p0 = [1., ] * (n - len(arg_spec.defaults))
    p0 = np.asarray(p0)

    ifixb = None
    if p0_fixed is not None:
        ifixb = (~np.asarray(p0_fixed)).astype(int)
        if ifixb.shape != p0.shape:
            raise ValueError('\'p0_fixed\' must have the same shape as \'p0\'.')

    data = odr.RealData(x, y, sx=sx, sy=sy, covx=covx, covy=covy)
    # noinspection PyTypeChecker
    model = odr.Model(lambda beta, x_i: f(x_i, *beta))
    odr_i = odr.ODR(data, model, beta0=p0, delta0=p0_d, ifixb=ifixb, **kwargs)
    out = odr_i.run()

    if report:
        tools.printh('odr_fit result:')
        for i, (val, err) in enumerate(zip(out.beta, np.sqrt(np.diag(out.cov_beta)))):
            print('{}: {} +/- {}'.format(i, val, err))
    return out.beta, out.cov_beta


# class Element:
#     """
#     Class representing a single chemical element. The user can define an electronic transition
#      to extract all spectroscopy data for that transition available in the users 'AtomicDatabase.db' database.
#      The transition frequencies and isotope shifts are stored inside a nested dictionary
#      with isotopes and the user-defined line labels as keys.
#     """
#
#     def __init__(self, label, decimals=3):
#         """
#         :param label: The label of a chemical element.
#         :param decimals: The number of decimal places (sub-MHz places) to use for frequency values.
#         """
#         self.label = label
#         self.decimals = decimals
#         self.isotopes = {iso['A']: iso for iso in dat.get_table_entries('Isotopes', {'element': self.label})}
#         self.lines = {}
#         self.a_ref = None
#         self.f_ref = {}
#         self.df_ref = {}
#         self.f = {}
#         self.refs_f = {}
#         self.df = {}
#         self.refs_df = {}
#
#     def check_ref_args(self, references: Union[array_iter, dict, str]) -> Union[dict, None]:
#         """
#         :param references: A reference str, iterable of reference str's
#          or a dictionary of these with isotopes or lines as keys.
#         :returns: A dictionary of the specified references which has the structure {a: {line: [references, ]}}.
#         """
#         if references is None:
#             return None
#         if isinstance(references, dict):
#             if any(a not in self.isotopes.keys() for a in references.keys()):
#                 if any(line not in self.lines.keys() for line in references.keys()):
#                     return None
#                 else:
#                     if not all(isinstance(iso_dict, dict) for iso_dict in references.values()):
#                         refs = {a: {line: tools.check_iterable(references[line]) if line in references.keys() else []
#                                     for line in self.lines.keys()}
#                                 for a in self.isotopes.keys()}
#                     else:
#                         refs = {a: {line: tools.check_iterable(references[line][a])
#                                     if line in references.keys() and a in references[line].keys() else []
#                                     for line in self.lines.keys()}
#                                 for a in self.isotopes.keys()}
#             else:
#                 if not all(isinstance(line_dict, dict) for line_dict in references.values()):
#                     refs = {a: {line: tools.check_iterable(references[a]) if a in references.keys() else []
#                                 for line in self.lines.keys()}
#                             for a in self.isotopes.keys()}
#                 else:
#                     refs = {a: {line: tools.check_iterable(references[a][line])
#                                 if a in references.keys() and line in references[a].keys() else []
#                                 for line in self.lines.keys()}
#                             for a in self.isotopes.keys()}
#         else:
#             refs = {a: {line: tools.check_iterable(references) for line in self.lines.keys()}
#                     for a in self.isotopes.keys()}
#         return refs
#
#     def define_line(self, label: str, q: int,
#                     config_l: str, multiplet_l: int, l_l: int, j_l: float,
#                     config_u: str, multiplet_u: int, l_u: int, j_u: float):
#         """
#         :param label: The user-defined name of the line.
#         :param q: The charge state of the isotope (e).
#         :param config_l: The electronic configuration of the lower state.
#         :param multiplet_l: The spin multiplet of the lower state.
#         :param l_l: The angular momentum quantum number L of the lower state.
#         :param j_l: The total angular momentum quantum number J of the lower state.
#         :param config_u: The electronic configuration of the upper state.
#         :param multiplet_u: The spin multiplet of the upper state.
#         :param l_u: The angular momentum quantum number L of the upper state.
#         :param j_u: The total angular momentum quantum number J of the upper state.
#         :returns: None. Extracts the available electronic transition/isotope shift entries for the specified line
#          and stores them in 'self.f' and 'self.df', respectively. The definition of the line is stored in 'self.lines'.
#         """
#         conditions = {'q': q, 'config_l': config_l, 'multiplet_l': multiplet_l, 'L_l': l_l, 'J_l': j_l,
#                       'config_u': config_u, 'multiplet_u': multiplet_u, 'L_u': l_u, 'J_u': j_u}
#         self.lines[label] = conditions
#
#         transitions = dat.get_table_entries('Lines', conditions)
#         shifts = dat.get_table_entries('IsotopeShifts', conditions)
#
#         for tr in transitions:
#             tools.add_nested_key(self.f_ref, [tr['A'], label, tr['reference']], [tr['f'], tr['f_d']])
#         # self.choose_f()
#
#         if shifts:
#             for shift in shifts:
#                 tools.add_nested_key(self.df_ref, ['{0}-{1}'.format(shift['A'], shift['ARef']), label,
#                                                    shift['reference']], [shift['df'], shift['df_d']])
#             # self.choose_df(min([shift['ARef'] for shift in shifts]))
#
#     def choose_f(self, references: Union[array_iter, dict, str] = None,
#                  exclude: Union[array_iter, dict, str] = None) -> None:
#         """
#         :param references: A reference str, iterable of reference str's,
#          or a dictionary of these with isotopes or lines as keys.
#          The most accurate frequency values will be determined from the specified references.
#         :param exclude: A reference str, iterable of reference str's,
#          or a dictionary of these with isotopes or lines as keys.
#          The specified references will be excluded when choosing frequency values.
#         :returns: None. Fills the dictionary self.f and self.refs_f
#          with the chosen transition frequencies and references.
#         """
#         f, refs_f = {}, {}
#         if exclude is None:
#             exclude = []
#         refs, exc = self.check_ref_args(references), self.check_ref_args(exclude)
#         if refs is None:
#             refs = {a: {line: [ref for ref in self.f_ref[a][line].keys()]
#                         for line in self.lines.keys()}
#                     for a in self.isotopes.keys() if a in self.f_ref.keys()}
#         for a, line_dict in refs.items():
#             f[a], refs_f[a] = {}, {}
#             for line, ref_list in line_dict.items():
#                 uncertainties = [[ref, self.f_ref[a][line][ref][1]] for ref in ref_list
#                                  if a in self.f_ref.keys() and line in self.f_ref[a].keys()
#                                  and ref in self.f_ref[a][line].keys() and ref not in exc[a][line]]
#                 if not uncertainties:
#                     continue
#                 i_min = int(np.argmin([unc[1] for unc in uncertainties]))
#                 ref_min = uncertainties[i_min][0]
#                 f[a][line] = self.f_ref[a][line][ref_min]
#                 refs_f[a][line] = ref_min
#         for a in refs.keys():
#             if f[a] == {}:
#                 del f[a]
#                 del refs_f[a]
#         self.f, self.refs_f = f, refs_f
#
#     def choose_df(self, a_ref: int, references: Union[array_iter, dict, str] = None,
#                   exclude: Union[array_iter, dict, str] = None) -> None:
#         """
#         :param a_ref: The mass number of the reference isotope.
#         :param references: A reference str, iterable of reference str's,
#          or a dictionary of these with isotopes or lines as keys.
#          The most accurate frequency values will be determined from the specified references.
#         :param exclude: A reference str, iterable of reference str's,
#          or a dictionary of these with isotopes or lines as keys.
#          The specified references will be excluded when choosing frequency values.
#         :returns: None. Fills the dictionary self.df and self.refs_df with the chosen isotope shifts and references.
#         """
#         self.a_ref = a_ref
#         df, refs_df = {}, {}
#         if exclude is None:
#             exclude = []
#         refs, exc = self.check_ref_args(references), self.check_ref_args(exclude)
#         if refs is None:
#             refs = {a: {line: [ref for ref in self.df_ref['{0}-{1}'.format(a, a_ref)][line].keys()]
#                         for line in self.lines.keys()}
#                     for a in self.isotopes.keys() if '{0}-{1}'.format(a, a_ref) in self.df_ref.keys()}
#         for a, line_dict in refs.items():
#             if a == a_ref:
#                 continue
#             df[a], refs_df[a] = {}, {}
#             for line, ref_list in line_dict.items():
#                 uncertainties = [[ref, self.df_ref['{0}-{1}'.format(a, a_ref)][line][ref][1]]
#                                  for ref in ref_list if '{0}-{1}'.format(a, a_ref) in self.df_ref.keys()
#                                  and line in self.df_ref['{0}-{1}'.format(a, a_ref)].keys()
#                                  and ref in self.df_ref['{0}-{1}'.format(a, a_ref)][line].keys()
#                                  and ref not in exc[a][line]]
#                 if not uncertainties:
#                     continue
#                 i_min = int(np.argmin([unc[1] for unc in uncertainties]))
#                 ref_min = uncertainties[i_min][0]
#                 df[a][line] = self.df_ref['{0}-{1}'.format(a, a_ref)][line][ref_min]
#                 refs_df[a][line] = ref_min
#         for a in refs.keys():
#             if a == a_ref:
#                 continue
#             if df[a] == {}:
#                 del df[a]
#                 del refs_df[a]
#         self.df, self.refs_df = df, refs_df
#
#     def get_gamma(self, line: str) -> tuple:
#         """
#         :param line: The label of the user-defined electronic transition.
#         :returns: The natural linewidth of the electronic transition and its uncertainty (MHz).
#         """
#         try:
#             conditions = self.lines[line]
#         except KeyError:
#             print('WARNING: The specified line was not defined.')
#             return np.nan, np.nan
#         entries = dat.get_table_entries('Lines', conditions)
#         gamma = np.array([[entry['Gamma'], entry['Gamma_d']] for entry in entries])
#         gamma = gamma[np.argmin(gamma[:, 1]), :]
#         return tuple(gamma)
#
#     def get_hyperfine_constants(self, a: int, line: str) -> tuple:
#         """
#         :param a: The mass number of the isotope.
#         :param line: The label of the user-defined electronic transition.
#         :returns: The natural linewidth of the electronic transition and its uncertainty (MHz).
#         """
#         try:
#             conditions = self.lines[line]
#         except KeyError:
#             print('WARNING: The specified line was not defined.')
#             return [np.nan, ] * 4, [np.nan, ] * 4
#         tools.merge_dicts(conditions, {'A': a})
#         entries = dat.get_table_entries('Lines', conditions)
#         if not entries:
#             print('WARNING: There are no entries for the specified mass number ({}) and line ({}).'.format(a, line))
#             return [np.nan, ] * 4, [np.nan, ] * 4
#         ret1, ret2 = [], []
#         for const in ['A_l', 'B_l', 'A_u', 'B_u']:
#             val = np.array([[entry[const], entry[const + '_d']] for entry in entries])
#             val = val[np.argmin(val[:, 1]), :]
#             ret1.append(np.nan if val[0] is None else val[0])
#             ret2.append(np.nan if val[1] is None else val[1])
#         return ret1, ret2
#
#     def spectroscopy_info(self, e_kin: float) -> dict:
#         """
#         :param e_kin: Kinetic energy of the isotopes (eV).
#         :returns: A dictionary of information for collinear laser spectroscopy with keys 'E', 'U', 'v', 'col' and 'acol'
#          for the kinetic energy (eV), the required voltage (V), the velocity of isotopes
#          with an initial velocity of 0 m/s (m/s), as well as the resonant laser frequencies
#          in collinear and anticollinear geometry (MHz).
#         """
#         info = {}
#         for a, line_dict in self.f.items():
#             info[a] = {}
#             for line, f in line_dict.items():
#                 v = ph.v_e(e_kin, self.isotopes[a]['mass'], 0.)
#                 col = np.around(ph.doppler(f[0], v, 0., return_frame='lab'), decimals=self.decimals)
#                 acol = np.around(ph.doppler(f[0], v, np.pi, return_frame='lab'), decimals=self.decimals)
#                 info[a][line] = \
#                     {'E': e_kin, 'U': np.nan if self.lines[line]['q'] == 0 else e_kin / self.lines[line]['q'],
#                      'v': v, 'col': col, 'acol': acol}
#         return info
#
#     def get_f(self, a: Union[int, Iterable[int]], line: Union[str, Iterable[str]],
#               rtype: type = ndarray) -> Union[ndarray, dict, ValueError]:
#         """
#         :param a: The mass number or an Iterable of mass numbers.
#         :param line: A line or an Iterable of previously defined lines.
#         :param rtype: The type in which the results will be returned. Currently supported special types are [dict, ].
#          The standard 'rtype' is a numpy array.
#         :returns: A subset of the dictionary 'self.f' as a dictionary or a numpy array.
#         """
#         return _get_rtype2(self.f, a, line, rtype=rtype)
#
#     def get_df(self, a: Union[int, Iterable[int]], line: Union[str, Iterable[str]], rtype: type = ndarray) \
#             -> Union[ndarray, dict, ValueError]:
#         """
#         :param a: The mass number or an Iterable of mass numbers.
#         :param line: A line or an Iterable of previously defined lines.
#         :param rtype: The type in which the results will be returned. Currently supported special types are [dict, ].
#          The standard 'rtype' is a numpy array.
#         :returns: A subset of the dictionary 'self.df' as a dictionary or a numpy array.
#         """
#         return _get_rtype2(self.df, a, line, rtype=rtype)
#
#     def calc_df(self, a: Iterable[int], a_ref: Iterable[int], line: Union[str, Iterable[str]],
#                 rtype: type = ndarray) -> Union[ndarray, dict, ValueError]:
#         """
#         :param a: An Iterable of the mass numbers of the isotopes.
#         :param a_ref: An Iterable of the mass numbers of the reference isotopes
#         :param line: A line or an Iterable of previously defined lines.
#         :param rtype: The type in which the results will be returned. Currently supported special types are [dict, ].
#          The standard 'rtype' is a numpy array.
#         :returns: Differences of transition frequencies and their uncertainties calculated from the 'self.f' dictionary
#          according to the lists 'a' and 'a_ref'. The output has shape (len(a), 2)
#          or shape (len(a), len(line), 2) if line is a list.
#         """
#         if isinstance(line, str):
#             a_dict = {k: v[line][0] for k, v in self.f.items()}
#             a_d_dict = {k:  v[line][1] for k, v in self.f.items()}
#             v_dict = tools.convolve_dict(a_dict, a, a_ref, operator='-')
#             v_d_dict = tools.convolve_dict(a_d_dict, a, a_ref, operator='gauss')
#             if rtype == dict:
#                 return {a_i: [round(v_dict[a_i], self.decimals), round(v_d_dict[a_i], self.decimals)] for a_i in a}
#             else:
#                 return np.array([[round(v_dict[a_i], self.decimals), round(v_d_dict[a_i], self.decimals)] for a_i in a])
#         else:
#             a_dict = {k: np.array([v[ll][0] for ll in line]) for k, v in self.f.items()}
#             a_d_dict = {k: np.array([v[ll][1] for ll in line]) for k, v in self.f.items()}
#             v_dict = tools.convolve_dict(a_dict, a, a_ref, operator='-')
#             v_d_dict = tools.convolve_dict(a_d_dict, a, a_ref, operator='gauss')
#             if rtype == dict:
#                 ValueError('rtype=dict not implemented for list of lines yet.')
#                 pass
#             else:
#                 return np.array([np.array([np.around(v_dict[a_i], self.decimals),
#                                            np.around(v_d_dict[a_i], self.decimals)]).T for a_i in a])
#
#     def _calc_f(self, a: Iterable[int], a_ref: Iterable[int], line: str,
#                 rtype: type = ndarray) -> Union[ndarray, dict, ValueError]:
#         # TODO?: Calculation of lines.
#         pass
#
#     def get_masses(self, a: Union[int, Iterable[int]] = None, rtype: type = ndarray) \
#             -> Union[ndarray, dict, ValueError]:
#         """
#         :param a: An Iterable of the mass numbers of the isotopes to be returned.
#         :param rtype: The type in which the results will be returned. Currently supported special types are [dict, ].
#          The standard 'rtype' is a numpy array.
#         :returns: The masses and their uncertainties of the specified mass numbers.
#          The output has shape (2, ) if a is an int value and shape (len(a), 2) if a is an Iterable.
#         """
#         if a is None:
#             a = list(self.isotopes.keys())
#         a_arr = np.asarray(a)
#         a_dim = len(a_arr.shape)
#         if a_dim == 0:
#             if rtype == dict:
#                 return {a: [self.isotopes[a]['mass'], self.isotopes[a]['mass_d']]}
#             else:
#                 return np.array([self.isotopes[a]['mass'], self.isotopes[a]['mass_d']])
#         else:
#             if rtype == dict:
#                 return {a_i: [self.isotopes[a_i]['mass'], self.isotopes[a_i]['mass_d']] for a_i in a}
#             else:
#                 return np.array([[self.isotopes[a_i]['mass'], self.isotopes[a_i]['mass_d']] for a_i in a])
#
#     def get_radii(self):
#         """
#         :returns: The default 'Radii' object for this element.
#         """
#         return Radii(self.label)
#
#     def get_king(self):
#         """
#         :returns: A 'King' object including all isotopes available in 'AtomicDatabase.db'.
#          The number of subtracted electrons in the 'King' object is equal to the charge number Z of the element.
#         """
#         a = list(self.isotopes.keys())
#         return King(a, self.get_masses(a), subtract_electrons=self.isotopes[a[0]]['Z'], element_label=self.label)
#
#
# class Radii:
#     """
#     A class which holds the information about nuclear charge radii from muonic atom spectroscopy
#      and elastic electron scattering. Differences of the even moments of nuclear charge radii
#      as well as the Lambda parameter can be returned in a dict or an array format
#      in the specified combination of mass numbers and reference mass numbers.
#     """
#     def __init__(self, element: str, a: array_iter = None, barrett: array_iter = None,
#                  vn: array_like = None, seltzer: array_iter = None, decimals: int = 4):
#         """
#         :param a: An iterable of mass numbers.
#         :param barrett: An iterable of Barrett radii. Must have shape (len(a), 2) and include the values in the first
#          and the uncertainties in the second column.
#         :param vn: An iterable of Vn factors. Must have either shape (3, ) or (1, 3)
#          in which case the same Vn factors are used for all specified mass numbers or shape (len(a), 3)
#          in which case individual Vn factors are used for every mass number.
#         :param seltzer: An iterable of the two Seltzer coefficients [C2/C1, C3/C1].
#         :param decimals: The number of decimal places (sub-square-fm places) to use for radial values.
#         Any of the parameters which is None is exported from the database 'NuclearChargeRadii.db'.
#         """
#         self.element = element
#         self.conditions = {'element': self.element}
#         self.decimals = decimals
#
#         self.a = None
#         if a is not None:
#             self.a: Union[object, list] = np.asarray(a, dtype=int).tolist()
#
#         if barrett is None:
#             barrett_ref = dat.get_table_entries('BarrettRadii', self.conditions)
#             self.a = list({entry['A'] for entry in barrett_ref})
#             references = {entry['reference'] for entry in barrett_ref}
#             self.barrett_ref = {ref: {entry['A']: [entry['val'], entry['StatErr'] + entry['SysErr']]
#                                       for entry in barrett_ref if entry['reference'] == ref
#                                       and (a is None or entry['A'] in a)} for ref in references}
#             self.ref_barrett = None
#             self.choose_barrett()
#         else:
#             if a is None:
#                 raise ValueError('\'a\' is not specified and therefore \'barrett\' cannot be '
#                                  'related to mass numbers unambiguously.')
#             self.barrett = np.asarray(barrett)
#
#         if vn is None:
#             self.vn = {entry['A']: entry for entry in dat.get_table_entries('Vn', self.conditions)}
#             self.vn = np.array([[self.vn[a_i]['V2'], self.vn[a_i]['V4'], self.vn[a_i]['V6']] if a_i in self.vn.keys()
#                                 else [self.vn[-1]['V2'], self.vn[-1]['V4'], self.vn[-1]['V6']]
#                                 for a_i in self.a])
#         else:
#             self.vn = np.asarray(vn)
#             if a is None and self.vn.shape not in [(3, ), (1, 3)]:
#                 raise ValueError('\'a\' is not specified and therefore \'vn\' cannot be '
#                                  'related to mass numbers unambiguously.')
#         if self.vn.shape == (3, ):
#             self.vn = np.expand_dims(self.vn, axis=0)
#         if self.vn.shape == (1, 3):
#             self.vn = np.array([self.vn[0], ] * len(self.a))
#
#         self.seltzer = seltzer
#         if self.seltzer is None:
#             self.seltzer = dat.get_table_entries('SeltzerCoefficients', self.conditions)
#             self.seltzer = [self.seltzer[0]['C2C1'], self.seltzer[0]['C3C1']]
#
#     def choose_barrett(self, references: Iterable[str] = None, mode: str = 'mean'):
#         """
#         :param references: An Iterable of references to consider.
#         :param mode: The mode how to choose the Barret radii from multiple references.
#          The Currently supported modes are {'minErr', mean}.
#         :returns: None. Sets the instance variables 'barrett' and 'used_references'.
#         """
#         modes = ['minUnc', 'mean']
#         barrett = []
#         ref_barrett = []
#         if mode == 'minUnc':
#             for a in self.a:
#                 uncertainties = [[ref, r[a][1]]
#                                  for ref, r in self.barrett_ref.items()
#                                  if ref in references and a in self.barrett_ref[ref].keys()]
#                 i_min = int(np.argmin([unc[1] for unc in uncertainties]))
#                 ref_min = uncertainties[i_min][0]
#                 barrett.append(self.barrett_ref[ref_min][a])
#                 ref_barrett.append(ref_min)
#         elif mode == 'mean':
#             for a in self.a:
#                 r = np.array([[r[a][0], r[a][1]] for ref, r in self.barrett_ref.items()
#                               if a in self.barrett_ref[ref].keys()])
#                 cov = np.expand_dims(r[:, 1], axis=1) * np.expand_dims(r[:, 1], axis=0)
#                 barrett.append(list(average(r[:, 0], cov=cov)))
#             ref_barrett = 'mean'
#         else:
#             raise ValueError('Mode \'{}\' is not supported. Choose from {}'.format(mode, modes))
#         self.barrett, self.ref_barrett = np.around(barrett, decimals=self.decimals), ref_barrett
#
#     def get_rn(self, n: int, a: array_like, a_ref: array_like, delta_barrett: array_iter = None,
#                decimals: int = 4, rtype: type = ndarray, key_format: str = None) -> Union[ndarray, dict, ValueError]:
#         """
#         :param n: The moment to return. Currently supported moments are [2, 4, 6].
#         :param a: The mass numbers of the moments to return.
#         :param a_ref: The reference mass numbers of the moments to return.
#         :param delta_barrett: The differences of Barrett radii with uncertainties.
#          If specified, it must have shape (len(a), 2) and include the values in the first
#          and the uncertainties in the second column.
#         :param decimals: The number of decimal places to return.
#         :param rtype: The type in which the results will be returned. Currently supported special types are [dict, ].
#          The standard 'rtype' is a numpy array.
#         :param key_format: If 'rtype' is dict, The key format is used to build the dictionary keys in str format.
#          The mass number is inserted into the first, the reference mass number into the second '{}' bracket.
#          Use numbers {1}(...){0} to change the order of the mass numbers.
#          If None, the mass numbers are used as integer keys.
#         :returns: Differences of the specified moment of nuclear charge radii for different mass
#          and reference mass numbers.
#         """
#         n = int(n)
#         if n not in [2, 4, 6]:
#             raise ValueError('n must be in [2, 4, 6] but is {}'.format(n))
#         a: Union[object, list, int] = np.asarray(a, dtype=int).tolist()
#         if isinstance(a, int):
#             a = [a, ]
#         a_ref: Union[object, list, int] = np.asarray(a_ref, dtype=int).tolist()
#         if isinstance(a_ref, int):
#             a_ref = [a_ref, ] * len(a)
#         i = np.array([self.a.index(a_i) for a_i in a])
#         i_ref = np.array([self.a.index(a_i) for a_i in a_ref])
#         if delta_barrett is None:
#             delta_barrett = np.array([self.barrett[i, 0] - self.barrett[i_ref, 0],
#                                       np.sqrt(self.barrett[i, 1] ** 2 + self.barrett[i_ref, 1] ** 2)]).T
#         delta_barrett = np.asarray(delta_barrett)
#         rn = eval('ph.delta_r{}'.format(n))(self.barrett[i, 0], self.barrett[i, 1],
#                                             self.barrett[i_ref, 0], self.barrett[i_ref, 1],
#                                             delta_barrett[:, 0], delta_barrett[:, 1],
#                                             self.vn[i, int(n / 2 - 1)], self.vn[i_ref, int(n / 2 - 1)])
#         rn = [[np.around(rn[0][j], decimals=decimals), np.around(rn[1][j], decimals=decimals)]
#               for j in range(rn[0].size)]
#         if rtype == dict:
#             if key_format is None:
#                 return {a_i: r for a_i, r in zip(a, rn)}
#             elif key_format.count('{}') == 1 and key_format.count('{') == 1 and key_format.count('}') == 1:
#                 return {key_format.format(a_i): r for a_i, r in zip(a, rn)}
#             elif key_format.count('{}') == 2 or ('{0}' in key_format and '{1}' in key_format) \
#                     and key_format.count('{') == 2 and key_format.count('}') == 2:
#                 return {key_format.format(a_i, a_ref_i): r for a_i, a_ref_i, r in zip(a, a_ref, rn)}
#             raise ValueError('Specified key_format ({}) not supported.'.format(key_format))
#         else:
#             return np.array(rn)
#
#     def get_lambda(self, a: array_like, a_ref: array_like, delta_barrett: array_iter = None,
#                    decimals: int = 4, rtype: type = ndarray, key_format: str = None) \
#             -> Union[ndarray, dict, ValueError]:
#         """
#         :param a: The mass numbers of the moments to return.
#         :param a_ref: The reference mass numbers of the moments to return.
#         :param delta_barrett: The differences of Barrett radii with uncertainties.
#          If specified, it must have shape (len(a), 2) and include the values in the first
#          and the uncertainties in the second column.
#         :param decimals: The number of decimal places to return.
#         :param rtype: The type in which the results will be returned. Currently supported special types are [dict, ].
#          The standard 'rtype' is a numpy array.
#         :param key_format: If 'rtype' is dict, The key format is used to build the dictionary keys in str format.
#          The mass number is inserted into the first, the reference mass number into the second '{}' bracket.
#          Use numbers {1}(...){0} to change the order of the mass numbers.
#          If None, the mass numbers are used as integer keys.
#         :returns: The Lambda parameter for different mass and reference mass numbers.
#         """
#         if self.seltzer is None:
#             return ValueError('Seltzer coefficients were not specified. '
#                               'Please assign the local variable\nself.seltzer = [C2/C1, C3/C1]')
#         r2 = self.get_rn(2, a, a_ref, delta_barrett=delta_barrett, decimals=decimals + 2, rtype=list)
#         r4 = self.get_rn(4, a, a_ref, delta_barrett=delta_barrett, decimals=decimals + 2, rtype=list)
#         r6 = self.get_rn(6, a, a_ref, delta_barrett=delta_barrett, decimals=decimals + 2, rtype=list)
#         ll = ph.lambda_rn(r2[:, 0], r2[:, 1], r4[:, 0], r4[:, 1], r6[:, 0], r6[:, 1], self.seltzer[0], self.seltzer[1])
#         ll = [[np.around(ll[0][j], decimals=decimals), np.around(ll[1][j], decimals=decimals)]
#               for j in range(ll[0].size)]
#         if rtype == dict:
#             print(key_format.count('{}'))
#             if key_format is None:
#                 return {a_i: r for a_i, r in zip(a, ll)}
#             elif key_format.count('{}') == 1 and key_format.count('{') == 1 and key_format.count('}') == 1:
#                 return {key_format.format(a_i): r for a_i, r in zip(a, ll)}
#             elif key_format.count('{}') == 2 or ('{0}' in key_format and '{1}' in key_format) \
#                     and key_format.count('{') == 2 and key_format.count('}') == 2:
#                 return {key_format.format(a_i, a_ref_i): r for a_i, a_ref_i, r in zip(a, a_ref, ll)}
#             raise ValueError('Specified key_format ({}) not supported.'.format(key_format))
#         else:
#             return np.array(ll)


class King:
    def __init__(self, a: array_iter, m: array_iter, x_abs: array_iter = None,
                 subtract_electrons: scalar = 0., element_label: str = None):
        """
        :param a: An Iterable of the mass numbers of the used isotopes.
        :param m: The masses and their uncertainties of the isotopes 'a' (amu). 'm' must have shape (len(a), 2).
        :param x_abs: Absolute values of the x-axis corresponding to the mass numbers a.
         If given, the 'x' parameter can be omitted when fitting (in 'fit' and 'fit_nd')
         and is determined automatically as the difference between the mass numbers 'a' and 'a_ref'.
         Must have shape (len(a), 2) or (len(a), n, 2) where n is the number of dimensions of the king plot without 'y'.
        :param subtract_electrons: The number of electron masses
         that should be subtracted from the specified isotope masses. If the ionization energy must be considered,
         less electrons can be subtracted. 'subtract_electrons' does not have to be an integer.
        :param element_label: The label of the element to enhance the printed and plotted information.
        """
        self.fontsize = 12.
        self.scale_y = 1e-3
        self.n = 1000000
        self.subtract_electrons = subtract_electrons
        self.element_label = element_label
        if self.element_label is None:
            self.element_label = ''
        self.a: Union[object, list] = np.asarray(a, dtype=int).tolist()
        self.x_abs = x_abs
        if self.x_abs is not None:
            self.x_abs = np.asarray(x_abs)
            if len(self.x_abs.shape) < 3:
                self.x_abs = np.expand_dims(self.x_abs, axis=1)
        self.m: Union[object, list, ndarray] = np.asarray(m, dtype=float).tolist()
        self.m = np.array([[m_i[0] - self.subtract_electrons * ph.m_e_u,
                            np.sqrt(m_i[1] ** 2 + (self.subtract_electrons * ph.m_e_u_d) ** 2)] for m_i in self.m])
        tools.check_shape((len(a), 2), self.m, allow_scalar=False)
        self.m_mod = np.array([[list(ph.mass_factor(_m[0], _m_ref[0], m_d=_m[1], m_ref_d=_m_ref[1], k_inf=True))
                                if _i != _j else [0., 0.]
                                for _j, _m in enumerate(self.m)] for _i, _m_ref in enumerate(self.m)])
        self.m_sample = None

        self.a_fit, self.a_ref = None, None
        self.x, self.y = None, None
        self.x_mod, self.y_mod = None, None
        self.corr = None
        self.results: Union[None, tuple] = None

        self.x_nd = None
        self.x_nd_mod = None
        self.cov = None
        self.results_nd: Union[None, tuple] = None
        self.x_results: Union[None, tuple] = None
        self.axis = 0

    def _correlation(self):
        """
        :returns: None. Sets the correlation coefficients of the modified x- and y-data points by sampling the data.
        """
        i = np.array([self.a.index(a_i) for a_i in self.a_fit])
        i_ref = np.array([self.a.index(a_i) for a_i in self.a_ref])
        self.m_sample = st.norm.rvs(loc=self.m[:, 0], scale=self.m[:, 1], size=(self.n, self.m.shape[0])).T
        x = st.norm.rvs(loc=self.x[:, 0], scale=self.x[:, 1], size=(self.n, self.x.shape[0])).T
        y = st.norm.rvs(loc=self.y[:, 0], scale=self.y[:, 1], size=(self.n, self.y.shape[0])).T
        m_mod_sample = ph.mass_factor(self.m_sample[i], self.m_sample[i_ref], k_inf=True)[0]
        x_mod = m_mod_sample * x
        y_mod = m_mod_sample * y
        self.corr = np.diag(np.corrcoef(x_mod, y_mod)[:x_mod.shape[0], x_mod.shape[0]:])

    def _correlation_nd(self):
        """
        :returns: None. Sets the correlation coefficients of the modified data by sampling it.
        """
        i = np.array([self.a.index(a_i) for a_i in self.a_fit])
        i_ref = np.array([self.a.index(a_i) for a_i in self.a_ref])
        self.m_sample = st.norm.rvs(loc=self.m[:, 0], scale=self.m[:, 1], size=(self.n, self.m.shape[0])).T
        x_nd = st.norm.rvs(loc=self.x_nd[:, :, 0], scale=self.x_nd[:, :, 1],
                           size=(self.n, self.x_nd.shape[0], self.x_nd.shape[1]))
        x_nd = np.transpose(x_nd, axes=[1, 2, 0])
        m_mod_sample = ph.mass_factor(self.m_sample[i], self.m_sample[i_ref], k_inf=True)[0]
        x_mod = np.expand_dims(m_mod_sample, axis=1) * x_nd
        self.cov = np.array([np.cov(x_mod_i) for x_mod_i in x_mod])

    def fit(self, a: array_iter, a_ref: array_iter, x: array_iter = None, y: array_iter = None,
            xy: Iterable[int] = None, func: Callable = york, alpha: scalar = 0, find_alpha: bool = False,
            show: bool = True, **kwargs):
        """
        :param a: An Iterable of the mass numbers of the used isotopes.
        :param a_ref: An Iterable of the mass numbers of the used reference isotopes.
        :param x: The x-values and their uncertainties of the King-fit to be performed. If 'mode' is "radii",
         this must contain the differences of mean square nuclear charge radii or the Lambda-factor.
         'x' must have shape (len(a), 2). Units: (fm ** 2) or (MHz).
        :param y: The isotope shifts and their uncertainties of the King-fit to be performed.
         'y' must have shape (len(a), 2). If None, 'y' is tried to be inherited from 'self.f'. Units: (MHz).
        :param xy: A 2-tuple of indices (ix, iy) which select the axes to use for the King-fit from 'King.x_abs'
         if x or y is not specified. The default value is (0, 1), fitting the second against the first axis.
        :param func: The fitting routine.
        :param alpha: An x-axis offset to reduce the correlation coefficient between the y-intercept and the slope.
         Unit: (u fm ** 2) or (u MHz).
        :param find_alpha: Whether to search for the best 'alpha'. Uses the given 'alpha' as a starting point.
         May not give the desired result if 'alpha' was initialized to far from its optimal value.
        :param show: Whether to plot the fit result.
        :param kwargs: Additional keyword arguments are passed to 'self.plot'.
        :returns: A list of results as defined by the routine 'linear_alpha':
         a, b, sigma_a, sigma_b, corr_ab, alpha. The best y-intercept and slope,
         their respective 1-sigma uncertainties, their correlation coefficient and the used alpha.
        """
        self.a_fit, self.a_ref = np.asarray(a), np.asarray(a_ref)
        i = np.array([self.a.index(a_i) for a_i in self.a_fit])
        i_ref = np.array([self.a.index(a_i) for a_i in self.a_ref])
        if xy is None:
            xy = (0, 1)
        if x is None:
            self.x = np.zeros((self.a_fit.size, 2), dtype=float)
            self.x[:, 0] = self.x_abs[i, xy[0], 0] - self.x_abs[i_ref, xy[0], 0]
            self.x[:, 1] = np.sqrt(self.x_abs[i, xy[0], 1] ** 2 + self.x_abs[i_ref, xy[0], 1] ** 2)
        else:
            self.x = np.asarray(x)
        if y is None:
            self.y = np.zeros((self.a_fit.size, 2), dtype=float)
            self.y[:, 0] = self.x_abs[i, xy[1], 0] - self.x_abs[i_ref, xy[1], 0]
            self.y[:, 1] = np.sqrt(self.x_abs[i, xy[1], 1] ** 2 + self.x_abs[i_ref, xy[1], 1] ** 2)
        else:
            self.y = np.asarray(y)
        m_mod = self.m_mod[i_ref, i, :]
        self.x_mod = np.array([self.x[:, 0] * m_mod[:, 0],
                               straight_x_std(self.x[:, 0], m_mod[:, 0], self.x[:, 1], 0., m_mod[:, 1], 0.)]).T
        self.y_mod = np.array([self.y[:, 0] * m_mod[:, 0],
                               straight_x_std(self.y[:, 0], m_mod[:, 0], self.y[:, 1], 0., m_mod[:, 1], 0.)]).T
        self._correlation()
        self.results = linear_alpha(self.x_mod[:, 0], self.y_mod[:, 0],
                                    sigma_x=self.x_mod[:, 1], sigma_y=self.y_mod[:, 1],
                                    corr=self.corr, func=func, alpha=alpha, find_alpha=find_alpha, show=False)
        if show:
            self.plot(**kwargs)
        return self.results

    def fit_nd(self, a: array_iter, a_ref: array_iter, x: array_iter = None,
               axis: int = 0, show: bool = True, **kwargs):
        """
        :param a: An Iterable of the mass numbers of the used isotopes.
        :param a_ref: An Iterable of the mass numbers of the used reference isotopes.
        :param x: The x data. 'x' is an iterable of vectors with uncertainties with arbitrary but fixed dimension n_vec.
         For a set of n_data data points, the shape of x must be (n_data, n_vec, 2).
        :param axis: The axis to use for the parametrization. For example, a King plot with the isotope shifts
         of two transitions ['D1', 'D2'], yields the slope F_D2 / F_D1 if 'axis' == 0.
        :param show: Whether to plot the fit result.
        :param kwargs: Additional keyword arguments are passed to 'self.plot_nd'.
        :returns: A list of results as defined by the routine 'linear_monte_carlo_nd':
         A tuple (a, b, sigma_a, sigma_b, corr_ab) of arrays. The best y-intercepts and slopes,
         their respective 1-sigma uncertainties and their correlation matrix.
         Additionally, a second tuple with the accepted points, offsets, slopes and a mask
         for the accepted points is returned if full_output == True.
        """
        self.axis = axis
        if axis is None:
            self.axis = 0
        if axis == -1:
            self.axis = x.shape[1] - 1
        self.a_fit, self.a_ref = np.asarray(a), np.asarray(a_ref)
        i = np.array([self.a.index(a_i) for a_i in self.a_fit])
        i_ref = np.array([self.a.index(a_i) for a_i in self.a_ref])
        if x is None:
            self.x_nd = np.zeros((self.a_fit.size, self.x_abs.shape[1], 2), dtype=float)
            self.x_nd[:, :, 0] = self.x_abs[i, :, 0] - self.x_abs[i_ref, :, 0]
            self.x_nd[:, :, 1] = np.sqrt(self.x_abs[i, :, 1] ** 2 + self.x_abs[i_ref, :, 1] ** 2)
        else:
            self.x_nd = np.asarray(x)
        m_mod = np.expand_dims(self.m_mod[i_ref, i, 0], axis=1)
        self.x_nd_mod = m_mod * self.x_nd[:, :, 0]
        self._correlation_nd()
        self.results_nd, self.x_results = linear_monte_carlo_nd(self.x_nd_mod, cov=self.cov,
                                                                axis=axis, full_output=True, show=False)
        # plt.plot(self.x_results[0][self.x_results[1], :, 0], self.x_results[1][self.x_results[1], :, 1], '.')
        # plt.show()
        if show:
            self.plot_nd(axis=self.axis, **kwargs)
        return self.results_nd, self.x_results

    def get_popt(self) -> (ndarray, ndarray):
        """
        :returns: The y-intercept and slope of the performed fit and their correlation,
         if alpha was zero (Only changes the y-intercept).
         This returns the mass- and field-shift factors K and F if the x-axis are radii
         and the y-intercept Ky - Fy/Fx * Kx and field-shift ratio Fy/Fx if the x-axis are isotope shifts.
        """
        if self.results is None:
            print('There are no results yet. Please use one of the fitting options.')
            return None, None
        (a, b, sigma_a, sigma_b, corr, alpha) = self.results
        f = [b, sigma_b]
        k = [a - b * alpha, np.sqrt(sigma_a ** 2 + (alpha * sigma_b) ** 2)]
        tools.printh('\npopt:')
        print('f = {} +/- {}\nk = {} +/- {}\ncorr = {}'.format(*f, *k, corr))
        return np.array(k), np.array(f), corr

    def get_popt_nd(self) -> (ndarray, ndarray):
        """
        :returns: The slopes and y-intercepts of the performed fit.
         This returns the mass- and field-shift factors K and F if the x-axis are radii
         and the y-intercept Ky - Fy/Fx * Kx and field-shift ratio Fy/Fx if the x-axis are isotope shifts.
         The returned arrays k and f have shape (n, 2) were n is number of dimensions of the King-fit.
        """
        if self.results_nd is None:
            print('There are no results yet. Please use one of the fitting options.')
            return None, None
        (a, b, sigma_a, sigma_b, corr) = self.results_nd
        f = [b, sigma_b]
        k = [a, sigma_a]
        r = np.diag(corr[a.size:, :a.size])
        tools.printh('\npopt_nd:')
        for i in range(b.size):
            print('{}{}:{}'.format(tools.COLORS.OKCYAN, i, tools.COLORS.ENDC))
            print('f = {}\t{}\nk = {}\t{}\ncorr = {}'.format(f[0][i], f[1][i], k[0][i], k[1][i], r[i]))
        return np.array(k).T, np.array(f).T, r

    def get_unmodified_x(self, a: array_iter, a_ref: array_iter, y: array_iter, results: tuple = None,
                         show: bool = False, **kwargs) -> Union[ndarray, None]:
        """
        :param a: An Iterable of mass numbers corresponding to the isotopes of the given 'y' values.
        :param a_ref: An Iterable of mass numbers corresponding to reference isotopes of the given 'y' values.
        :param y: The unmodified y values of the given mass numbers. Must have shape (len(a), 2).
        :param results: The results of a King plot. Must be a tuple
         of the shape (a, b, sigma_a, sigma_b, corr_ab, alpha), compare the module-level method 'linear_alpha'.
         If None, the results are derived from 'self.results'.
        :param show: Whether to draw the King plot with the specified values.
        :param kwargs: Additional keyword arguments are passed to 'self.plot'.
        :returns: The unmodified x values calculated with the fit results and the given 'y' values.
        """
        if results is None and self.results is None:
            print('There are no results yet. Please use one of the fitting options.')
            return
        if results is not None:
            (_a, b, sigma_a, sigma_b, corr, alpha) = results
        else:
            (_a, b, sigma_a, sigma_b, corr, alpha) = self.results
        a, a_ref, y = np.asarray(a), np.asarray(a_ref), np.asarray(y)
        i = np.array([self.a.index(a_i) for a_i in a])
        i_ref = np.array([self.a.index(a_i) for a_i in a_ref])
        m_mod = self.m_mod[i_ref, i, :]
        y_mod = y[:, 0] * m_mod[:, 0]
        y_mod_d = y[:, 1] * m_mod[:, 0]
        x_mod = straight(y_mod, -_a / b, 1. / b) + alpha
        x_mod_d = np.sqrt((y_mod_d / b) ** 2 + (sigma_a / b) ** 2 + (sigma_b * (x_mod - alpha) / b) ** 2
                          + 2. * (x_mod - alpha) / (b ** 2) * sigma_a * sigma_b * corr)
        if show:
            add_xy = np.array([x_mod, x_mod_d, y_mod, y_mod_d]).T
            add_a = np.array([a, a_ref]).T
            self.plot(add_xy=add_xy, add_a=add_a, **kwargs)
        return np.array([x_mod / m_mod[:, 0], np.abs(x_mod_d / m_mod[:, 0])]).T

    def get_unmodified_y(self, a: array_iter, a_ref: array_iter, x: array_iter, results: tuple = None) \
            -> Union[ndarray, None]:
        """
        :param a: An Iterable of mass numbers corresponding to the isotopes of the given 'x' values.
        :param a_ref: An Iterable of mass numbers corresponding to reference isotopes of the given 'x' values.
        :param x: The unmodified x values of the given mass numbers. Must have shape (len(a), 2).
        :param results: The results of a King plot. Must be a tuple
         of the shape (a, b, sigma_a, sigma_b, corr_ab, alpha), compare the module-level method 'linear_alpha'.
         If None, the results are derived from 'self.results'.
        :returns: The unmodified y values calculated with the fit results and the given 'x' values.
        """
        if results is None and self.results is None:
            print('There are no results yet. Please use one of the fitting options.')
            return
        if results is not None:
            (_a, b, sigma_a, sigma_b, corr, alpha) = results
        else:
            (_a, b, sigma_a, sigma_b, corr, alpha) = self.results
        a, a_ref, x = np.asarray(a), np.asarray(a_ref), np.asarray(x)
        i = np.array([self.a.index(a_i) for a_i in a])
        i_ref = np.array([self.a.index(a_i) for a_i in a_ref])
        m_mod = self.m_mod[i_ref, i, :]
        x_mod = x[:, 0] * m_mod[:, 0] - alpha
        x_mod_d = x[:, 1] * m_mod[:, 0]
        y_mod = straight(x_mod, _a, b)
        y_mod_d = straight_x_std(x_mod, b, x_mod_d, sigma_a, sigma_b, corr)
        return np.array([y_mod / m_mod[:, 0], y_mod_d / m_mod[:, 0]]).T

    def get_unmodified_input_x_nd(self) -> Union[ndarray, None]:
        """
        :returns: The unmodified x values improved by the 'fit_nd'.
        """
        if self.x_results is None:
            print('There are no results yet. Please use one of the fitting options.')
            return None
        i = np.array([self.a.index(a_i) for a_i in self.a_fit])
        i_ref = np.array([self.a.index(a_i) for a_i in self.a_ref])
        m_mod_sample = ph.mass_factor(self.m_sample[i], self.m_sample[i_ref], k_inf=True)[0]
        x = self.x_results[0] / np.expand_dims(m_mod_sample.T[self.x_results[-1]], axis=-1)
        return np.transpose(np.array([np.mean(x, axis=0), np.std(x, ddof=1, axis=0)]), axes=[1, 2, 0])

    def get_unmodified_x_nd(self, a: array_iter, a_ref: array_iter, y: array_iter, axis: int) -> Union[ndarray, None]:
        """
        :param a: An Iterable of mass numbers corresponding to the isotopes of the given 'y' values.
        :param a_ref: An Iterable of mass numbers corresponding to reference isotopes of the given 'y' values.
        :param y: The unmodified y values of the given mass numbers. Must have shape (len(a), 2).
        :param axis: The axis of the input values.
        :returns: The unmodified values improved by 'fit_nd'.
        """
        if self.x_results is None:
            print('There are no results yet. Please use one of the fitting options.')
            return None
        _a = np.expand_dims(self.x_results[1], axis=1)
        b = np.expand_dims(self.x_results[2], axis=1)
        a, a_ref, y = np.asarray(a), np.asarray(a_ref), np.asarray(y)
        i = np.array([self.a.index(a_i) for a_i in a])
        i_ref = np.array([self.a.index(a_i) for a_i in a_ref])
        m = self.m_sample[i, :]
        m = m[:, self.x_results[-1]]
        m_ref = self.m_sample[i_ref, :]
        m_ref = m_ref[:, self.x_results[-1]]
        m_mod_sample = ph.mass_factor(m, m_ref, k_inf=True)[0].T
        y_sample = st.norm.rvs(loc=y[:, 0], scale=y[:, 1], size=(m_mod_sample.shape[0], len(a)))
        y_sample = m_mod_sample * y_sample
        x_sample = straight(y_sample, -_a[:, :, axis] / b[:, :, axis], 1. / b[:, :, axis])
        x_sample = straight(np.expand_dims(x_sample, axis=-1), _a, b)
        x_sample /= np.expand_dims(m_mod_sample, axis=-1)
        return np.array([np.mean(x_sample, axis=0), np.std(x_sample, axis=0, ddof=1)]).T

    def plot(self, mode: str = '', sigma2d: bool = True, show: bool = True, add_xy: array_like = None,
             add_a: array_like = None):
        """
        :param mode: The mode of the King-fit. If mode='radii', the :math:`x`-axis must contain the differences of
         mean square nuclear charge radii or the Lambda-factor. For every other value,
         the :math:`x`-axis is assumed to be an isotope shift such that the slope corresponds to
         a field-shift ratio :math:`F(y_i) / F(x)`.
        :param sigma2d: Whether to draw the actual two-dimensional uncertainty bounds or the classical errorbars.
        :param show: Whether to show the plot.
        :param add_xy: Additional :math:`x` and :math:`y` data to plot. Must have shape (:, 4).
        :param add_a: Additional mass numbers for the additional data- Must have shape (:, 2).
         The reference is the second column.
        :returns: None. Generates a King-Plot based on the modified axes 'self.x_mod' and 'self.y_mod'
         as well as the fit results 'self.results'.
        """
        if self.results is None:
            print('There are no results yet. Please use one of the fitting options.')
            return
        (a, b, sigma_a, sigma_b, corr, alpha) = self.results
        scale_x = 1e-3
        offset_str = ' - {}'.format(round(alpha * scale_x, 1)) if alpha != 0 else ''
        xlabel = 'x{} (mode=\'shifts\' or =\'radii\' for right x-label)'.format(offset_str)
        if mode == 'radii':
            scale_x = 1.
            offset_str = ' - {}'.format(round(alpha * scale_x, 1)) if alpha != 0 else ''
            xlabel = r'$\mu\,\delta\langle r^2\rangle^{A, A^\prime}' + offset_str + r'\quad(\mathrm{u}\,\mathrm{fm}^2)$'
        elif mode == 'shifts':
            scale_x = 1e-3
            offset_str = ' - {}'.format(round(alpha * scale_x, 1)) if alpha != 0 else ''
            xlabel = r'$\mu\,\delta\nu_x^{A, A^\prime}' + offset_str + r'\quad(\mathrm{u\,GHz})$'

        if sigma2d:
            plt.plot((self.x_mod[:, 0] - alpha) * scale_x,
                     self.y_mod[:, 0] * self.scale_y, 'k.', label='Data')
            draw_sigma2d((self.x_mod[:, 0] - alpha) * scale_x, self.y_mod[:, 0] * self.scale_y,
                         self.x_mod[:, 1] * scale_x, self.y_mod[:, 1] * self.scale_y, self.corr, n=2)
            if add_xy is not None:
                plt.plot((add_xy[:, 0] - alpha) * scale_x,
                         add_xy[:, 2] * self.scale_y, 'r.', label='Data')
                draw_sigma2d((add_xy[:, 0] - alpha) * scale_x, add_xy[:, 2] * self.scale_y,
                             add_xy[:, 1] * scale_x, add_xy[:, 3] * self.scale_y, self.corr, n=2, **{'color': 'r'})
        else:
            plt.errorbar((self.x_mod[:, 0] - alpha) * scale_x, self.y_mod[:, 0] * self.scale_y,
                         xerr=self.x_mod[:, 1] * scale_x, yerr=self.y_mod[:, 1] * self.scale_y,
                         fmt='k.', label='Data')
            if add_xy is not None:
                plt.errorbar((add_xy[:, 0] - alpha) * scale_x, add_xy[:, 2] * self.scale_y,
                             xerr=add_xy[:, 1] * scale_x, yerr=add_xy[:, 3] * self.scale_y,
                             fmt='r.', label='Add. Data')
        for j in range(len(self.a_fit)):
            pm = -1. if b > 0 else 1.
            plt.text((self.x_mod[j, 0] + 1. * self.x_mod[j, 1] - alpha) * scale_x,
                     (self.y_mod[j, 0] + pm * 5. * self.y_mod[j, 1]) * self.scale_y,
                     '{} - {}'.format(self.a_fit[j], self.a_ref[j]), fontsize=self.fontsize,
                     horizontalalignment='left', verticalalignment='top' if b > 0 else 'bottom')
        if add_a is not None and add_xy is not None:
            for j in range(add_xy.shape[0]):
                pm = 1. if b > 0 else -1.
                plt.text((add_xy[j, 0] - 1. * add_xy[j, 1] - alpha) * scale_x,
                         (add_xy[j, 2] + pm * 5. * add_xy[j, 3]) * self.scale_y,
                         '{} - {}'.format(*add_a[j]), fontsize=self.fontsize,
                         horizontalalignment='right', verticalalignment='bottom' if b > 0 else 'top')

        min_x, max_x = np.min(self.x_mod[:, 0]), np.max(self.x_mod[:, 0])
        if add_xy is not None:
            min_x, max_x = np.min([min_x, np.min(add_xy[:, 0])]), np.max([max_x, np.max(add_xy[:, 0])])
        x_cont = np.linspace(min_x - 0.1 * (max_x - min_x), max_x + 0.1 * (max_x - min_x), 1001) - alpha
        plt.plot(x_cont * scale_x, straight(x_cont, a, b) * self.scale_y, 'b-', label='Fit')
        y_min = straight(x_cont, a, b) - straight_std(x_cont, sigma_a, sigma_b, corr)
        y_max = straight(x_cont, a, b) + straight_std(x_cont, sigma_a, sigma_b, corr)
        plt.fill_between(x_cont * scale_x, y_min * self.scale_y, y_max * self.scale_y,
                         color='b', alpha=0.3, antialiased=True)
        
        plt.xlabel(xlabel, fontsize=self.fontsize)
        plt.ylabel(r'$\mu\,\delta\nu_y^{A, A^\prime}\quad(\mathrm{u\,GHz})$', fontsize=self.fontsize)
        plt.xticks(fontsize=self.fontsize)
        plt.yticks(fontsize=self.fontsize)

        plt.legend(numpoints=1, loc='best', fontsize=self.fontsize)
        plt.margins(0.1)
        plt.tight_layout()
        if show:
            plt.show()
            plt.gcf().clear()

    def plot_nd(self, axis: int = 0, mode: str = '', sigma2d: bool = True):
        """
        :param axis: The axis which is used as the x-axis throughout the plots.
        :param mode: The mode of the King-plot. If mode='radii', the :math:`x`-axis must contain the differences of
         mean square nuclear charge radii or the Lambda-factor. For every other value,
         the :math:`x`-axis is assumed to be an isotope shift such that the slope corresponds to
         a field-shift ratio :math:`F(y_i) / F(x)`.
        :param sigma2d: Whether to draw the actual two-dimensional uncertainty bounds or the classical errorbars.
        :returns: None. Generates a King-Plot based on the modified axes 'self.x_mod_nd' and 'self.y_mod_nd'
         as well as the fit results 'self.results_nd'.
        """
        if self.results_nd is None:
            print('There are no results yet. Please use one of the fitting options.')
            return
        (a, b, sigma_a, sigma_b, corr) = self.results_nd
        alpha = 0

        scale_x = 1e-3
        offset_str = ' - {}'.format(round(alpha * scale_x, 1)) if alpha != 0 else ''
        xlabel = 'x{} (mode=\'shifts\' or =\'radii\' for right x-label)'.format(offset_str)
        if mode == 'radii':
            scale_x = 1.
            offset_str = ' - {}'.format(round(alpha * scale_x, 1)) if alpha != 0 else ''
            xlabel = r'$\mu\,\delta\langle r^2\rangle^{A, A^\prime}' + offset_str + r'\quad(\mathrm{u}\,\mathrm{fm}^2)$'
        elif mode == 'shifts':
            scale_x = 1e-3
            offset_str = ' - {}'.format(round(alpha * scale_x, 1)) if alpha != 0 else ''
            xlabel = r'$\mu\,\delta\nu_x^{A, A^\prime}' + offset_str + r'\quad(\mathrm{u\,GHz})$'
        
        n_rows = self.x_nd_mod.shape[1] - 1
        rows = [i for i in range(n_rows + 1) if i != axis]
        for row, i in enumerate(rows):
            plt.subplot(n_rows, 1, row + 1)

            if sigma2d:
                plt.plot((self.x_nd_mod[:, axis] - alpha) * scale_x,
                         self.x_nd_mod[:, i] * self.scale_y, 'k.', label='Data')
                draw_sigma2d((self.x_nd_mod[:, axis] - alpha) * scale_x, self.x_nd_mod[:, i] * self.scale_y,
                             np.sqrt(self.cov[:, axis, axis]) * scale_x, np.sqrt(self.cov[:, i, i]) * self.scale_y,
                             self.cov[:, axis, i] / np.sqrt(self.cov[:, axis, axis] * self.cov[:, i, i]), n=2)
            else:
                plt.errorbar((self.x_nd_mod[:, axis] - alpha) * scale_x, self.x_nd_mod[:, i] * self.scale_y,
                             xerr=np.sqrt(self.cov[:, axis, axis]) * scale_x,
                             yerr=np.sqrt(self.cov[:, i, i]) * self.scale_y, fmt='k.', label='Data')

            for j in range(len(self.a_fit)):
                pm = -1. if b[row] > 0 else 1.
                plt.text((self.x_nd_mod[j, axis] + 1. * np.sqrt(self.cov[j, axis, axis]) - alpha) * scale_x,
                         (self.x_nd_mod[j, i] + pm * 5. * np.sqrt(self.cov[j, i, i])) * self.scale_y,
                         '{} - {}'.format(self.a_fit[j], self.a_ref[j]), fontsize=self.fontsize,
                         horizontalalignment='left', verticalalignment='top' if b[row] > 0 else 'bottom')

            min_x, max_x = np.min(self.x_nd_mod[:, axis]), np.max(self.x_nd_mod[:, axis])
            x_cont = np.linspace(min_x - 0.1 * (max_x - min_x), max_x + 0.1 * (max_x - min_x), 1001) - alpha

            plt.plot(x_cont * scale_x, straight(x_cont, a[row], b[row]) * self.scale_y, 'b-', label='Fit')

            y_min = straight(x_cont, a[row], b[row]) \
                - straight_std(x_cont, sigma_a[row], sigma_b[row], corr[row, row + n_rows])
            y_max = straight(x_cont, a[row], b[row]) \
                + straight_std(x_cont, sigma_a[row], sigma_b[row], corr[row, row + n_rows])
            plt.fill_between(x_cont * scale_x, y_min * self.scale_y, y_max * self.scale_y,
                             color='b', alpha=0.3, antialiased=True)

            plt.xlabel(xlabel, fontsize=self.fontsize)
            plt.ylabel(r'$\mu\,\delta\nu_{y' + str(i) + r'}^{A, A^\prime}\quad(\mathrm{u\,GHz})$',
                       fontsize=self.fontsize)
            plt.xticks(fontsize=self.fontsize)
            plt.yticks(fontsize=self.fontsize)
            plt.legend(numpoints=1, loc='best', fontsize=self.fontsize)
            plt.margins(0.1)
            # plt.tight_layout()

        plt.show()
        plt.gcf().clear()
