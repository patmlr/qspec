# -*- coding: utf-8 -*-
"""
examples.analyze_ex
===================

Example script / Guide for the qspec.analyze module.
"""

import numpy as np
import matplotlib.pyplot as plt

import qspec as qs


def example(n=None):
    """
    Run one or several of the available examples. Scroll to the end for the function call.

    :param n: The number of the example or a list of numbers.

    Example 0: Fit of a straight line with different algorithms.

    Example 1: Fit of a straight in 3d.

    Example 2: King-fit with different algorithms.

    :returns: None.
    """
    if n is None:
        n = {0, 1, }
    if isinstance(n, int):
        n = {n, }

    if 0 in n:
        """
        Example 0: Fit of a straight line with different algorithms.
        """
        # Input
        num = 20
        sigma_x = 0.02
        sigma_y = 0.02

        # Data
        x = np.sort(np.random.random(num)) * 1
        y = qs.straight(x, 69, 0.42)
        sx = np.full(num, sigma_x)
        sy = np.full(num, sigma_y)
        c = np.sort(np.random.random(num))

        # Generating some statistics
        dxy = np.array([np.random.multivariate_normal(
            mean=[0, 0], cov=[[sigma_x ** 2, ci * sigma_x * sigma_y],
                              [ci * sigma_x * sigma_y, sigma_y ** 2]], size=1) for ci in c]).squeeze()
        x += dxy[:, 0]
        y += dxy[:, 1]

        # Fitting and plotting
        popt_cf, pcov_cf = qs.curve_fit(qs.straight, x, y, sigma=sy, report=True)
        # Do a standard fit with y-axis uncertainties.

        popt_of, pcov_of = qs.odr_fit(qs.straight, x, y, sigma_x=sx, sigma_y=sy, report=True)
        # Do an orthogonal-distance-regression fit with x- and y-axis uncertainties.

        popt_yf, pcov_yf = qs.york_fit(x, y, sigma_x=sx, sigma_y=sy, corr=c, report=True)
        # Do an orthogonal-distance-regression linear fit using the algorithm
        # of York et al. [York et al., Am. J. Phys. 72, 367 (2004)]
        # with x- and y-axis uncertainties as well as correlations between the two axes.

        popt_lf, pcov_lf = qs.linear_fit(x, y, sigma_x=sx, sigma_y=sy, corr=c, report=True)
        # Do an orthogonal-distance-regression linear fit using maximum likelihood optimization
        # with x- and y-axis uncertainties as well as correlations between the two axes.

        plt.plot(x, y, 'k.')
        qs.draw_sigma2d(x, y, sx, sy, c, n=1)  # Draw 2d-uncertainties with correlations.

        x_cont = np.linspace(-0.1, 1.1, 1001)

        plt.plot(x_cont, qs.straight(x_cont, *popt_cf), label='curve_fit')
        plt.plot(x_cont, qs.straight(x_cont, *popt_of), label='odr_fit')
        plt.plot(x_cont, qs.straight(x_cont, *popt_yf), label='york_fit')
        plt.plot(x_cont, qs.straight(x_cont, *popt_lf), label='linear_fit')
        plt.legend(loc=2)
        plt.show()

    if 1 in n:
        """
        Example 1: Linear fit in 3d.
        """
        size = 10
        dim = 3

        s = np.array([1, 2.7, -0.2])
        d = 1.
        rho = -0.8
        optimize_cov = False

        sigma = np.array([np.ones(dim) * d for _ in range(size)], dtype=float)
        mean = np.array([np.zeros(dim) + t * s for t in range(size)], dtype=float)
        mean = np.array([np.random.normal(scale=sigma[0]) + t * s for t in range(size)])
        # corr = np.array([np.array([[1, rho], [rho, 1]]) for _ in range(size)], dtype=float)
        corr = np.array([np.identity(dim) for _ in range(size)], dtype=float)
        cov = sigma[:, :, None] * sigma[:, None, :] * corr

        popt, pcov = qs.linear_nd_fit(mean, cov=cov, p0=None, axis=0, optimize_cov=optimize_cov)
        # popt, pcov = qs.linear_nd_monte_carlo(mean, cov=cov, n_samples=100000, method='py',
        #                                       axis=0, report=True, optimize_sampling=False, optimize_cov=optimize_cov)

        x_min, x_max = np.min(mean[:, 0]), np.max(mean[:, 0])
        x_lim = x_min - 0.2 * (x_max - x_min), x_max + 0.2 * (x_max - x_min)
        t = np.linspace(x_lim[0] - popt[0], x_lim[1] - popt[0], 2)

        x_fit = popt[0] + t * popt[3]
        y_fit = popt[1] + t * popt[4]
        s_fit = qs.straight_std(x_fit - popt[0], np.sqrt(pcov[1, 1]), np.sqrt(pcov[4, 4]),
                                pcov[1, 4] / np.sqrt(pcov[1, 1] * pcov[4, 4]))

        plt.subplot(1, 2, 1)
        plt.plot(mean[:, 0], mean[:, 1], '.k')
        qs.draw_sigma2d(mean[:, 0], mean[:, 1], sigma[:, 0], sigma[:, 1], corr[:, 0, 1])
        plt.plot(x_fit, y_fit, label='mvn')
        plt.fill_between(x_fit, y_fit - s_fit, y_fit + s_fit, alpha=0.3)

        x_fit = popt[0] + t * popt[3]
        y_fit = popt[2] + t * popt[5]
        s_fit = qs.straight_std(x_fit, np.sqrt(pcov[2, 2]), np.sqrt(pcov[5, 5]),
                                pcov[2, 5] / np.sqrt(pcov[2, 2] * pcov[5, 5]))

        plt.subplot(1, 2, 2)
        plt.plot(mean[:, 0], mean[:, 2], '.k')
        qs.draw_sigma2d(mean[:, 0], mean[:, 2], sigma[:, 0], sigma[:, 2], corr[:, 0, 2])
        plt.plot(x_fit, y_fit, label='mvn')
        plt.fill_between(x_fit, y_fit - s_fit, y_fit + s_fit, alpha=0.3)

        plt.show()

    if 2 in n:
        """
        Example 2: King-fit with different algorithms.
        """

        # The mass numbers of the stable Ca isotopes.
        a = np.array([40, 42, 43, 44, 46, 48])

        # The masses of the isotopes.
        m = np.array([(39.962590865, 22e-9), (41.958617828, 159e-9), (42.958766430, 244e-9),
                      (43.955481543, 348e-9), (45.953687988, 2399e-9), (47.952522904, 103e-9)])

        # Use absolute values given in the shape (#isotopes, #observables, 2).
        # (D1, D2, D3P1, D3P3, D5P3)
        vals = [[(755222765.661, 0.099), (761905012.531, 0.107),  # 40Ca
                 (346000235.128, 0.138), (352682481.933, 0.131), (350862882.626, 0.133)],

                [(755223191.147, 0.104), (761905438.574, 0.096),  # 42Ca
                 (345997884.981, 0.302), (352680132.391, 0.311), (350860536.899, 0.268)],

                [(0, 0), ] * 5,  # Fill the unkown axes.  # 43Ca

                [(755223614.656, 0.098), (761905862.618, 0.092),  # 44Ca
                 (345995736.343, 0.298), (352677984.496, 0.258), (350858392.304, 0.261)],

                [(0, 0), ] * 5,  # Fill the unkown axes.  # 46Ca

                [(755224471.117, 0.102), (761906720.109, 0.114),  # 48Ca
                 (345991937.638, 0.284), (352674186.532, 0.368), (350854600.002, 0.215)]]

        vals = np.array(vals, dtype=float)

        # Construct a King object
        king = qs.King(a=a, m=m, x_abs=vals, subtract_electrons=20, element_label='Ca', n_samples=500000)

        i_fit = np.array([1, 3, 5])
        a_fit = a[i_fit]  # Choose the isotopes to fit.

        i_ref = np.array([0, 5, 1])
        a_ref = a[i_ref]  # Choose individual reference isotopes

        # Put in values for isotopes known only for one observable, ...
        a_unknown = [43, 46]
        a_unknown_ref = [40, 40]
        y = np.array([(679.443, 0.568), (1299.071, 0.583)])

        xy = (0, 1)  # Choose the x- and y-axis (observables) to fit, i.e. vals[:, xy[1]] against vals[:, xy[0]].
        popt, pcov = king.fit(a_fit, a_ref, xy=xy, mode='shifts')  # Do a simple 2d-King fit.
        x, cov, cov_stat = king.get_unmodified(a_unknown, a_unknown_ref, y, axis=1, show=True)

        qs.printh('\nD1 isotope shifts (43, 46):')  # Print in pink!
        print(f'Fitted straight: {popt[0]} + {popt[1]} * x')
        for i in range(y.shape[0]):
            print(f'\n{a_unknown[i]}Ca - {a_unknown_ref[i]}Ca: {tuple(x[i])} +/- {tuple(np.sqrt(np.diag(cov[i])))}')
            qs.print_cov(cov[i], normalize=True)

        popt, pcov = king.fit_nd(a_fit, a_ref, axis=0, optimize_cov=True, mode='shifts', show=True)  # Do a 5d-King fit.
        x, cov, cov_stat = king.get_unmodified(a_unknown, a_unknown_ref, y, axis=1, show=True)

        qs.printh('\n(D1, D2, D3P1, D3P3, D5P3) isotope shifts (43, 46):')
        print(f'Fitted straight: {tuple(popt[:popt.size // 2])} + {tuple(popt[popt.size // 2:])} * x')
        for i in range(y.shape[0]):
            print(f'\n{a_unknown[i]}Ca - {a_unknown_ref[i]}Ca: {tuple(x[i])} +/- {tuple(np.sqrt(np.diag(cov[i])))}')
            qs.print_cov(cov[i], normalize=True)


if __name__ == '__main__':
    example({0})
