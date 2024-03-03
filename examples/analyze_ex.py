# -*- coding: utf-8 -*-
"""
examples.analyze_ex

Created on 15.08.2021

@author: Patrick Mueller

Example script / Guide for the qspec.analyze module.
"""

from time import time
import numpy as np
import matplotlib.pyplot as plt

import qspec as qs


def example(n=None):
    """
    Run one or several of the available examples. Scroll to the end for the function call.

    :param n: The number of the example or a list of numbers.

    Example 0: Fit of a straight line with different algorithms.

    Example 1: King-fit with different algorithms.

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

        a, b, sigma_a, sigma_b, c_ab = qs.york_fit(x, y, sigma_x=sx, sigma_y=sy, corr=c, report=True)
        # Do an orthogonal-distance-regression linear fit using the algorithm
        # of York et al. [York et al., Am. J. Phys. 72, 367 (2004)]
        # with x- and y-axis uncertainties as well as correlations between the two axes.

        plt.plot(x, y, 'k.')
        qs.draw_sigma2d(x, y, sx, sy, c, n=1)  # Draw 2d-uncertainties with correlations.

        x_cont = np.linspace(-0.1, 1.1, 1001)

        plt.plot(x_cont, qs.straight(x_cont, *popt_cf), label='curve_fit')

        qs.draw_straight_unc_area(  # Draw the uncertainty area of the curve_fit result.
            x_cont, qs.straight(x_cont, *popt_cf), np.sqrt(pcov_cf[0, 0]), np.sqrt(pcov_cf[1, 1]),
            pcov_cf[0, 1] / (np.sqrt(pcov_cf[0, 0]) * np.sqrt(pcov_cf[1, 1])))

        plt.plot(x_cont, qs.straight(x_cont, *popt_of), label='odr_fit')
        plt.plot(x_cont, qs.straight(x_cont, a, b), label='york')
        plt.legend(loc=2)
        plt.show()

    if 1 in n:
        """
        Example 1: King-fit with different algorithms.
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
        king = qs.King(a=a, m=m, x_abs=vals, subtract_electrons=20, element_label='Ca', n_samples=2325764)  # 2325764, 500000

        i_fit = np.array([1, 3, 5])
        a_fit = a[i_fit]  # Choose the isotopes to fit.

        i_ref = np.array([0, 5, 1])
        a_ref = a[i_ref]  # Choose individual reference isotopes

        xy = (0, 1)  # Choose the x- and y-axis (observables) to fit, i.e. vals[:, xy[1]] against vals[:, xy[0]].
        results = king.fit(a_fit, a_ref, xy=xy, mode='shifts')  # Do a simple 2d-King fit.
        t = time()
        results = king.fit_nd(a_fit, a_ref, axis=0, mode='shifts', show=False)  # Do a 5d-King fit.
        print(f'Time difference (sec) = {time() - t:.3f}')

        # Put in values for isotopes known only for one observable, ...
        a_unknown = [43, 46]
        a_unknown_ref = [40, 40]
        y = np.array([(679.443, 0.568), (1299.071, 0.583)])

        # ... get spectroscopically improved values for the other observables and ...
        x = king.get_unmodified_x(a_unknown, a_unknown_ref, y, show=True)  # Requires king.fit().
        qs.printh('\nD1 isotope shifts (43, 46):')  # Print in pink!
        print(x)
        x = king.get_unmodified_x_nd(a_unknown, a_unknown_ref, y, 1)  # Requires king.fit_nd().
        qs.printh('\n(D1, D2, D3P1, D3P3, D5P3) isotope shifts (43, 46):')
        print(x)

        # ... return the parameters for the straight fit.
        k, f, corr = king.get_popt()
        k, f, corr = king.get_popt_nd()

    if 2 in n:
        size = 5
        dim = 2
        
        s = np.array([1, 2.7])  # 2.
        d = 0.5
        rho = -0.8

        sigma = np.array([np.ones(dim) * d for _ in range(size)], dtype=float)
        mean = np.array([np.zeros(dim) + t * s for t in range(size)], dtype=float)
        mean = np.array([np.random.normal(scale=sigma[0]) + t * s for t in range(size)])
        corr = np.array([np.array([[1, rho], [rho, 1]]) for _ in range(size)], dtype=float)
        cov = sigma[:, :, None] * sigma[:, None, :] * corr

        plt.plot(mean[:, 0], mean[:, 1], '.k')
        qs.draw_sigma2d(mean[:, 0], mean[:, 1], sigma[:, 0], sigma[:, 1], corr[:, 0, 1])
        x_lim = plt.xlim()
        t = np.linspace(x_lim[0], x_lim[1], 2)

        a, b, sigma_a, sigma_b, corr_ab = qs.york_fit(mean[:, 0], mean[:, 1], sigma[:, 0], sigma[:, 1], corr[:, 0, 1])
        plt.plot(t, qs.straight(t, a, b), label='york')
        print(a, b)
        print(sigma_a ** 2, sigma_b ** 2, corr_ab * sigma_a * sigma_b)

        p0 = np.array([0, 3, 1, 1])
        popt, pcov = qs.linear_nd_fit(mean, cov=cov, p0=p0, axis=None, optimize_cov=True)
        # plt.plot(t, qs.straight(t, popt[1], popt[3]), label='mvn')
        plt.plot(popt[0] + t * popt[2], popt[1] + t * popt[3], label='mvn')
        print(popt)
        print(pcov)

        popt, pcov, *_ = qs.linear_nd_monte_carlo(mean, cov, n_samples=100000, method='py', axis=None,
                                                  optimize_cov=True, report=False)
        plt.plot(popt[0] + t * popt[2], popt[1] + t * popt[3], label='mc')
        print(popt)
        print(pcov)

        plt.legend()
        plt.show()


if __name__ == '__main__':
    example({2, })
