# -*- coding: utf-8 -*-
"""
examples.stats_ex

Created on 25.01.2022

@author: Patrick Mueller

Example script / Guide for the qspec.stats module.
"""

import numpy as np
import matplotlib.pyplot as plt

import qspec as pc


def example(n=None):
    """
    Run one or several of the available examples. Scroll to the end for the function call.

    :param n: The number of the example or a list of numbers.

    Example 0: Create an observable with uncertainties and do statistics.

    Example 1: Fit a function and plot its median and 1-sigma percentiles.

    :returns: None.
    """
    if n is None:
        n = {0, }
    if isinstance(n, int):
        n = {n, }

    def sin(_x, _f, _a, _b, _c):
        return _a * np.sin(_x * _f + _b) + _c

    if 0 in n:
        """
        Example 0: Create an observable with uncertainties and do statistics.
        """
        f = pc.Observable(60, 1, 1.4)  # Create an observable with possibly asymmetric uncertainties.
        # f.hist()  # show the statistics of f.

        # Currently the ratio between the uncertainties must not exceed 1.5.
        # The statistic of the observable is estimated using a skew normal distribution.
        print('5 * f = {}'.format(5 * f))
        print('f ** 2 = {}'.format(f ** 2))
        print('f + 2 * f = {}'.format(f + 2 * f))  # If formulae are getting to long or contain non-standard operators,
        # it is more efficient/necessary to use propagate and define the function (see 'sin' above):

        x, a, b, c = 0.2, 3, 10, 5
        print('\n1. sin = {}'.format(pc.propagate(sin, [x, f, a, b, c])))

        x = pc.Observable(x, 0.01)  # All values with uncertainties.
        a = pc.Observable(a, 0.2, 0.18)
        b = pc.Observable(b, 0.8)
        c = pc.Observable(c, 1, 1.1)
        print('2. sin = {}'.format(pc.propagate(sin, [x, f, a, b, c])))

        num = 101
        x = np.linspace(-0.2, 0.2, num)
        p0 = f.rvs(num), a.rvs(num), b.rvs(num), c.rvs(num)
        y = sin(x, *p0)  # generate samples with .rvs()
        popt, pcov = pc.curve_fit(sin, x, y, p0=np.array(p0)[:, 0])
        plt.plot(x, y, 'C0.')
        plt.plot(x, sin(x, *popt), 'C1-')
        plt.show()

    if 1 in n:
        """
        Example 1: Fit a function and plot its median and 1-sigma percentiles.
        """
        f = pc.Observable(60, 0.5, 0.7)
        a = pc.Observable(3, 0.1, 0.13)
        b = pc.Observable(10, 0.8)
        c = pc.Observable(5, 1, 1.1)

        num = 101
        x = np.linspace(-0.2, 0.2, num)
        p0 = f.rvs(num), a.rvs(num), b.rvs(num), c.rvs(num)  # generate samples with .rvs()
        y = sin(x, *p0)
        sigma = np.ones_like(x) * 1
        # until here its just generating data. The use case starts below.

        # fit the sin-function, use the first sample for initialization.
        popt, pcov = pc.curve_fit(sin, x, y, sigma=sigma, absolute_sigma=True, p0=np.array(p0)[:, 0])
        y_fit = sin(x, *popt)

        # plot the data and fitted function
        plt.errorbar(x, y, yerr=sigma, fmt='C0.', label='Data')
        plt.plot(x, y_fit, 'C2-', label='Fit')

        # get proper error bands including correlations.
        y_med, y_min, y_max = pc.propagate_fit(sin, x, popt, pcov, sample_size=100000)
        plt.plot(x, y_med, 'C1-', label='Median')
        plt.fill_between(x, y_min, y_max, color='C1', alpha=0.7)

        plt.legend(loc=4)
        plt.show()


if __name__ == '__main__':
    example({0, 1, })
