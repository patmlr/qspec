# -*- coding: utf-8 -*-
"""
examples.analyze_ex

Created on 15.08.2021

@author: Patrick Mueller

Example script / Guide for the PyCLS.analyze module.
"""

import numpy as np
import matplotlib.pyplot as plt

import pycol.analyze as an


def linear_fit(n=20, sigma_x=0.02, sigma_y=0.02, fit_functions=None):
    # Data
    x = np.sort(np.random.random(n)) * 1
    y = an.straight(x, 69, 0.42)
    sx = np.full(n, sigma_x)
    sy = np.full(n, sigma_y)
    c = np.sort(np.random.random(n))

    # Generating some statistics
    dxy = np.array([np.random.multivariate_normal(mean=[0, 0], cov=[[sigma_x ** 2, ci * sigma_x * sigma_y],
                                                                    [ci * sigma_x * sigma_y, sigma_y ** 2]],
                                                  size=1) for ci in c]).squeeze()
    x += dxy[:, 0]
    y += dxy[:, 1]

    # Fitting and plotting

    popt_cf, pcov_cf = an.curve_fit(an.straight, x, y, sigma=sy)
    popt_of, pcov_of = an.odr_fit(an.straight, x, y, sigma_x=sx, sigma_y=sy)
    a, b, sigma_a, sigma_b, c_ab = an.york(x, y, sigma_x=sx, sigma_y=sy, corr=c)

    plt.plot(x, y, 'k.')
    an.draw_sigma2d(x, y, sx, sy, c, n=1)

    x_cont = np.linspace(-0.1, 1.1, 1001)
    plt.plot(x_cont, an.straight(x_cont, *popt_cf), label='curve_fit')
    # an.draw_straight_unc_area(x_cont, an.straight(x_cont, *popt_cf), np.sqrt(pcov_cf[0, 0]), np.sqrt(pcov_cf[1, 1]),
    #                           pcov_cf[0, 1] / (np.sqrt(pcov_cf[0, 0]) * np.sqrt(pcov_cf[1, 1])))
    plt.plot(x_cont, an.straight(x_cont, *popt_of), label='odr_fit')
    plt.plot(x_cont, an.straight(x_cont, a, b), label='york')
    plt.show()


linear_fit()
