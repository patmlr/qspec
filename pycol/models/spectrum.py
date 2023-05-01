# -*- coding: utf-8 -*-
"""
pycol._lineshapes.spectrum

Created on 14.03.2022

@author: Patrick Mueller

Spectrum classes for lineshape models.
"""

import numpy as np
from scipy.stats import norm, cauchy
from scipy.special import voigt_profile

from pycol.physics import source_energy_pdf
from pycol.models.base import Model


# The names of the spectra. Includes all spectra that appear in the GUI.
SPECTRA = ['Gauss', 'Lorentz', 'Voigt', 'GaussChi2']


class Spectrum(Model):
    def __init__(self):
        super().__init__(model=None)
        self.type = 'Spectrum'

    def evaluate(self, x, *args, **kwargs):
        return np.zeros_like(x)

    @property
    def dx(self):
        return self.fwhm() * 0.02

    def fwhm(self):
        return 1.

    def min(self):
        return -5 * self.fwhm()

    def max(self):
        return 5 * self.fwhm()


class Lorentz(Spectrum):
    def __init__(self):
        super().__init__()
        self.type = 'Lorentz'

        self._add_arg('Gamma', 1., False, False)

    def evaluate(self, x, *args, **kwargs):  # Normalize to the maximum.
        scale = 0.5 * args[0]
        return np.pi * scale * cauchy.pdf(x, loc=0, scale=scale)

    def fwhm(self):
        return abs(self.vals[self.p['Gamma']])

    def min(self):
        return -10 * self.fwhm()

    def max(self):
        return 10 * self.fwhm()


class Gauss(Spectrum):
    def __init__(self):
        super().__init__()
        self.type = 'Gauss'

        self._add_arg('sigma', 1., False, False)

    def evaluate(self, x, *args, **kwargs):  # Normalize to the maximum.
        return np.sqrt(2 * np.pi) * args[0] * norm.pdf(x, loc=0, scale=args[0])

    def fwhm(self):
        return abs(np.sqrt(8 * np.log(2)) * self.vals[self.p['sigma']])

    def min(self):
        return -2.5 * self.fwhm()

    def max(self):
        return 2.5 * self.fwhm()


class Voigt(Spectrum):
    def __init__(self):
        super().__init__()
        self.type = 'Voigt'

        self._add_arg('Gamma', 1., False, False)
        self._add_arg('sigma', 1., False, False)

    def evaluate(self, x, *args, **kwargs):  # Normalize to the maximum.
        return voigt_profile(x, args[1], 0.5 * args[0]) / voigt_profile(0, args[1], 0.5 * args[0])

    def fwhm(self):
        f_l = self.vals[self.p['Gamma']]
        f_g = np.sqrt(8 * np.log(2)) * self.vals[self.p['sigma']]
        return abs(0.5346 * f_l + np.sqrt(0.2166 * f_l ** 2 + f_g ** 2))


def _gauss_chi2_taylor_fwhm(sigma, xi):
    a = [2.70991004e+00, 2.31314470e-01, -8.11610976e-02, -1.48897229e-02, 1.11677618e-01,  # order 1, 2
         8.06412708e-03, -6.59788156e-04, -1.06067275e-02, 1.47400503e-03]  # order 3
    return a[0] * sigma + a[1] * xi + (a[2] * sigma ** 2 + a[3] * xi ** 2 + a[4] * sigma * xi) / 2 \
        + (a[5] * sigma ** 3 + a[6] * xi ** 3 + a[7] * sigma ** 2 * xi + a[8] * sigma * xi ** 2) / 6


def _gauss_chi2_fwhm(sigma, xi):
    x = (sigma, xi)
    a, b, c = (1.39048239, 0.49098627, 0.61375536)
    return a * x[0] * (np.arctan(b * (np.abs(x[1] / x[0]) ** c - 1)) + np.arctan(b)) + np.sqrt(8 * np.log(2)) * x[0]


class GaussChi2(Spectrum):  # TODO: The fwhm of GaussChi2 is fitting quiet good now, but could be improved further.
    def __init__(self):
        super().__init__()
        self.type = 'GaussBoltzmann'

        self._add_arg('sigma', 1., False, False)
        self._add_arg('xi', 1., False, False)

    def evaluate(self, x, *args, **kwargs):  # Maximum unknown, normalize to 0 position, f(0) ~> 0.8 * max(f(x)).
        return source_energy_pdf(x, 0, args[0], args[1], collinear=True) \
            / source_energy_pdf(0, 0, args[0], args[1], collinear=True)

    def fwhm(self):
        sigma = np.abs(self.vals[self.p['sigma']])
        xi = np.abs(self.vals[self.p['xi']])
        if xi == 0:
            return np.sqrt(8 * np.log(2)) * sigma
        return _gauss_chi2_fwhm(sigma, xi)
        # return _gauss_chi2_taylor_fwhm(sigma, xi)

    def min(self):
        sigma = np.abs(self.vals[self.p['sigma']])
        xi = self.vals[self.p['xi']]
        if xi < 0:
            return -5 * np.sqrt(2 * np.log(2)) * sigma
        return -5 * (np.sqrt(2 * np.log(2)) * sigma + xi)

    def max(self):
        sigma = np.abs(self.vals[self.p['sigma']])
        xi = self.vals[self.p['xi']]
        if xi > 0:
            return 5 * np.sqrt(2 * np.log(2)) * sigma
        return 5 * (np.sqrt(2 * np.log(2)) * sigma - xi)
