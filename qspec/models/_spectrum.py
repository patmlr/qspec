# -*- coding: utf-8 -*-
"""
qspec.models._spectrum

Created on 14.03.2022

@author: Patrick Mueller

Spectrum classes for lineshape models.
"""

import numpy as np
from scipy.stats import norm, cauchy
from scipy.special import voigt_profile
from scipy.special import wofz

from qspec.physics import source_energy_pdf
from qspec.models._base import Model

__all__ = ['SPECTRA', 'fwhm_voigt', 'fwhm_voigt_d', 'Spectrum', 'Gauss', 'Lorentz', 'LorentzQI',
           'Voigt', 'VoigtDerivative', 'VoigtAsy', 'VoigtCEC', 'GaussChi2']


# The names of the spectra. Includes all spectra that appear in the GUI.
SPECTRA = ['Gauss', 'Lorentz', 'Voigt', 'VoigtDerivative', 'VoigtAsy', 'VoigtCEC', 'GaussChi2']


def fwhm_voigt(gamma, sigma):
    f_l = abs(gamma)
    f_g = abs(np.sqrt(8 * np.log(2)) * sigma)
    return 0.5346 * f_l + np.sqrt(0.2166 * f_l ** 2 + f_g ** 2)


def fwhm_voigt_d(gamma, gamma_d, sigma, sigma_d):
    f_l = abs(gamma)
    f_l_d = abs(gamma_d)
    f_g = abs(np.sqrt(8 * np.log(2)) * sigma)
    f_g_d = abs(np.sqrt(8 * np.log(2)) * sigma_d)
    f_g_d = f_g / np.sqrt(0.2166 * f_l ** 2 + f_g ** 2) * f_g_d
    f_l_d = (0.5346 + 0.2166 * f_l / np.sqrt(0.2166 * f_l ** 2 + f_g ** 2)) * f_l_d
    return np.sqrt(f_l_d ** 2 + f_g_d ** 2)


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


class LorentzQI(Spectrum):
    def __init__(self):
        super().__init__()
        self.type = 'LorentzQI'

        self._add_arg('Gamma', 1., False, False)

    def evaluate_qi(self, x, x_qi, *args, **kwargs):
        scale = 0.5 * args[0]
        return 2 * scale ** 2 * np.real(1 / ((x + 1j * scale) * (x - x_qi - 1j * scale)))

    def evaluate(self, x, *args, **kwargs):  # Normalize to the maximum.
        scale = 0.5 * args[0]
        return scale * np.pi * cauchy.pdf(x, loc=0, scale=scale)

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
        return fwhm_voigt(self.vals[self.p['Gamma']], self.vals[self.p['sigma']])


class VoigtDerivative(Spectrum):
    def __init__(self):
        super().__init__()
        self.type = 'VoigtDerivative'

        self._add_arg('Gamma', 1., False, False)
        self._add_arg('sigma', 1., False, False)

    def evaluate(self, x, *args, **kwargs):  # Normalize to the maximum. (Pos of min/max to be determined).
        z = (x + 1j * 0.5 * args[0]) / (np.sqrt(2) * args[1])
        fwhm = abs(0.5346 * args[0] + np.sqrt(0.2166 * args[0] ** 2 + args[1] ** 2))  # for normalization
        return -(z * wofz(z)).real / (np.sqrt(np.pi) * args[1] ** 2) / voigt_profile(0, args[1], 0.5 * args[0]) * fwhm

    def fwhm(self):
        return fwhm_voigt(self.vals[self.p['Gamma']], self.vals[self.p['sigma']])


class VoigtAsy(Spectrum):
    def __init__(self):
        super().__init__()
        self.type = 'VoigtAsy'

        self._add_arg('Gamma', 1., False, False)
        self._add_arg('sigma', 1., False, False)
        self._add_arg('asyPar', 1., False, False)

    def evaluate(self, x, *args, **kwargs):  # Normalize to the maximum.
        gamma = args[0] / (1 + np.exp(args[2] * x))
        return voigt_profile(x, args[1], gamma) / voigt_profile(0, args[1], 0.5 * args[0])

    def fwhm(self):
        return fwhm_voigt(self.vals[self.p['Gamma']], self.vals[self.p['sigma']])


class VoigtCEC(Spectrum):
    def __init__(self):
        super().__init__()
        self.type = 'VoigtCEC'

        self._add_arg('Gamma', 1., False, False)
        self._add_arg('sigma', 1., False, False)
        self._add_arg('shift', 0., False, False)
        self._add_arg('ratio', 0., False, False)
        self._add_arg('n', 0, True, False)

    def evaluate(self, x, *args, **kwargs):  # Normalize to the maximum.
        return np.sum([args[3] ** i * voigt_profile(x - i * args[2], args[1], 0.5 * args[0])
                       for i in range(int(args[4]) + 1)], axis=0) / voigt_profile(0, args[1], 0.5 * args[0])

    def fwhm(self):
        return fwhm_voigt(self.vals[self.p['Gamma']], self.vals[self.p['sigma']])


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
