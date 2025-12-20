"""
qspec.models._spectrum
======================

Spectrum classes for lineshape models.
"""

import numpy as np
from scipy.special import voigt_profile, wofz
from scipy.stats import cauchy, norm

from qspec.models._base import Model
from qspec.physics import source_energy_pdf
from qspec.qtypes import Any, array_like, asarray, ndarray, scalar

__all__ = [
    "SPECTRA",
    "Gauss",
    "GaussChi2",
    "Lorentz",
    "LorentzQI",
    "Spectrum",
    "Voigt",
    "VoigtAsy",
    "VoigtCEC",
    "VoigtDerivative",
    "fwhm_voigt",
    "fwhm_voigt_d",
]


# The names of the spectra. Includes all spectra that appear in the GUI.
SPECTRA = ["Gauss", "Lorentz", "Voigt", "VoigtDerivative", "VoigtAsy", "VoigtCEC", "GaussChi2"]


def fwhm_voigt(gamma: array_like, sigma: array_like) -> ndarray:
    r"""
    The full width at half maximum (FWHM) of a Voigt profile

    $$
    \delta\nu = 0.5346\Gamma + \sqrt{0.2166\Gamma^2 + 8\log(2)\sigma^2}
    $$

    :param gamma: The FWHM of the Lorentzian $\Gamma$ (arb. unit).
    :param sigma: The standard deviation of the Gaussian $\sigma$ ([`gamma`]).
    :returns: The FWHM $\delta\nu$ of a Voigt profile.
    """
    gamma, sigma = np.asarray(gamma, dtype=float), np.asarray(sigma, dtype=float)

    f_l = np.abs(gamma)
    f_g = np.abs(np.sqrt(8 * np.log(2)) * sigma)

    return 0.5346 * f_l + np.sqrt(0.2166 * f_l**2 + f_g**2)


def fwhm_voigt_d(gamma: array_like, gamma_d: array_like, sigma: array_like, sigma_d: array_like) -> ndarray:
    r"""
    The uncertainty of the full width at half maximum (FWHM) of a Voigt profile

    $$
    \Delta\delta\nu = \sqrt{\left(\frac{8\log(2)\sigma\Delta\sigma}{\sqrt{0.2166\Gamma^2 + 8\log(2)\sigma^2}}\right)^2
    + \left(\left(0.5346 + \frac{0.2166\Gamma}{\sqrt{0.2166\Gamma^2 + 8\log(2)\sigma^2}}\right)\Gamma_d\right)^2}
    $$

    :param gamma: The FWHM of the Lorentzian $\Gamma$ (arb. unit).
    :param gamma_d: The uncertainty of `gamma` $\Delta\Gamma$.
    :param sigma: The standard deviation of the Gaussian $\sigma$ ([`gamma`]).
    :param sigma_d: The uncertainty of `sigma` $\Delta\sigma$.
    :returns: The uncertainty of the FWHM of a Voigt profile $\Delta\delta\nu$.
    """
    gamma, gamma_d, sigma, sigma_d = asarray(gamma, gamma_d, sigma, sigma_d, dtype=float)

    f_l = abs(gamma)
    f_l_d = abs(gamma_d)

    f_g = abs(np.sqrt(8 * np.log(2)) * sigma)
    f_g_d = abs(np.sqrt(8 * np.log(2)) * sigma_d)

    f_g_d = f_g / np.sqrt(0.2166 * f_l**2 + f_g**2) * f_g_d
    f_l_d = (0.5346 + 0.2166 * f_l / np.sqrt(0.2166 * f_l**2 + f_g**2)) * f_l_d

    return np.sqrt(f_l_d**2 + f_g_d**2)


class Spectrum(Model):
    def __init__(self) -> None:
        super().__init__(model=None)
        self.type = "Spectrum"

    def evaluate(self, x: ndarray, *args: scalar, **kwargs: Any) -> ndarray:
        return np.zeros_like(x)

    @property
    def dx(self) -> float:
        """
        A hint for an x-axis step size for a smooth display of the model.
        """
        return self.fwhm() * 0.02

    def fwhm(self) -> float:
        """
        :returns: The full width at half maximum (FWHM) of a spectrum.
        """
        return 1.0

    def min(self) -> float:
        """
        :returns: A hint for an x-axis minimum for a complete display of the model.
        """
        return -5 * self.fwhm()

    def max(self) -> float:
        """
        :returns: A hint for an x-axis maximum for a complete display of the model.
        """
        return 5 * self.fwhm()


class Lorentz(Spectrum):
    def __init__(self) -> None:
        r"""
        A Lorentzian peak function of the form

        $$
        f(x, \Gamma) = \frac{\Gamma^2}{(2x)^2 + \Gamma^2}.
        $$
        """
        super().__init__()
        self.type = "Lorentz"

        self._add_arg("Gamma", 1.0, False, False)

    def evaluate(self, x: array_like, *args: scalar, **kwargs: Any) -> ndarray:  # Normalize to the maximum.
        scale = 0.5 * args[0]
        return np.pi * scale * cauchy.pdf(x, loc=0, scale=scale)

    def fwhm(self) -> float:
        """
        :returns: The full width at half maximum (FWHM) of a spectrum.
        """
        return abs(float(self.vals[self.p["Gamma"]]))

    def min(self) -> float:
        """
        :returns: A hint for an x-axis minimum for a complete display of the model.
        """
        return -10 * self.fwhm()

    def max(self) -> float:
        """
        :returns: A hint for an x-axis maximum for a complete display of the model.
        """
        return 10 * self.fwhm()


class LorentzQI(Spectrum):
    def __init__(self) -> None:
        r"""
        A Lorentzian peak function of the form

        $$
        f(x, \Gamma) = \frac{\Gamma^2}{(2x)^2 + \Gamma^2}.
        $$

        A dispersive term for consideration of quantum interference (QI) effects is available through `evaluate_qi`

        $$
        f(x, x_\mathrm{qi}, \Gamma) = \mathrm{Re}\left[\frac{\Gamma^2}
        {(2x + i\Gamma)(2(x - x_\mathrm{qi}) - i\Gamma)}\right].
        $$

        This model needs to be implemented in another model that combines `evaluate` with `evaluate_qi`,
        see `HyperfineQI`.
        """
        super().__init__()
        self.type = "LorentzQI"

        self._add_arg("Gamma", 1.0, False, False)

    def evaluate_qi(self, x: ndarray, x_qi: ndarray, *args: scalar, **kwargs: Any) -> ndarray:
        """
        :param x: The input values.
        :param x_qi: The x value of the interfering Lorentzian.
        :param args: The function parameters. Must have length self.size.
        :param kwargs: Additional keyword arguments.
        :returns: The function results of the dispersive Lorentzian at the input values 'x'.
        """
        scale = 0.5 * args[0]
        return 2 * scale**2 * np.real(1 / ((x + 1j * scale) * (x - x_qi - 1j * scale)))

    def evaluate(self, x: ndarray, *args: scalar, **kwargs: Any) -> ndarray:  # Normalize to the maximum.
        scale = 0.5 * args[0]
        return scale * np.pi * cauchy.pdf(x, loc=0, scale=scale)

    def fwhm(self) -> float:
        """
        :returns: The full width at half maximum (FWHM) of a spectrum.
        """
        return abs(self.vals[self.p["Gamma"]])

    def min(self) -> float:
        """
        :returns: A hint for an x-axis minimum for a complete display of the model.
        """
        return -10 * self.fwhm()

    def max(self) -> float:
        """
        :returns: A hint for an x-axis maximum for a complete display of the model.
        """
        return 10 * self.fwhm()


class Gauss(Spectrum):
    def __init__(self) -> None:
        r"""
        A Gaussian peak function of the form

        $$
        f(x, \sigma) = \exp\left(-\frac{x^2}{2\sigma^2}\right).
        $$
        """
        super().__init__()
        self.type = "Gauss"

        self._add_arg("sigma", 1.0, False, False)

    def evaluate(self, x: ndarray, *args: scalar, **kwargs: Any) -> ndarray:  # Normalize to the maximum.
        return np.sqrt(2 * np.pi) * args[0] * norm.pdf(x, loc=0, scale=args[0])

    def fwhm(self) -> float:
        """
        :returns: The full width at half maximum (FWHM) of a spectrum.
        """
        return abs(np.sqrt(8 * np.log(2)) * self.vals[self.p["sigma"]])

    def min(self) -> float:
        """
        :returns: A hint for an x-axis minimum for a complete display of the model.
        """
        return -2.5 * self.fwhm()

    def max(self) -> float:
        """
        :returns: A hint for an x-axis maximum for a complete display of the model.
        """
        return 2.5 * self.fwhm()


class Voigt(Spectrum):
    def __init__(self) -> None:
        r"""
        A Voigt peak function of the form

        $$
        f(x, \Gamma, \sigma) = \frac{\mathrm{Re}[w(z)]}{\mathrm{Re}[w(i\Gamma/(\sqrt{8}\sigma))]},
        \quad z = \frac{2x + i\Gamma}{\sqrt{8}\sigma},
        $$

        with the complex Faddeeva function $w$.
        """
        super().__init__()
        self.type = "Voigt"

        self._add_arg("Gamma", 1.0, False, False)
        self._add_arg("sigma", 1.0, False, False)

    def evaluate(self, x: ndarray, *args: scalar, **kwargs: Any) -> ndarray:  # Normalize to the maximum.
        return voigt_profile(x, args[1], 0.5 * args[0]) / voigt_profile(0, args[1], 0.5 * args[0])

    def fwhm(self) -> float:
        """
        :returns: The full width at half maximum (FWHM) of a spectrum.
        """
        return float(fwhm_voigt(self.vals[self.p["Gamma"]], self.vals[self.p["sigma"]]))


class VoigtDerivative(Spectrum):
    def __init__(self) -> None:
        r"""
        The first derivative of a Voigt peak function of the form

        $$\begin{aligned}
        f(x, \Gamma, \sigma) &= -\frac{\sqrt{2}\delta\nu}{\sigma}\frac{\mathrm{Re}[z w(z)]}
        {\mathrm{Re}[w(i\Gamma/(\sqrt{8}\sigma))]},
        \quad z = \frac{2x + i\Gamma}{\sqrt{8}\sigma},\\[2ex]
        \delta\nu &= 0.5346\Gamma + \sqrt{0.2166\Gamma^2 + 8\log(2)\sigma^2},
        \end{aligned}$$

        with the complex Faddeeva function $w$.
        """
        super().__init__()
        self.type = "VoigtDerivative"

        self._add_arg("Gamma", 1.0, False, False)
        self._add_arg("sigma", 1.0, False, False)

    def evaluate(
        self, x: ndarray, *args: scalar, **kwargs: Any
    ) -> ndarray:  # Normalize to the maximum. (Pos of min/max to be determined).
        z = (x + 1j * 0.5 * args[0]) / (np.sqrt(2) * args[1])
        fwhm = abs(0.5346 * args[0] + np.sqrt(0.2166 * args[0] ** 2 + args[1] ** 2))  # for normalization
        return -(z * wofz(z)).real / (np.sqrt(np.pi) * args[1] ** 2) / voigt_profile(0, args[1], 0.5 * args[0]) * fwhm

    def fwhm(self) -> float:
        """
        :returns: The full width at half maximum (FWHM) of a spectrum.
        """
        return float(fwhm_voigt(self.vals[self.p["Gamma"]], self.vals[self.p["sigma"]]))


class VoigtAsy(Spectrum):
    def __init__(self) -> None:
        r"""
        An asymmetric Voigt peak function of the form

        $$
        f(x, \Gamma, \sigma, \alpha) = \frac{\mathrm{Re}[w(z)]}{\mathrm{Re}[w(i\Gamma/(\sqrt{8}\sigma))]},
        \quad z = \frac{2x + i[\Gamma/(1 + \exp(\alpha x))]}{\sqrt{8}\sigma},
        $$

        with the complex Faddeeva function $w$.
        """
        super().__init__()
        self.type = "VoigtAsy"

        self._add_arg("Gamma", 1.0, False, False)
        self._add_arg("sigma", 1.0, False, False)
        self._add_arg("asyPar", 1.0, False, False)

    def evaluate(self, x: ndarray, *args: scalar, **kwargs: Any) -> ndarray:  # Normalize to the maximum.
        gamma = args[0] / (1 + np.exp(args[2] * x))
        return voigt_profile(x, args[1], gamma) / voigt_profile(0, args[1], 0.5 * args[0])

    def fwhm(self) -> float:
        """
        :returns: The full width at half maximum (FWHM) of a spectrum.
        """
        return float(fwhm_voigt(self.vals[self.p["Gamma"]], self.vals[self.p["sigma"]]))


class VoigtCEC(Spectrum):
    def __init__(self) -> None:
        r"""
        A series of `n + 1` `Voigt` peak functions of the form

        $$
        f[n](x, \Gamma, \sigma, x_\mathrm{cec}, p_\mathrm{cec}) = \sum\limits_{k = 0}^n p_\mathrm{cec}^k\mathrm{Voigt}
        (x - k x_\mathrm{cec}, \Gamma, \sigma).
        $$
        """
        super().__init__()
        self.type = "VoigtCEC"

        self._add_arg("Gamma", 1.0, False, False)
        self._add_arg("sigma", 1.0, False, False)
        self._add_arg("shift", 0.0, False, False)
        self._add_arg("ratio", 0.0, False, False)
        self._add_arg("n", 0, True, False)

    def evaluate(self, x: ndarray, *args: scalar, **kwargs: Any) -> ndarray:  # Normalize to the maximum.
        return np.sum(
            [args[3] ** i * voigt_profile(x - i * args[2], args[1], 0.5 * args[0]) for i in range(int(args[4]) + 1)],
            axis=0,
        ) / voigt_profile(0, args[1], 0.5 * args[0])

    def fwhm(self) -> float:
        """
        :returns: The full width at half maximum (FWHM) of a spectrum.
        """
        return float(fwhm_voigt(self.vals[self.p["Gamma"]], self.vals[self.p["sigma"]]))


def _gauss_chi2_taylor_fwhm(sigma: float, xi: float) -> float:
    """
    :param sigma: The standard deviation of the Gaussian.
    :param xi: The asymmetry parameter.
    :returns: An estimated Taylor expansion of the FWHM of the 'GaussChi2' model.
    """
    a = [
        2.70991004e00,
        2.31314470e-01,
        -8.11610976e-02,
        -1.48897229e-02,
        1.11677618e-01,  # order 1, 2
        8.06412708e-03,
        -6.59788156e-04,
        -1.06067275e-02,
        1.47400503e-03,
    ]  # order 3
    return (
        a[0] * sigma
        + a[1] * xi
        + (a[2] * sigma**2 + a[3] * xi**2 + a[4] * sigma * xi) / 2
        + (a[5] * sigma**3 + a[6] * xi**3 + a[7] * sigma**2 * xi + a[8] * sigma * xi**2) / 6
    )


def _gauss_chi2_fwhm(sigma: float, xi: float) -> float:
    """
    :param sigma: The standard deviation of the Gaussian.
    :param xi: The asymmetry parameter.
    :returns: An estimation of the FWHM of the 'GaussChi2' model.
    """
    x = (sigma, xi)
    a, b, c = (1.39048239, 0.49098627, 0.61375536)
    return a * x[0] * (np.arctan(b * (np.abs(x[1] / x[0]) ** c - 1)) + np.arctan(b)) + np.sqrt(8 * np.log(2)) * x[0]


class GaussChi2(Spectrum):  # TODO: The fwhm of GaussChi2 is fitting quite well now, but could be improved further.
    """
    An analytical convolution of a Gaussian with a Boltzmann distribution.
    This could be the velocity distribution of a thermalized ion ensemble produced
    on a linear or noisy voltage potential.
    """

    def __init__(self) -> None:
        super().__init__()
        self.type = "GaussChi2"

        self._add_arg("sigma", 1.0, False, False)
        self._add_arg("xi", 1.0, False, False)

    def evaluate(
        self, x: ndarray, *args: scalar, **kwargs: Any
    ) -> ndarray:  # Maximum unknown, normalize to 0 position, f(0) ~> 0.8 * max(f(x)).
        return source_energy_pdf(x, 0, args[0], args[1], collinear=True) / source_energy_pdf(
            0, 0, args[0], args[1], collinear=True
        )

    def fwhm(self) -> float:
        """
        :returns: The (approximate) full width at half maximum (FWHM) of a spectrum.
        """
        sigma = abs(self.vals[self.p["sigma"]])
        xi = abs(self.vals[self.p["xi"]])
        if xi == 0:
            return np.sqrt(8 * np.log(2)) * sigma
        return _gauss_chi2_fwhm(sigma, xi)
        # return _gauss_chi2_taylor_fwhm(sigma, xi)

    def min(self) -> float:
        """
        :returns: A hint for an x-axis minimum for a complete display of the model.
        """
        sigma = np.abs(self.vals[self.p["sigma"]])
        xi = self.vals[self.p["xi"]]
        if xi < 0:
            return -5 * np.sqrt(2 * np.log(2)) * sigma
        return -5 * (np.sqrt(2 * np.log(2)) * sigma + xi)

    def max(self) -> float:
        """
        :returns: A hint for an x-axis maximum for a complete display of the model.
        """
        sigma = np.abs(self.vals[self.p["sigma"]])
        xi = self.vals[self.p["xi"]]
        if xi > 0:
            return 5 * np.sqrt(2 * np.log(2)) * sigma
        return 5 * (np.sqrt(2 * np.log(2)) * sigma - xi)
