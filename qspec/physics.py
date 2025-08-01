# -*- coding: utf-8 -*-
"""
qspec.physics
=============

Module for physical functions useful for CLS.
"""

import string
import numpy as np
import scipy.constants as sc
import scipy.stats as st
import scipy.special as sp

from qspec.qtypes import *
from qspec import tools
from qspec.algebra import mu_j_m1, mu_jj_m1

__all__ = ['L_LABEL', 'E_NORM', 'pi', 'LEMNISCATE', 'mu_N', 'mu_B', 'g_s', 'me_u', 'me_u_d', 'gp_s', 'gn_s',
           'inv_cm_to_freq', 'freq_to_inv_cm', 'wavelength_to_freq', 'freq_to_wavelength', 'inv_cm_to_wavelength',
           'wavelength_to_inv_cm', 'beta', 'gamma', 'gamma_e', 'gamma_e_kin', 'e_rest', 'e_kin', 'e_total', 'e_el',
           'v_e', 'v_e_d1', 'v_el', 'v_el_d1', 'p_v', 'p_e', 'p_el', 'doppler', 'doppler_d1', 'doppler_e_d1',
           'doppler_el_d1', 'inverse_doppler', 'inverse_doppler_d1', 'alpha_atom', 'v_recoil', 'f_recoil',
           'f_recoil_v', 'get_f', 'get_m', 'hyperfine', 'lande_n', 'lande_j', 'lande_jj', 'g_j', 'lande_f',
           'zeeman_linear', 'hyper_zeeman_linear',
           'hyper_zeeman_ij', 'hyper_zeeman_num', 'hyper_zeeman_12', 'hyper_zeeman_12_d', 'a_hyper_mu', 'a_einstein_m1',
           'temperature_doppler', 'saturation_intensity', 'saturation', 'rabi_s', 'scattering_rate', 'mass_factor',
           'delta_r2', 'delta_r4', 'delta_r6', 'lambda_r', 'lambda_rn', 'schmidt_line', 'sellmeier',
           'gamma_3d', 'boost', 'doppler_3d', 'gaussian_beam_3d', 'gaussian_doppler_3d', 't_xi', 'normal_vx_pdf',
           'normal_vx_rvs', 'chi2_ex_pdf', 'chi2_ex_rvs', 'convolved_boltzmann_norm_pdf',
           'convolved_thermal_norm_v_pdf', 'convolved_thermal_norm_f_pdf', 'convolved_thermal_norm_f_lin_pdf',
           'source_energy_pdf']


L_LABEL = ['S', 'P', 'D', ] + list(string.ascii_uppercase[5:])
E_NORM = sc.e
pi = np.pi
LEMNISCATE = 2.6220575543
mu_N = sc.physical_constants['nuclear magneton'][0]
mu_B = sc.physical_constants['Bohr magneton'][0]
g_s = sc.physical_constants['electron g factor'][0]
me_u = sc.physical_constants['electron mass in u'][0]
me_u_d = sc.physical_constants['electron mass in u'][2]
gp_s = sc.physical_constants['proton g factor'][0]
gn_s = sc.physical_constants['neutron g factor'][0]


""" Units """

# --- freq <-> cm ---

def inv_cm_to_freq(k: array_like) -> ndarray:
    r"""
    Convert cm<sup>-1</sup> into MHz using $f = 10^{-4}ck$.

    :param k: The wavenumber $k \equiv f/c$ of a transition (1/cm).
    :returns: (freq) The frequency corresponding to the wavenumber `k` (MHz).
    """
    return np.asarray(k, dtype=float) * sc.c * 1e-4


def freq_to_inv_cm(f: array_like) -> ndarray:
    r"""
    Convert MHz into cm<sup>-1</sup> using $k = 10^4 f / c$.

    :param f: The frequency $f \equiv ck$ of a transition (MHz).
    :returns: (k) The wavenumber $k$ corresponding to the frequency `f` (1/cm).
    """
    return np.asarray(f, dtype=float) / sc.c * 1e4


# --- wav <-> freq ---

def wavelength_to_freq(lam: array_like) -> ndarray:
    r"""
    Convert &mu;m into MHz using $f = c / \lambda$.

    :param lam: The wavelength $\lambda$ of a transition (&mu;m).
    :returns: (freq) The frequency corresponding to the wavelength `lam` (MHz).
    """
    return sc.c / np.asarray(lam, dtype=float)


def freq_to_wavelength(f: array_like) -> ndarray:
    r"""
    Convert MHz into &mu;m using $\lambda = c / f$.

    :param f: The frequency $f$ of a transition (MHz).
    :returns: (lambda) The wavelength corresponding to the frequency `f` (&mu;m).
    """
    return sc.c / np.asarray(f, dtype=float)


# --- cm <-> wav ---

def inv_cm_to_wavelength(k: array_like) -> ndarray:
    r"""
    Convert cm<sup>-1</sup> into &mu;m using $\lambda = 10^4 / k$.

    :param k: The wavenumber $k$ of a transition (cm<sup>-1</sup>).
    :returns: (lambda) The wavelength corresponding to the wavenumber `k` (um).
    """
    return 1e4 / np.asarray(k, dtype=float)


def wavelength_to_inv_cm(lam: array_like) -> ndarray:
    r"""
    Convert &mu;m into cm<sup>-1</sup> using $\lambda = 10^4 / \lambda$.

    :param lam: The wavelength $\lambda$ of a transition (&mu;m).
    :returns: (k) The wavenumber $k$ corresponding to the wavelength `lam` (cm<sup>-1</sup>).
    """
    return 1e4 / np.asarray(lam, dtype=float)


""" 1-D kinematics """


def beta(v: array_like) -> ndarray:
    r"""
    The relativistic velocity $\beta = v / c$.

    :param v: The velocity $v$ of a body (m/s).
    :returns: (beta) The velocity `v` relative to the vacuum speed of light $c$.
    """
    return np.asarray(v, dtype=float) / sc.c


def gamma_beta(b: array_like) -> ndarray:
    r"""
    The time-dilation/Lorentz factor $\gamma = \sqrt{1 - \beta^2}^{-1}$.

    :param b: The relativistic velocity $\beta$ of a body.
    :returns: (gamma) The time-dilation/Lorentz factor $\gamma$ corresponding to the relativistic velocity `b`.
    """
    return 1. / np.sqrt(1. - np.asarray(b, dtype=float) ** 2)


def gamma(v: array_like) -> ndarray:
    r"""
    The time-dilation/Lorentz factor $\gamma = \sqrt{1 - (v/c)^2}^{-1}$.

    :param v: The velocity of a body (m/s).
    :returns: (gamma) The time-dilation/Lorentz factor $\gamma$ corresponding to the velocity `v`.
    """
    return gamma_beta(beta(v))


def gamma_e(e: array_like, m: array_like) -> ndarray:
    r"""
    The time-dilation/Lorentz factor $\gamma = E / (mc^2)$.

    :param e: The total energy $E$ of a body, including the energy of the rest mass (eV).
    :param m: The mass $m$ of the body (u).
    :returns: (gamma) The time-dilation/Lorentz factor $\gamma$ corresponding to the total energy `e` of a body with mass `m`.
    """
    return np.asarray(e, dtype=float) / e_rest(m)


def gamma_e_kin(e: array_like, m: array_like) -> ndarray:
    r"""
    The time-dilation/Lorentz factor $\gamma = 1 + E_\mathrm{kin} / (mc^2)$.

    :param e: The kinetic energy $E_\mathrm{kin}$ of a body (eV).
    :param m: The mass $m$ of the body (u).
    :returns: (gamma) The time-dilation/Lorentz factor $\gamma$ corresponding to the kinetic energy `e` of a body with mass `m`.
    """
    return 1. + gamma_e(e, m)


def e_rest(m: array_like) -> ndarray:
    r"""
    The resting energy $E_\mathrm{rest} = mc^2$.

    :param m: The mass $m$ of a body (u).
    :returns: (E_rest) The resting energy $E_\mathrm{rest}$ of the body with mass `m` (eV).
    """
    return np.asarray(m, dtype=float) * sc.atomic_mass * sc.c ** 2 / E_NORM


def e_kin(v: array_like, m: array_like, relativistic: bool = True) -> ndarray:
    r"""
    The kinetic energy $E_\mathrm{kin} = \begin{cases}(\gamma(v) - 1) mc^2, & \mathrm{True}\\
    \frac{1}{2}mv^2 & \mathrm{False}\end{cases}$.

    :param v: The velocity $v$ of a body (m/s).
    :param m: The mass $m$ of the body (u).
    :param relativistic: The calculation is performed either relativistically (`True`) or classically (`False`).
     The default is `True`.
    :returns: (E_kin) The kinetic energy $E_\mathrm{kin}$ of a body with velocity `v` and mass `m` (eV).
    """
    if relativistic:
        return (gamma(v) - 1.) * e_rest(m)
    else:
        v, m = np.asarray(v, dtype=float), np.asarray(m, dtype=float)
        return 0.5 * m * sc.atomic_mass * v ** 2 / E_NORM


def e_total(v: array_like, m: array_like) -> ndarray:
    r"""
    The total energy $E = \gamma(v)mc^2$.

    :param v: The velocity $v$ of a body (m/s).
    :param m: The mass $m$ of the body (u).
    :returns: (E) The total energy $E$ of a body with velocity `v` and mass `m` (eV).
    """
    return gamma(v) * e_rest(m)


def e_el(u: array_like, q: array_like) -> ndarray:
    r"""
    The potential energy difference $E_\mathrm{pot} = qU$.

    :param u: An electric potential difference $U$ (V).
    :param q: The electric charge $q$ of a body (e).
    :returns: (E_pot) The potential energy difference $E_\mathrm{pot}$ of a body with electric charge `q`
     inside an electric potential with voltage `u` (eV).
    """
    q, u = np.asarray(q, dtype=float), np.asarray(u, dtype=float)
    return q * u


def v_e(e: array_like, m: array_like, v0: array_like = 0, relativistic: bool = True) -> ndarray:
    r"""
    The velocity $v = \begin{cases}c\sqrt{1 - \left[\gamma(v_0) + E / (mc^2)\right]^{-2}}, & \mathrm{True}\\
    \sqrt{v_0^2 + 2E / m}, & \mathrm{False}\end{cases}$

    :param e: The energy $E$ added to the kinetic energy of a body with velocity `v0` (eV).
    :param m: The mass $m$ of the body (u).
    :param v0: The initial velocity $v_0$ of the body (m/s).
    :param relativistic: The calculation is performed either relativistically (`True`) or classically (`False`).
     The default is `True`.
    :returns: (v) The velocity $v$ of a body with mass `m` and velocity `v0`
     after the addition of the energy `e` (m/s).
    """
    if relativistic:
        return sc.c * np.sqrt(1. - (1. / (gamma(v0) + gamma_e(e, m))) ** 2)
    else:
        v0, e, m = np.asarray(v0, dtype=float), np.asarray(e, dtype=float), np.asarray(m, dtype=float)
        return np.sqrt(v0 ** 2 + 2. * e * E_NORM / (m * sc.atomic_mass))


def v_e_d1(e: array_like, m: array_like, v0: array_like = 0, relativistic: bool = True) -> ndarray:
    r"""
    The first derivative $\frac{\partial v}{\partial E} = \frac{1}{amv(E)}$ with
     $a = \begin{cases}(\gamma(v_0) + E / (mc^2))^3, & \mathrm{True}\\1, & \mathrm{False}\end{cases}$

    :param e: The energy $E$ added to the kinetic energy of a body with velocity `v0` (eV).
    :param m: The mass $m$ of the body (u).
    :param v0: The initial velocity $v_0$ of the body (m/s).
    :param relativistic: The calculation is performed either relativistically (`True`) or classically (`False`).
     The default is `True`.
    :returns: (dv_dE) The first derivative $\partial v / \partial E$ of the velocity $v(E)$ of a body with mass `m`
     and velocity `v0` with respect to the added energy `e` (m/(s eV)).
    """
    m = np.asarray(m, dtype=float)
    dv = 1. / (m * sc.atomic_mass * v_e(e, m, v0=v0, relativistic=relativistic))
    if relativistic:
        dv /= (gamma(v0) + gamma_e(e, m)) ** 3
    return dv * E_NORM


def v_el(u: array_like, q: array_like, m: array_like, v0: array_like = 0, relativistic: bool = True) -> ndarray:
    r"""
    The velocity $v = \begin{cases}c\sqrt{1 - \left[\gamma(v_0) + qU / (mc^2)\right]^{-2}}, & \mathrm{True}\\
    \sqrt{v_0^2 + 2qU / m}, & \mathrm{False}\end{cases}$

    :param u: The electric potential difference $U$ added to the kinetic energy of a body with velocity `v0` (V).
    :param q: The electric charge $q$ of the body (e).
    :param m: The mass $m$ of the body (u).
    :param v0: The initial velocity $v_0$ of the body (m/s).
    :param relativistic: The calculation is performed either relativistically (`True`) or classically (`False`).
     The default is `True`.
    :returns: (v) The velocity $v$ of the body with starting velocity `v0`, electric charge `q` and mass `m`
     after electrostatic acceleration with the voltage `u` (m/s).
    """
    return v_e(e_el(u, q), m, v0=v0, relativistic=relativistic)


def v_el_d1(u: array_like, q: array_like, m: array_like, v0: array_like = 0, relativistic: bool = True) -> ndarray:
    r"""
    The first derivative $\frac{\partial v}{\partial E} = \frac{q}{amv(E)}$ with
     $a = \begin{cases}(\gamma(v_0) + qU / (mc^2))^3, & \mathrm{True}\\1, & \mathrm{False}\end{cases}$

    :param u: The electric potential difference $U$ added to the kinetic energy of a body with velocity `v0` (V).
    :param q: The electric charge $q$ of the body (e).
    :param m: The mass $m$ of the body (u).
    :param v0: The initial velocity $v_0$ of the body (m/s).
    :param relativistic: The calculation is performed either relativistically (`True`) or classically (`False`).
     The default is `True`.
    :returns: (dv_dU) The first derivative $\partial v / \partial U$ of the velocity $v$ of the body
     with starting velocity `v0`, electric charge `q` and mass `m` after electrostatic acceleration
     with the voltage `u` (m/(s V)).
    """
    q = np.asarray(q, dtype=float)
    return v_e_d1(e_el(u, q), m, v0=v0, relativistic=relativistic) * q


def p_v(v: array_like, m: array_like, relativistic: bool = True) -> ndarray:
    r"""
    The momentum $p = \gamma(v)mv$.

    :param v: The velocity $v$ of a body (m/s).
    :param m: The mass $m$ of the body (u).
    :param relativistic: The calculation is performed either relativistically (`True`) or classically (`False`).
     The default is `True`.
    :returns: (p) The momentum $p$ of a body with velocity `v` and mass `m` (u m/s).
    """
    v, m = np.asarray(v, dtype=float), np.asarray(m, dtype=float)
    if relativistic:
        return gamma(v) * m * v
    else:
        return m * v


def p_e(e: array_like, m: array_like, p0: array_like = 0, relativistic: bool = True) -> ndarray:
    r"""
    The momentum $p = \frac{1}{c}\sqrt{E^2 + (p_0 c)^2 + 2E\sqrt{(p_0 c)^2 + (mc^2)^2}}$.

    :param e: The energy $E$ added to the kinetic energy of a body with momentum p0 (eV).
    :param m: The mass $m$ of the body (u).
    :param p0: The initial momentum of the body (u m/s).
    :param relativistic: The calculation is performed either relativistically (`True`) or classically (`False`).
     The default is `True`.
    :returns: (p) The momentum $p$ of a body with mass `m` and starting momentum `p0`
     after the addition of the energy `e` (u m/s).
    """
    e, p0 = np.asarray(e, dtype=float), np.asarray(p0, dtype=float)

    if relativistic:
        pc_square = (p0 * sc.atomic_mass * sc.c) ** 2 / E_NORM ** 2
        return np.sqrt(e ** 2 + pc_square + 2 * e * np.sqrt(pc_square + e_rest(m) ** 2)) / (sc.c * sc.atomic_mass)

    else:
        m = np.asarray(m, dtype=float)
        return np.sqrt((p0 * sc.atomic_mass) ** 2 + 2 * m * sc.atomic_mass * e * E_NORM) / sc.atomic_mass


def p_el(u: array_like, q: array_like, m: array_like, p0: array_like = 0, relativistic: bool = True) -> ndarray:
    r"""
    The momentum $p = \frac{1}{c}\sqrt{(qU)^2 + (p_0 c)^2 + 2qU\sqrt{(p_0 c)^2 + (mc^2)^2}}$.

    :param u: The electric potential difference $U$ added to the kinetic energy of a body with momentum `p0` (V).
    :param q: The electric charge $q$ of the body (e).
    :param m: The mass $m$ of the body (u).
    :param p0: The initial momentum of the body (u m/s).
    :param relativistic: The calculation is performed either relativistically (`True`) or classically (`False`).
     The default is `True`.
    :returns: (p) The momentum $p$ of a body with electric charge `q`, mass `m` and starting momentum `p0`
     after the addition of the energy `e` (u m/s).
    """
    return p_e(e_el(u, q), m, p0, relativistic=relativistic)


def doppler(f: array_like, v: array_like, alpha: array_like, return_frame: str = 'atom') -> ndarray:
    r"""
    The Doppler-shifted frequency $f^\prime = \begin{cases}f\gamma(v)(1 - \frac{v}{c}\cos(\alpha)), & \mathrm{atom}\\
    f[\gamma(v)(1 - \frac{v}{c}\cos(\alpha))]^{-1}, & \mathrm{lab}\end{cases}$

    :param f: The frequency $f$ of light (arb. units).
    :param v: The velocity $v$ of a body (m/s).
    :param alpha: The angle $\alpha$ between the velocity- and the light-vector in the laboratory frame (rad).
    :param return_frame: The coordinate system for which the frequency is returned. Can be either `'atom'` or `'lab'`.
    :returns: (f_doppler) the Doppler-shifted frequency $f^\prime$ in either the rest frame of the atom
     or the laboratory frame ([`f`]).
    :raises ValueError: `return_frame` must be either `'atom'` or `'lab'`.
    """
    f, alpha = np.asarray(f, dtype=float), np.asarray(alpha, dtype=float)

    if return_frame == 'atom':
        # Return freq in the atomic system, alpha=0 == Col, alpha in laboratory system
        return f * gamma(v) * (1. - beta(v) * np.cos(alpha))

    elif return_frame == 'lab':
        # Return freq in the laboratory system, alpha=0 == Col, alpha in laboratory system
        return f / (gamma(v) * (1. - beta(v) * np.cos(alpha)))

    else:
        raise ValueError('return_frame must be either \'atom\' or \'lab\'.')


def doppler_d1(f: array_like, v: array_like, alpha: array_like, return_frame: str = 'atom') -> ndarray:
    r"""
    The first derivative $\frac{\partial f^\prime}{\partial v} = a\frac{f^\prime}{c}\gamma^3(v)(\frac{v}{c}
     - \cos(\alpha))$ with $a = \begin{cases}f / f^\prime, & \mathrm{atom}\\
     -f^\prime / f, & \mathrm{lab}\end{cases}$

    :param f: The frequency $f$ of light (arb. units).
    :param v: The velocity $v$ of a body (m/s).
    :param alpha: The angle $\alpha$ between the velocity- and the light-vector in the laboratory frame (rad).
    :param return_frame: The coordinate system for which the frequency is returned. Can be either `'atom'` or `'lab'`.
    :returns: (df_dv) the first derivative $\partial f^\prime / \partial v$ of the Doppler-shifted frequency
     $f^\prime$ with respect to `v` in either the rest frame of the atom or the laboratory frame ([`f`] s/m).
    :raises ValueError: `return_frame` must be either `'atom'` or `'lab'`.
    """
    f = np.asarray(f, dtype=float)

    if return_frame == 'atom':
        # Return df/dv in the atomic system, alpha=0 == Col, alpha in laboratory system.
        return f * gamma(v) ** 3 * (beta(v) - np.cos(alpha)) / sc.c

    elif return_frame == 'lab':
        # Return df/dv in the laboratory system, alpha=0 == Col, alpha in laboratory system.
        f_lab = doppler(f, v, alpha, return_frame='lab')
        return -f_lab / f * doppler_d1(f_lab, v, alpha, return_frame='atom')

    else:
        raise ValueError('return_frame must be either \'atom\' or \'lab\'.')


def doppler_e_d1(f: array_like, alpha: array_like, e: array_like, m: array_like,
                 v0: array_like = 0, return_frame: str = 'atom', relativistic: bool = True) -> ndarray:
    r"""
    The first derivative $\frac{\partial f^\prime}{\partial E} =
     \frac{\partial f^\prime}{\partial v}\frac{\partial v}{\partial E}$.
    Implemented as `doppler_d1(f, v, alpha, return_frame) * v_e_d1(e, m, v0, relativistic)`.

    :param f: The frequency $f$ of light (arb. units).
    :param alpha: The angle $\alpha$ between the velocity- and the light-vector in the laboratory frame (rad).
    :param e: The energy $E$ added to the kinetic energy of a body with velocity `v0` (eV).
    :param m: The mass $m$ of the body (u).
    :param v0: The initial velocity $v_0$ of the body (m/s).
    :param return_frame: The coordinate system for which the frequency is returned. Can be either `'atom'` or `'lab'`.
    :param relativistic: The calculation is performed either relativistically (`True`) or classically (`False`).
     The default is `True`.
    :returns: (df_dE) The first derivative $\partial f^\prime / \partial E$ of the Doppler-shifted frequency
     $f^\prime$ with respect to `e` in either the rest frame of the atom or the laboratory frame ([`f`] / eV).
    :raises ValueError: `return_frame` must be either `'atom'` or `'lab'`.
    """
    v = v_e(e, m, v0=v0, relativistic=relativistic)
    return doppler_d1(f, v, alpha, return_frame=return_frame) * v_e_d1(e, m, v0=v0, relativistic=relativistic)


def doppler_el_d1(f: array_like, alpha: array_like, u: array_like, q: array_like, m: array_like,
                  v0: array_like = 0., return_frame: str = 'atom', relativistic: bool = True) -> ndarray:
    r"""
    The first derivative $\frac{\partial f^\prime}{\partial U} =
     \frac{\partial f^\prime}{\partial v}\frac{\partial v}{\partial U}$.
    Implemented as `doppler_d1(f, v, alpha, return_frame) * v_el_d1(u, q, m, v0, relativistic)`.

    :param f: The frequency $f$ of light (arb. units).
    :param alpha: The angle $\alpha$ between the velocity- and the light-vector in the laboratory frame (rad).
    :param u: The electric potential difference $U$ added to the kinetic energy of a body with velocity `v0` (V).
    :param q: The electric charge $q$ of the body (e).
    :param m: The mass $m$ of the body (u).
    :param v0: The initial velocity $v_0$ of the body (m/s).
    :param return_frame: The coordinate system for which the frequency is returned. Can be either `'atom'` or `'lab'`.
    :param relativistic: The calculation is performed either relativistically (`True`) or classically (`False`).
     The default is `True`.
    :returns: (df_dU) The first derivative $\partial f^\prime / \partial U$ of the Doppler-shifted frequency
     $f^\prime$ with respect to `u` in either the rest frame of the atom or the laboratory frame ([`f`] / V).
    :raises ValueError: `return_frame` must be either `'atom'` or `'lab'`.
    """
    v = v_el(u, q, m, v0=v0, relativistic=relativistic)
    return doppler_d1(f, v, alpha, return_frame=return_frame) * v_el_d1(u, q, m, v0=v0, relativistic=relativistic)


def inverse_doppler(f_atom: array_like, f_lab: array_like, alpha: array_like,
                    mode: str = 'raise-raise', return_mask: bool = False) -> (ndarray, Optional[ndarray]):
    r"""
    The velocity
    $$\begin{aligned}v &= \frac{c}{s}\left[\cos(\alpha) \pm (f_\mathrm{atom}/f_\mathrm{lab})\sqrt{s - 1}\right]\\[2ex]
      s &= (f_\mathrm{atom}/f_\mathrm{lab})^2 + \cos(\alpha)^2\end{aligned}.$$
    For angles $-\pi/2 < \alpha < \pi/2$, there can be two solutions.
    Depending on the combination of `f_atom`, `f_lab` and `alpha`, the situation may be physically impossible.
    Specify `mode` to choose the desired behavior.

    :param f_atom: The frequency of light $f_\mathrm{atom}$ in the atom's rest frame (arb. units).
    :param f_lab: The frequency of light $f_\mathrm{lab}$ in the laboratory frame ([`f_atom`]).
    :param alpha: The angle $\alpha$ between the velocity- and the light-vector in the laboratory frame (rad).
    :param mode: The mode how to handle `nan` values and ambiguous velocities. Available options are:
    <ul>
    <li> `'raise-raise'`: Raise an error if there are `nan` values or if the velocity is ambiguous.</li>
    <li> `'raise-small'`: Raise an error if there are `nan` values and return the smaller velocity.</li>
    <li> `'raise-large'`: Raise an error if there are `nan` values and return the larger velocity.</li>
    <li> `'isnan-raise'`: Ignore `nan` values and raise an error if the velocity is ambiguous.</li>
    <li> `'isnan-small'`: Ignore `nan` values and return the smaller velocity.</li>
    <li> `'isnan-large'`: Ignore `nan` values and return the larger velocity.</li>
    </ul>
    :param return_mask: Whether the mask where the velocity is ambiguous is returned as a second argument.
    :returns: (v) The velocity $v$ required to shift `f_lab` to `f_atom`.
     Optionally returns the mask where the velocity is ambiguous (m/s).
    :raises ValueError: `mode` must be `'raise-raise'`, `'raise-small'`, `'raise-large'`, `'isnan-raise'`,
     `'isnan-small'`, or `'isnan-large'`. For additionally raised errors, see the description of the `mode` parameter.
    """
    modes = {'raise-raise', 'raise-small', 'raise-large', 'isnan-raise', 'isnan-small', 'isnan-large'}
    if mode not in modes:
        raise ValueError('mode must be in {}.'.format(modes))

    f_atom, f_lab, alpha = \
        np.asarray(f_atom, dtype=float), np.asarray(f_lab, dtype=float), np.asarray(alpha, dtype=float)
    scalar_true = tools.check_shape((), f_atom, f_lab, alpha, return_mode=True)
    if scalar_true:  # To make array masking work, scalars need to be converted to 1d-arrays.
        alpha = np.array([alpha])

    cos = np.cos(alpha)
    square_sum = (f_atom / f_lab) ** 2 + cos ** 2
    nan = square_sum < 1.
    np.seterr(invalid='ignore')
    bet1 = (cos + f_atom / f_lab * np.sqrt(square_sum - 1.)) / square_sum
    bet2 = (cos - f_atom / f_lab * np.sqrt(square_sum - 1.)) / square_sum
    np.seterr(invalid='warn')

    bet1[nan] = 2.
    bet2[nan] = 2.
    mask1 = np.abs(0.5 - bet1) < 0.5
    mask1 += bet1 == 0.
    mask2 = np.abs(0.5 - bet2) < 0.5
    mask2 += bet2 == 0.
    ambiguous = ~(~mask1 + ~mask2)
    nan = ~(mask1 + mask2)
    bet = np.zeros_like(square_sum)
    bet[nan] = np.nan

    if mode[6:] in ['small', 'raise']:
        bet[mask1] = bet1[mask1]
        bet[mask2] = bet2[mask2]
    elif mode[6:] == 'large':
        bet[mask2] = bet2[mask2]
        bet[mask1] = bet1[mask1]
    if np.any(nan):
        if mode[:5] == 'raise':
            raise ValueError('Situation is physically impossible for at least one argument.')
    if np.any(ambiguous):
        if mode[6:] == 'raise':
            raise ValueError('Situation allows two different velocities.')

    if return_mask:
        if scalar_true:
            return (bet * sc.c)[0], ambiguous
        return bet * sc.c, ambiguous
    if scalar_true:
        return (bet * sc.c)[0]
    return bet * sc.c


def inverse_doppler_d1(f_atom: array_like, f_lab: array_like, alpha: array_like,
                       mode: str = 'raise-raise', return_mask: bool = False) -> (ndarray, Optional[ndarray]):
    r"""
    The first derivative
    $$\begin{aligned}\frac{\partial v}{\partial f_\mathrm{atom}} &=
      \frac{1}{sf_\mathrm{lab}}\left[\pm\left(\sqrt{s - 1}
      + \frac{(f_\mathrm{atom}/f_\mathrm{lab})^2}{\sqrt{s - 1}}\right)
      - 2\frac{v}{c}(f_\mathrm{atom}/f_\mathrm{lab})\right]\\[2ex]
      s &= (f_\mathrm{atom}/f_\mathrm{lab})^2 + \cos(\alpha)^2\end{aligned}.$$
    For angles $-\pi/2 < \alpha < \pi/2$, there can be two solutions.
    Depending on the combination of `f_atom`, `f_lab` and `alpha`, the situation may be physically impossible.
    Specify `mode` to choose the desired behavior.

    :param f_atom: The frequency of light $f_\mathrm{atom}$ in the atom's rest frame (arb. units).
    :param f_lab: The frequency of light $f_\mathrm{lab}$ in the laboratory frame ([`f_atom`]).
    :param alpha: The angle $\alpha$ between the velocity- and the light-vector in the laboratory frame (rad).
    :param mode: The mode how to handle `nan` values and ambiguous velocities. Available options are:
    <ul>
    <li>`'raise-raise'`: Raise an error if there are `nan` values or if the velocity is ambiguous.</li>
    <li>`'raise-small'`: Raise an error if there are `nan` values and return the smaller velocity.</li>
    <li>`'raise-large'`: Raise an error if there are `nan` values and return the larger velocity.</li>
    <li>`'isnan-raise'`: Ignore `nan` values and raise an error if the velocity is ambiguous.</li>
    <li>`'isnan-small'`: Ignore `nan` values and return the smaller velocity.</li>
    <li>`'isnan-large'`: Ignore `nan` values and return the larger velocity.</li>
    </ul>
    :param return_mask: Whether the mask where the velocity is ambiguous is returned as a second argument.
    :returns: (dv_df) The first derivative $\partial v / \partial f_\mathrm{atom}$ of the velocity $v$ required to shift
     `f_lab` to `f_atom`. Optionally returns the mask where the velocity is ambiguous (m/(s MHz)).
    """
    modes = ['raise-raise', 'raise-small', 'raise-large', 'isnan-raise', 'isnan-small', 'isnan-large']
    if mode not in modes:
        raise ValueError('mode must be in {}.'.format(modes))

    f_atom, f_lab, alpha = \
        np.asarray(f_atom, dtype=float), np.asarray(f_lab, dtype=float), np.asarray(alpha, dtype=float)
    scalar_true = tools.check_shape((), f_atom, f_lab, alpha, return_mode=True)
    if scalar_true:  # To make array masking work, scalars need to be converted to 1d-arrays.
        alpha = np.array([alpha])

    v, ambiguous = inverse_doppler(f_atom, f_lab, alpha, mode=mode, return_mask=True)
    cos = np.cos(alpha)
    square_sum = (f_atom / f_lab) ** 2 + cos ** 2
    np.seterr(invalid='ignore')
    bet = np.sqrt(square_sum - 1.) + (f_atom / f_lab) ** 2 / np.sqrt(square_sum - 1.)
    np.seterr(invalid='warn')
    if mode[6:] in ['small', 'raise']:
        bet[ambiguous] = -bet[ambiguous]
    bet += -2. * v * (f_atom / f_lab) / sc.c
    bet /= f_lab * square_sum

    if return_mask:
        if scalar_true:
            return (bet * sc.c)[0], ambiguous
        return bet * sc.c, ambiguous
    if scalar_true:
        return (bet * sc.c)[0]
    return bet * sc.c


def alpha_atom(alpha: array_like, v: array_like) -> ndarray:
    r"""
    The angle in the rest frame of the atom
     $\alpha^\prime = \arccos\left[\frac{(v/c) + \cos(\alpha)}{1 + (v/c)\cos(\alpha)}\right]$.

    :param alpha: The angle $\alpha$ between a velocity- and a wave-vector in the laboratory frame (rad).
    :param v: The velocity $v$ of a body (m/s).
    :returns: (alpha) The angle $\alpha^\prime$ between the velocity-
     and the wave-vector in the atom's rest frame (rad).
    """
    alpha = np.asarray(alpha, dtype=float)
    cos = np.cos(alpha)
    arg = (beta(v) + cos) / (1. + beta(v) * cos)
    return np.arccos(arg)


def v_recoil(f: array_like, m: array_like) -> ndarray:
    r"""
    The change of velocity of an atom at rest $\delta v = hf / (mc)$
     due to the absorption of a photon with frequency $f$.

    :param f: The frequency of light $f$ in the atom's rest frame (MHz).
    :param m: The mass $m$ of the atom (u).
    :returns: (dv_rec) The change of velocity  $\delta v$ (m/s).
    """
    f, m = np.asarray(f, dtype=float), np.asarray(m, dtype=float)
    return sc.h * f / (m * sc.atomic_mass * sc.c) * 1e6


def f_recoil(f: array_like, m: array_like) -> ndarray:
    r"""
    The change of a transition frequency of an atom at rest $\delta f = hf^2 / (2mc^2)$
     due to the absorption of a photon with frequency $f$.

    :param f: The frequency of light in the atoms rest frame (MHz).
    :param m: The mass $m$ of the atom (u).
    :returns: (df_rec) The change of the transition frequency $\delta f$ (MHz).
    """
    f, m = np.asarray(f, dtype=float), np.asarray(m, dtype=float)
    return (sc.h * (f * 1e6) ** 2) / (2 * m * sc.atomic_mass * sc.c ** 2) * 1e-6


def f_recoil_v(v: array_like, alpha: array_like, f_lab: array_like, m: array_like) -> ndarray:
    r"""
    The change of a transition frequency of an atom moving with velocity $v$ (in the direction of its velocity vector)
    due to the absorption of a laser photon with frequency $f$.
    Implemented as `df = f_recoil(doppler(f_lab, v, alpha, return_frame='atom'), m)`

    :param v: The velocity $v$ of the atom (m/s).
    :param alpha: The angle $\alpha$ between a velocity- and a wave-vector in the laboratory frame (rad).
    :param f_lab: The frequency of light $f$ in the laboratory frame (MHz).
    :param m: The mass $m$ of the atom (u).
    :returns: (df_rec) The change of the transition frequency $\delta f$ (MHz).
    """
    f = doppler(f_lab, v, alpha, return_frame='atom')
    return f_recoil(f, m)


""" Atomic physics """


def get_f(i: quant_like, j: quant_like) -> list[quant]:
    r"""
    All quantum numbers fulfilling $|I - J| \leq F \leq I + J$, where $F\in\mathbb{N}_0$.

    :param i: The nuclear spin quantum number $I$.
    :param j: The electronic total angular momentum quantum number $J$.
    :returns: (F_list) All possible $F$ quantum numbers.
    """
    return [quant(k + abs(i - j)) for k in range(int(i + j - abs(i - j) + 1))]


def get_m(f: quant_like) -> list[quant]:
    r"""
    All quantum numbers fulfilling $-F \leq m \leq F$, where $m\in\mathbb{Z}$.

    :param f: The total angular momentum quantum number $F$.
    :returns: (m_list) All possible magnetic quantum numbers $m$ of the specified quantum number $F$.
    """
    return [quant(k - f) for k in range(int(2 * f + 1))]


def lande_n(gyro: array_like) -> ndarray:
    r"""
    The nuclear g-factor $g_I = \gamma_I h / \mu_N$ calculated from the gyromagnetic ratio `gyro`.

    :param gyro: The gyromagnetic ratio $\gamma_I$ (MHz).
    :returns: (g_I) The nuclear g-factor $g_I$.
    """
    gyro = np.asarray(gyro, dtype=float)
    return gyro * sc.h / mu_N


def lande_j(s: quant_like, l: quant_like, j: quant_like, approx_g_s: bool = False) -> float:
    r"""
    The electronic g-factor in the LS-coupling scheme

    $$
    g_J = -\frac{J(J + 1) + L(L + 1) - S(S + 1)}{2J(J + 1)} + g_s\,\frac{J(J + 1) - L(L + 1) + S(S + 1)}{2J(J + 1)}.
    $$

    Note that in this definition the negative charge of the electron is included in the g-factor,
    such that $g_s$ is negative.

    :param s: The electron spin quantum number $S$.
    :param l: The electronic angular momentum quantum number $L$.
    :param j: The electronic total angular momentum quantum number $J$.
    :param approx_g_s: Whether to use g_s = -2 (`True`) or the QED result g_s = -2.0023... (`False`).
     The default is `False`.
    :returns: (g_J) The electronic g-factor $g_J$.
    """
    if j == 0:
        return 0.
    g = -2 if approx_g_s else g_s
    jj = j * (j + 1)
    ls = l * (l + 1) - s * (s + 1)
    val = -(jj + ls) / (2 * jj)
    val += (jj - ls) / (2 * jj) * g
    return val


def lande_jj(j0: quant_like, j1: quant_like, j: quant_like, g0: array_like, g1: array_like) -> float:
    r"""
    The electronic g-factor of a state with angular momentum $\vec{J} = \vec{J}_0 + \vec{J}_1$

    $$
    g_J = g_0\frac{J(J + 1) + J_0(J_0 + 1) - J_1(J_1 + 1)}{2J(J + 1)} + g_1\frac{J(J + 1) - J_0(J_0 + 1) + J_1(J_1 + 1)}{2J(J + 1)}.
    $$

    :param j0: The angular momentum quantum number $J_0$.
    :param j1: The angular momentum quantum number $J_1$.
    :param j: The angular momentum quantum number $J$.
    :param g0: The g-factor $g_{J_0}$.
    :param g1: The g-factor $g_{J_1}$.
    :returns: (g_J) The g-factor $g_J$.
    """
    jj = j * (j + 1)
    jj01 = j0 * (j0 + 1) - j1 * (j1 + 1)
    return 0.5 * g0 * (jj + jj01) / jj + 0.5 * g1 * (jj - jj01) / jj


def g_j(j: quant_like = 0, ls: quant_iter = None, jj: quant_iter = None, gj: array_like = None) -> float:
    r"""
    The electronic g-factor of a state with angular momentum $\vec{J}$ in the LS- or jj-coupling scheme.
    See `lande_j` and `lande_jj`.

    :param j: The angular momentum quantum number $J$.
    :param ls: A list or a single pair of electronic angular momentum and spin quantum numbers $(l_i, s_i)$
     used to calculate the electronic g-factor in the LS-coupling scheme. If this is a list of LS-pairs,
     A list of $j_i$ quantum numbers needs to specified for the parameter `jj`. It is overwritten if `gj` is specified.
    :param jj: A list of two electronic total angular momentum quantum numbers $(j_0, j_1)$
     used to calculate the electronic g-factor in the jj-coupling scheme.
     Either a list of two $(l_i, s_i)$ pairs needs to be specified for the parameter `ls`
     or a list of g-factors $g_{j_i}$ for the parameter `gj`. The parameter `gj` overwrites `ls`.
     If `gj` is a single scalar value, it also overwrites `jj`.
    :param gj: A list of two $g_{j_i}$ or a single electronic g-factor $g_J$. If `gj` is a list, `jj` is required
     and `ls` is overwritten. If `gj` is a scalar, both `ls` and `jj` are overwritten.
    :returns: (g_J) The g-factor $g_J$.
    """
    if ls is None and gj is None:
        raise ValueError('The parameter \'ls\' needs to be specified if \'gj\' is not given.')
    if gj is None:
        if jj is None:
            gj = lande_j(ls[1], ls[0], j)
        else:
            g0 = lande_j(ls[0][1], ls[0][0], j)
            g1 = lande_j(ls[1][1], ls[1][0], j)
            gj = lande_jj(jj[0], jj[1], j, g0, g1)
    elif hasattr(gj, '__getitem__'):
        gj = lande_jj(jj[0], jj[1], j, gj[0], gj[1])
    return gj


def lande_f(i: quant_like, j: quant_like, f: quant_like, gi: array_like, gj: array_like) -> ndarray:
    r"""
    The total atomic g-factor in the IJ-coupling scheme

    $$
    g_F = g_J\,\frac{F(F + 1) + J(J + 1) - I(I + 1)}{2F(F + 1)}
     + g_I\,\frac{\mu_\mathrm{N}}{\mu_\mathrm{B}} \frac{F(F + 1) - J(J + 1) + I(I + 1)}{2F(F + 1)}.
    $$

    Note that in this definition the electric charges are included in the g-factors,
    such that $g_s$, the g-factor of the electron, must be negative
    and $g_\mathrm{p}$, the g-factor of the proton, must be positive.

    :param i: The nuclear spin quantum number $I$.
    :param j: The electronic total angular momentum quantum number $J$.
    :param f: The total angular momentum quantum number $F$.
    :param gi: The nuclear g-factor $g_I$.
    :param gj: The electronic g-factor $g_J$.
    :returns: (g_F) The total atomic g-factor $g_F$.
    """
    gi, gj = np.asarray(gi, dtype=float), np.asarray(gj, dtype=float)
    ff = f * (f + 1.)
    ji = j * (j + 1.) - i * (i + 1.)
    val = (ff + ji) / (2 * ff) * gj
    val += (ff - ji) / (2 * ff) * gi * mu_N / mu_B
    return val


def hyperfine(i: quant_like, j: quant_like, f: quant_like,
              a_hyper: array_like = 0., b_hyper: array_like = 0., c_hyper: array_like = 0.) -> ndarray:
    r"""
    The hyperfine structure shift of an atomic state `(i, j, f)` with the hyperfine constants `a` and `b` and `c`

    $$\begin{aligned}
    \Delta_\mathrm{hfs} &= A\frac{K}{2} + B\frac{\frac{3}{4}K(K + 1) - I(I + 1)J(J + 1)}{2I(2I - 1)J(2J - 1)}\\[1ex]
    &\quad + C\frac{\left[\splitdfrac{\frac{5}{4}K^3 + 5K^2 - 5I(I + 1)J(J + 1)}
    {+ K(I(I + 1) + J(J + 1) - 3I(I + 1)J(J + 1) + 3)}\right]}{I(I - 1)(2I - 1)J(J - 1)(2J - 1)}\\[3ex]
    K &= F(F + 1) - I(I + 1) - J(J + 1)
    \end{aligned}$$

    :param i: The nuclear spin quantum number $I$.
    :param j: The electronic total angular momentum quantum number $J$.
    :param f: The total angular momentum quantum number $F$.
    :param a_hyper: The magnetic dipole hyperfine constant $A = \mu_I \mathcal{B}_J / (IJ)$ (arb. units).
    :param b_hyper: The electric quadrupole hyperfine constant $B = eQ_I (\partial^2 V_J / \partial z^2)$ ([`a_hyper`]).
    :param c_hyper: The magnetic octupole hyperfine constant $C = \Omega_I T_J^{(3)}$ ([`a_hyper`])
    :returns: (dnu_hfs) The hyperfine structure shift $\Delta_\mathrm{hfs}$ ([`a_hyper`]).
    """
    a_hyper = np.asarray(a_hyper, dtype=float)

    if i < 0. or j < 0. or f < 0.:
        raise ValueError('All quantum numbers must be >= 0.')
    if f < abs(i - j) or f > i + j:
        raise ValueError('f does not fulfill |i - j| <= f <= i + j.')

    if i == 0. or j == 0.:
        return np.zeros_like(a_hyper)

    k = f * (f + 1) - i * (i + 1) - j * (j + 1)
    shift = 0.5 * a_hyper * k

    if i > 0.5 and j > 0.5:
        b_hyper = np.asarray(b_hyper, dtype=float)
        k_2 = 3 * k * (k + 1) - 4 * i * (i + 1) * j * (j + 1)
        k_2 /= 2 * i * (2 * i - 1) * j * (2 * j - 1)
        shift += 0.25 * b_hyper * k_2

    if i > 1 and j > 1:
        c_hyper = np.asarray(c_hyper, dtype=float)
        k_3 = k ** 3 + 4 * k ** 2 + 0.8 * k * (-3 * i * (i + 1) * j * (j + 1) + i * (i + 1) + j * (j + 1) + 3) \
            - 4 * i * (i + 1) * j * (j + 1)
        k_3 /= i * (i - 1) * (2 * i - 1) * j * (j - 1) * (2 * j - 1)
        shift += 1.25 * c_hyper * k_3

    return shift


def zeeman_linear(m: quant_like, g: array_like, b_field: array_like = 0., as_freq: bool = True) -> ndarray:
    r"""
    The shift of an atomic state with magnetic quantum number `m` due to the linear Zeeman effect
     $\Delta_\mathrm{Zeeman} = -gm\mu_\mathrm{B}\mathcal{B}$

    :param m: The magnetic quantum number $m$.
    :param g: The g-factor $g$.
    :param b_field: The B-field $\mathcal{B}$ (T).
    :param as_freq: The shift can be returned in energy (`False`, eV) or frequency units (`True`, MHz).
     The default is `True`
    :returns: (dnu_zeeman) The linear Zeeman shift $\Delta_\mathrm{Zeeman}$
     in energy or frequency units (MHz if `as_freq` else eV).
    """
    g, b_field = np.asarray(g, dtype=float), np.asarray(b_field, dtype=float)

    z_unit = 1e-6 / sc.h if as_freq else 1 / E_NORM
    return -g * m * mu_B * b_field * z_unit


def hyper_zeeman_linear(i: quant_like, j: quant_like, f: quant_like, m: quant_like,
                        a_hyper: array_like = 0., b_hyper: array_like = 0., c_hyper: array_like = 0.,
                        g_f: array_like = 0., b_field: array_like = 0., as_freq: bool = True) -> ndarray:
    r"""
    The total energy shift of an atomic state with quantum numbers `F` and `m` due to the hyperfine structure splitting
     and the linear Zeeman effect $\Delta = \Delta_\mathrm{hfs} + \Delta_\mathrm{Zeeman}$.

    :param i: The nuclear spin quantum number $I$.
    :param j: The electronic total angular momentum quantum number $J$.
    :param f: The total angular momentum quantum number $F$.
    :param m: The magnetic quantum number $m_F$.
    :param a_hyper: The magnetic dipole hyperfine constant $A = \mu_I \mathcal{B}_J / (IJ)$ (MHz if `as_freq` else eV).
    :param b_hyper: The electric quadrupole hyperfine constant $B = eQ_I (\partial^2 V_J / \partial z^2)$ ([`a_hyper`]).
    :param c_hyper: The magnetic octupole hyperfine constant $C = \Omega_I T_J^{(3)}$ ([`a_hyper`]).
    :param g_f: The atomic g-factor $g_F$.
    :param b_field: The B-field $\mathcal{B}$ (T).
    :param as_freq: The shift can be returned in energy (`False`, eV) or frequency units (`True`, MHz).
     The default is `True`
    :returns: (dnu_hfs_zeeman) The hyperfine structure + linear Zeeman shift $\Delta$ (MHz if `as_freq` else eV)
    """
    return hyperfine(i, j, f, a_hyper, b_hyper, c_hyper) + zeeman_linear(m, g_f, b_field, as_freq=as_freq)


def hyper_zeeman_ij(mi0: quant_like, mj0: quant_like, mi1: quant_like, mj1: quant_like, i: quant_like, j: quant_like,
                    a_hyper: array_like = 0., b_hyper: array_like = 0.,
                    gi: array_like = 0., gj: array_like = 0., b_field: array_like = 0.,
                    as_freq: bool = True) -> ndarray:
    r"""
    The matrix element $\langle m_{i, 0} m_{j, 0}| H_\mathrm{hfs} + H_\mathrm{Zeeman} |m_{i, 1} m_{j, 1}\rangle$.

    :param mi0: The first magnetic quantum number $m_{i, 0}$ of the nuclear spin $I$.
    :param mj0: The first magnetic quantum number $m_{j, 0}$ of the total electronic angular momentum $J$.
    :param mi1: The second magnetic quantum number $m_{i, 1}$ of the nuclear spin $I$.
    :param mj1: The second magnetic quantum number $m_{j, 1}$ of the total electronic angular momentum $J$.
    :param i: The nuclear spin quantum number $I$.
    :param j: The electronic total angular momentum quantum number $J$.
    :param a_hyper: The magnetic dipole hyperfine constant $A = \mu_I \mathcal{B}_J / (IJ)$ (MHz if `as_freq` else eV).
    :param b_hyper: The electric quadrupole hyperfine constant $B = eQ_I (\partial^2 V_J / \partial z^2)$ ([`a_hyper`]).
    :param gi: The nuclear g-factor $g_I$ or the gyromagnetic ratio $\gamma_I$ if `g_n_as_gyro == True`.
    :param gj: The electronic g-factor $g_J$.
    :param b_field: The B-field $\mathcal{B}$ (T).
    :param as_freq: The matrix element can be returned in energy (`False`, eV) or frequency units (`True`, MHz).
     The default is `True`
    :returns: (h_ij) One matrix element of the hyperfine-structure + Zeeman-effect hamiltonian.
    """
    a_hyper, b_hyper = np.asarray(a_hyper, dtype=float), np.asarray(b_hyper, dtype=float)
    gi, gj = np.asarray(gi, dtype=float), np.asarray(gj, dtype=float)

    z_unit = 1e-6 / sc.h if as_freq else 1 / E_NORM
    b_field = (np.asarray(b_field, dtype=float) + np.zeros_like(a_hyper) + np.zeros_like(b_hyper)
               + np.zeros_like(gi) + np.zeros_like(gj)) * z_unit

    b_hyper_n = b_hyper / (2 * i * (2 * i - 1) * j * (2 * j - 1)) if i > 0.5 and j > 0.5 else 0.

    if mi0 + mj0 != mi1 + mj1:
        return np.zeros_like(b_field, dtype=float)

    elif mi0 == mi1 and mj0 == mj1:
        ret = a_hyper * mi0 * mj0 - (mi0 * gi * mu_N + mj0 * gj * mu_B) * b_field
        ret += b_hyper_n * (3 * (mi0 * mj0) ** 2 - i * (i + 1) * j * (j + 1) + 1.5 * mi0 * mj0
                            + 0.75 * (j - mj0) * (j + mj0 + 1) * (i + mi0) * (i - mi0 + 1)
                            + 0.75 * (i - mi0) * (i + mi0 + 1) * (j + mj0) * (j - mj0 + 1))
        return ret

    elif mi0 == mi1 + 1 and mj0 == mj1 - 1:
        return np.full_like(b_field, (0.5 * a_hyper + 1.5 * b_hyper_n * (0.5 + mi0 * mj0 + mi1 * mj1))
                            * np.sqrt((i - mi1) * (i + mi1 + 1) * (j + mj1) * (j - mj1 + 1)))

    elif mi0 == mi1 - 1 and mj0 == mj1 + 1:
        return np.full_like(b_field, (0.5 * a_hyper + 1.5 * b_hyper_n * (0.5 + mi0 * mj0 + mi1 * mj1))
                            * np.sqrt((i + mi1) * (i - mi1 + 1) * (j - mj1) * (j + mj1 + 1)))

    elif mi0 == mi1 + 2 and mj0 == mj1 - 2:
        return np.full_like(b_field,
                            0.75 * b_hyper_n * np.sqrt((j + mj1) * (j - mj1 + 1) * (j + mj1 - 1) * (j - mj1 + 2)
                                                       * (i - mi1) * (i + mi1 + 1) * (i - mi1 - 1) * (i + mi1 + 2)))

    elif mi0 == mi1 - 2 and mj0 == mj1 + 2:
        return np.full_like(b_field,
                            0.75 * b_hyper_n * np.sqrt((i + mi1) * (i - mi1 + 1) * (i + mi1 - 1) * (i - mi1 + 2)
                                                       * (j - mj1) * (j + mj1 + 1) * (j - mj1 - 1) * (j + mj1 + 2)))

    return np.zeros_like(b_field, dtype=float)


def hyper_zeeman_num(i: quant_like, j: quant_like, a_hyper: array_like = 0., b_hyper: array_like = 0.,
                     gi: array_like = 0., gj: array_like = 0., b_field: array_like = 0.,
                     g_n_as_gyro: bool = False, as_freq: bool = True) \
        -> (list[ndarray], list[quant], list[list[quant]], list[list[tuple[quant, quant]]]):
    r"""
    The shifted energies/frequencies of the hyperfine structure states generated by the quantum numbers $I$ and $J$.
    This function numerically calculates the full diagonalization of the Hyperfine-structure + Zeeman-effect Hamiltonian

    $$
    H = \sum\limits_{k=1}^{2} \vec{T}_I^{(k)}\cdot\vec{T}_J^{(k)} - \vec{\mu}_F\cdot\vec{\mathcal{B}}.
    $$

    :param i: The nuclear spin quantum number $I$.
    :param j: The electronic total angular momentum quantum number $J$.
    :param gi: The nuclear g-factor $g_I$ or the gyromagnetic ratio $\gamma_I$ if `g_n_as_gyro == True`.
    :param gj: The electronic g-factor $g_J$.
    :param a_hyper: The magnetic dipole hyperfine constant $A = \mu_I \mathcal{B}_J / (IJ)$ (MHz if `as_freq` else eV).
    :param b_hyper: The electric quadrupole hyperfine constant $B = eQ_I (\partial^2 V_J / \partial z^2)$ ([`a_hyper`]).
    :param b_field: The B-field $\mathcal{B}$ (T).
    :param g_n_as_gyro: Whether `gi` is the nuclear g-factor or the gyromagnetic ratio $\gamma_I$ (MHz).
    :param as_freq: The matrix element can be returned in energy (`False`, eV) or frequency units (`True`, MHz).
     The default is `True`
    :returns: (e_eig, m_list, f_list, mi_mj_list) The eigenvalues of the Hamiltonian $H$
     sorted according to lists of $m_F$, $F$ and $(m_I, m_J)$, which are returned as the second to forth arguments.
    """
    b_field = np.asarray(b_field, dtype=float).flatten()

    gi = lande_n(gi) if g_n_as_gyro else gi

    f_list = get_f(i, j)
    # mf_list = [get_m(_f) for _f in f_list]
    mi_list = get_m(i)
    mj_list = get_m(j)

    m_list = get_m(max(f_list))
    fm_list = [[_f for _f in f_list if abs(_m) <= _f] for _m in m_list]
    mi_mj_list = [[(_mi, _m - _mi) for _mi in mi_list if _m - _mi in mj_list] for _m in m_list]

    # n_list = [sum(int(abs(_m) <= _f) for _f in f_list) for _m in m_list]
    h_list = [np.array([[hyper_zeeman_ij(_mi0, _mj0, _mi1, _mj1, i, j, a_hyper, b_hyper, gi, gj, b_field,
                                         as_freq=as_freq)
                         for (_mi1, _mj1) in _mi_mj_list] for (_mi0, _mj0) in _mi_mj_list], dtype=float)
              for _m, _mi_mj_list in zip(m_list, mi_mj_list)]
    h_list = [np.transpose(_h, axes=[2, 0, 1]) for _h in h_list]

    h_eig = [np.linalg.eigh(_h) for _h in h_list]

    e_eig = [_h_eig[0] for _h_eig in h_eig]
    # v_eig = [_h_eig[1] for _h_eig in h_eig]

    e_0 = [np.array([hyperfine(i, j, _f, a_hyper, b_hyper) for _f in _f_list], dtype=float) for _f_list in fm_list]
    inv_order_fm = [list(np.argsort(_e_0)) for _e_0 in e_0]
    order_fm = [np.array([_inv_order.index(k) for k in range(len(_inv_order))], dtype=int)
                for _inv_order in inv_order_fm]
    e_eig = [_e_eig[:, _order] for _e_eig, _order in zip(e_eig, order_fm)]

    e_b = [np.array([-(mi * gi * mu_N + mj * gj * mu_B) * 100. / sc.h * 1e-6  # B = 100. can be any positive number.
                     for (mi, mj) in _mi_mj_list], dtype=float) for _mi_mj_list in mi_mj_list]
    inv_order_ij = [list(np.argsort(_e_b)) for _e_b in e_b]
    mi_mj_list = [[_mi_mj_list[k] for k in _inv_order]
                  for _inv_order, _mi_mj_list in zip(inv_order_ij, mi_mj_list)]
    mi_mj_list = [[_mi_mj_list[k] for k in _order]
                  for _order, _mi_mj_list in zip(order_fm, mi_mj_list)]

    return e_eig, m_list, fm_list, mi_mj_list


def hyper_zeeman_12(j: quant_like, m: quant_like, a_hyper: array_like = 0.,
                    gi: array_like = 0., gj: array_like = 0., b_field: array_like = 0.,
                    g_n_as_gyro: bool = False, as_freq: bool = True) -> (ndarray, ndarray):
    r"""
    The two eigenvalues of the hyperfine structure + Zeeman effect hamailtonian for a nuclear spin of $I=1/2$
    and the magnetic quantum number `m`, calculated analytically using the Breit-Rabi equation.

    :param j: The electronic total angular momentum quantum number $J$.
    :param m: The magnetic quantum number $m_F$.
    :param a_hyper: The magnetic dipole hyperfine constant $A = \mu_I \mathcal{B}_J / (IJ)$ (MHz if `as_freq` else eV).
    :param gi: The nuclear g-factor $g_I$ or the gyromagnetic ratio $\gamma_I$ if `g_n_as_gyro == True`.
    :param gj: The electronic g-factor $g_J$.
    :param b_field: The B-field $\mathcal{B}$ (T).
    :param g_n_as_gyro: Whether `gi` is the nuclear g-factor or the gyromagnetic ratio $\gamma_I$ (MHz).
    :param as_freq: The shift can be returned in energy (`False`, eV) or frequency units (`True`, MHz).
     The default is `True`
    :returns: (x0, x1) The two solutions of the Breit-Rabi equation,
     where `x0` and `x1` correspond to $F = J \mp 1/2$, respectively.
    """
    a_hyper = np.asarray(a_hyper, dtype=float)
    gi, gj = np.asarray(gi, dtype=float), np.asarray(gj, dtype=float)

    gi = lande_n(gi) if g_n_as_gyro else gi

    z_unit = 1e-6 / sc.h if as_freq else 1 / E_NORM
    b_field = np.asarray(b_field, dtype=float) * z_unit

    x_b0 = a_hyper * (j + 0.5)
    _x = b_field * (mu_B * gj - mu_N * gi) / x_b0

    x = -x_b0 / (2 * (2 * j + 1)) - mu_B * gj * m * b_field

    if m == j + 0.5:
        x0 = x + 0.5 * x_b0 * (1 + _x)
        x1 = x + 0.5 * x_b0 * (1 + _x)
    elif m == -j - 0.5:
        x0 = x + 0.5 * x_b0 * (1 - _x)
        x1 = x + 0.5 * x_b0 * (1 - _x)
    else:
        x0 = x - 0.5 * x_b0 * np.sqrt(1 + 2 * m * _x / (j + 0.5) + _x ** 2)
        x1 = x + 0.5 * x_b0 * np.sqrt(1 + 2 * m * _x / (j + 0.5) + _x ** 2)

    return x0, x1


def hyper_zeeman_12_d(j: quant_like, m: quant_like, a_hyper: array_like = 0.,
                      gi: array_like = 0., gj: array_like = 0., b_field: array_like = 0.,
                      g_n_as_gyro: bool = False, as_freq: bool = True) -> (ndarray, ndarray):
    r"""
    The first derivative of the two eigenvalues of the hyperfine structure + Zeeman effect hamailtonian,
    with respect to the `b-field`, for a nuclear spin of $I=1/2$ and the magnetic quantum number `m`,
    calculated analytically using the Breit-Rabi equation.

    :param j: The electronic total angular momentum quantum number $J$.
    :param m: The magnetic quantum number $m_F$.
    :param a_hyper: The magnetic dipole hyperfine constant $A = \mu_I \mathcal{B}_J / (IJ)$ (MHz if `as_freq` else eV).
    :param gi: The nuclear g-factor $g_I$ or the gyromagnetic ratio $\gamma_I$ if `g_n_as_gyro == True`.
    :param gj: The electronic g-factor $g_J$.
    :param b_field: The B-field $\mathcal{B}$ (T).
    :param g_n_as_gyro: Whether `gi` is the nuclear g-factor or the gyromagnetic ratio $\gamma_I$ (MHz).
    :param as_freq: The shift can be returned in energy (`False`, eV) or frequency units (`True`, MHz).
     The default is `True`
    :returns: (x0, x1) The first derivatives of the two solutions of the Breit-Rabi equation,
     where `x0` and `x1` correspond to $F = J \mp 1/2$, respectively.
    """
    a_hyper = np.asarray(a_hyper, dtype=float)
    gi, gj = np.asarray(gi, dtype=float), np.asarray(gj, dtype=float)

    gi = lande_n(gi) if g_n_as_gyro else gi

    z_unit = 1e-6 / sc.h if as_freq else 1 / E_NORM
    b_field = np.asarray(b_field, dtype=float) * z_unit

    x_b0 = a_hyper * (j + 0.5)

    _x = b_field * (mu_B * gj - mu_N * gi) / x_b0
    _dx = (mu_B * gj - mu_N * gi) / x_b0 * z_unit

    dx = -mu_B * gj * m * z_unit

    if m == j + 0.5:
        x0 = dx + 0.5 * x_b0 * _dx
        x1 = dx + 0.5 * x_b0 * _dx
    elif m == -j - 0.5:
        x0 = dx - 0.5 * x_b0 * _dx
        x1 = dx - 0.5 * x_b0 * _dx
    else:
        x0 = dx - 0.25 * x_b0 / np.sqrt(1 + 2 * m * _x / (j + 0.5) + _x ** 2) * (2 * m / (j + 0.5) + 2 * _x) * _dx
        x1 = dx + 0.25 * x_b0 / np.sqrt(1 + 2 * m * _x / (j + 0.5) + _x ** 2) * (2 * m / (j + 0.5) + 2 * _x) * _dx

    return x0, x1


def a_hyper_mu(i: quant_like, j: quant_like, mu: array_like, b_field: array_like) -> ndarray:
    r"""
    The magnetic dipole hyperfine structure constant as a function of the nuclear magnetic moment `mu`
    and the magnetic field `b_field` of the electrons at the nucleus $A = \mu\mathcal{B} / (IJ)$.

    :param i: The nuclear spin quantum number $I$.
    :param j: The electronic total angular momentum quantum number $J$.
    :param mu: The magnetic moment of the nucleus in units of the nuclear magneton ($\mu_\mathrm{N}$).
    :param b_field: The B-field $\mathcal{B}$ of the atomic electrons at the nucleus (T).
    :returns: (a_hyper) The hyperfine structure constant $A$ (MHz).
    """
    mu, b = np.asarray(mu, dtype=float), np.asarray(b_field, dtype=float)
    if i == 0 or j == 0:
        return np.zeros_like(mu * b)
    return mu * mu_N * b / (i * j * sc.h) * 1e-6


def a_einstein_m1(f: array_like, mu: array_like = None, j_l: quant_like = 0, j_u: quant_like = 0,
                  ls: quant_iter = (0, 0), jj_l: quant_iter = None, jj_u: quant_iter = None) -> ndarray:
    r"""
    The Einstein coefficient of an M1 transition

    $$
    A_\mathrm{ul} = \frac{2\mu_0(2\pi f)^3}{3hc^3}
    \frac{|\langle J_\mathrm{l}|\vec{\mu}|J_\mathrm{u}\rangle|^2}{2J_\mathrm{u} + 1},
    $$

    where $\mu_0$ is the vacuum permeability, $\vec{\mu}$ is the electronic magnetic moment operator
    in units of the Bohr magneton, $h$ is the Planck constant, and $c$ is the speed of light.
    If `mu` is `None`, this calculation neglects spin-orbit couplings, as $\Delta S = \Delta L = 0$ holds strictly
    in the estimation of `mu`. Consequently, M1 transition rates of transition such as
    $^3\mathrm{S}_1\rightarrow\, ^1\mathrm{S}_0$ cannot be calculated with this function without specifying `mu`.

    :param f: The frequency $f$ of a transition (MHz).
    :param mu: The magnetic dipole moment $\mu$ of a transition ($\mu_\mathrm{B}$).
     If `None`, the parameters `j_l`, `j_u`, `ls`, `jj_l`, `jj_u` are used to determine `mu`.
     Otherwise, they are omitted.
    :param j_l: The electronic total angular momentum quantum number $J_\mathrm{l}$ of the lower state.
    :param j_u: The electronic total angular momentum quantum number $J_\mathrm{u}$ of the upper state.
    :param ls: A list `[(l_c, s_c), (l_o, l_c)]` or a single pair `(L, S)`
     of electronic angular momentum and spin quantum numbers $(l, s)$ valid for both the lower and upper state.
     If this is a list of ls-pairs, the first pair specifies $l$ and $s$ of the core electron(s)
     and the second pair those of the outer electron(s). In this case, also a tuple of $(j_\mathrm{c}, j_\mathrm{o})$
     quantum numbers needs to specified for the parameters `jj_l` and `jj_u`.
    :param jj_l: A tuple of electronic total angular momentum quantum numbers $(j_\mathrm{c}, j_\mathrm{o})$
     of the lower state. Only needs to be specified if `ls` is a list of ls-pairs.
    :param jj_u: A tuple of electronic total angular momentum quantum numbers $(j_\mathrm{c}, j_\mathrm{o})$
     of the upper state. Only needs to be specified if `ls` is a list of ls-pairs.
    :returns: (A_ul) The Einstein coefficient $A_\mathrm{ul}$ of an M1 transition (MHz).
    """
    f = np.asarray(f, dtype=float)

    if mu is None:
        if jj_l is None or jj_u is None:
            mu = mu_j_m1(ls[1], ls[0], j_l, j_u)
        else:
            mu = mu_jj_m1(ls[0][1], ls[0][0], jj_l[0], jj_l[1], j_l, ls[1][1], ls[1][0], jj_u[0], jj_u[1], j_u)

    mu = np.asarray(mu, dtype=float)
    mu = (mu * mu_B) ** 2 / (2 * j_u + 1)
    return 8 * mu * np.pi ** 2 * sc.mu_0 * f ** 3 / (3 * sc.hbar * sc.c ** 3) * 1e12


def temperature_doppler(a: array_like) -> ndarray:
    r"""
    The minimum temperature achievable through Doppler cooling in a closed two-level transition (Doppler limit)

    $$
    T_\mathrm{Doppler} = \frac{\hbar \Gamma}{2k_\mathrm{B}}.
    $$

    :param a: The inverse lifetime $\Gamma = 1 / \tau$ of the upper state of the cooling transition (MHz).
    :returns: (T_doppler) The temperature $T_\mathrm{Doppler}$ (K).
    """
    a = np.asarray(a, dtype=float)
    return 0.5 * sc.hbar * a * 1e6 / sc.k


def saturation_intensity(f: array_like, a: array_like) -> ndarray:
    r"""
    The saturation intensity of an electronic dipole transition

    $$
    I_0 = \frac{\pi hf^3A_{ki}}{3c^2}.
    $$

    :param f: The frequency $f$ of the transition $|i\rangle\rightarrow|k\rangle$ (MHz).
    :param a: The Einstein $A_{ki}$ coefficient (MHz).
    :returns: (sat_intensity) The saturation intensity $I_0$.
    """
    f, a = np.asarray(f, dtype=float), np.asarray(a, dtype=float)
    return np.pi * (f * 1e6) ** 3 * sc.h * a * 1e6 / (3 * sc.c ** 2)


def saturation(intensity: array_like, f: array_like, a: array_like) -> ndarray:
    r"""
    The saturation parameter $s = I / I_0$, with the `saturation_intensity` $I_0$.

    :param intensity: The intensity of the laser (&mu;W/mm<sup>2</sup> = W/m<sup>2</sup>).
    :param f: The frequency $f$ of the transition $|i\rangle\rightarrow|k\rangle$ (MHz).
    :param a: The Einstein $A_{ki}$ coefficient (MHz).
    :returns: (s) The saturation parameter $s$.
    """
    intensity = np.asarray(intensity, dtype=float)
    return intensity / saturation_intensity(f, a)


def rabi_s(a: array_like, s: array_like) -> ndarray:
    r"""
    The Rabi frequency $\Omega = A_{ki}\sqrt{s/2}$.

    :param a: The Einstein $A_{ki}$ coefficient (MHz).
    :param s: The saturation parameter $s$.
    :returns: (omega) The Rabi frequency $\Omega$.
    """
    a, s = np.asarray(a), np.asarray(s)
    return a * np.sqrt(s / 2.)


def scattering_rate(df: array_like, a: array_like, s: array_like) -> ndarray:
    r"""
    The equilibrium scattering rate of a closed two-level transition

    $$\begin{aligned}
    \Gamma_\mathrm{sc} &= \frac{s}{2}\frac{A_{ki}^3}{(4\pi\Delta f)^2 + (1 + s)A_{ki}^2}\\[2ex]
    \lim\limits_{s\rightarrow\inf}\Gamma_\mathrm{sc}\big|_{\Delta f = 0} &= A_{ki} / 2.
    \end{aligned}$$

    :param df: The frequency detuning $\Delta f$ of the absorbed light.
     This must be differences of real frequencies, such that $\Delta\omega = 2\pi\Delta f$ (MHz).
    :param a: The Einstein $A_{ki}$ coefficient (MHz).
    :param s: The saturation parameter $s$.
    :returns: (Gamma) The equilibrium scattering rate $\Gamma_\mathrm{sc}$
     of a closed two-level transition (MHz).
    """
    df, a, s = np.asarray(df), np.asarray(a), np.asarray(s)
    return 0.125 * s * a ** 3 / (0.25 * (1 + s) * a ** 2 + (2 * np.pi * df) ** 2)


def mass_factor(m0: array_like, m1: array_like, m0_d: array_like, m1_d: array_like) -> (ndarray, ndarray):
    r"""
    The specific mass factor required to calculate modified isotope shifts or charge radii and its uncertainty

    $$\begin{aligned}
    \mu &= \frac{m_0 m_1}{m_0 - m_1}\\
    \Delta\mu &= \mu\sqrt{\left(\frac{\Delta m_0}{m_0} - \frac{\Delta m_0}{m_0 - m_1}\right)^2
    + \left(\frac{\Delta m_1}{m_1} + \frac{\Delta m_1}{m_0 - m_1}\right)^2}.
    \end{aligned}$$

    Use $m_0 = M_0 + m_\mathrm{e}$ and/or $m_1 = M_1 + m_\mathrm{e}$ with the nuclear masses $M_0$ and $M_1$
    and the electron mass $m_\mathrm{e}$ for King-plots.
    Compare (6.4) with (3.17) in [W. H. King, Isotope shifts in atomic spectra (1984)].

    :param m0: The mass $m_0$ of the first isotope (u).
    :param m1: The mass $m_1$ of the second isotope (u).
    :param m0_d: The mass uncertainty $\Delta m_0$ of the first isotope (u).
    :param m1_d: The mass uncertainty $\Delta m_1$ of the second isotope (u).
    :returns: (mu, mu_d) The mass factor $\mu$ and its uncertainty $\Delta\mu$
     required to calculate modified isotope shifts or charge radii.
    """
    m0, m1, m0_d, m1_d = (np.asarray(m0, dtype=float), np.asarray(m1, dtype=float),
                          np.asarray(m0_d, dtype=float), np.asarray(m1_d, dtype=float))
    scalar_true = tools.check_shape((), m0, m1, return_mode=True)
    if scalar_true:
        m0 = np.array([m0])

    m_rel = m0 - m1
    mask = m_rel == 0
    m_rel[mask] = 1

    mu = m0 * m1 / m_rel
    mu_d = ((mu / m0 - mu / m_rel) * m0_d) ** 2
    mu_d += ((mu / m1 + mu / m_rel) * m1_d) ** 2
    mu_d = np.sqrt(mu_d)

    mu[mask] = np.inf
    mu_d[mask] = 0

    if scalar_true:
        return mu[0], mu_d[0]
    return mu, mu_d


def delta_r2(r: array_like, r_d: array_like, r_ref: array_like, r_ref_d: array_like,
             dr: array_like = None, dr_d: array_like = None, v2: array_like = 1., v2_ref: array_like = 1.) \
        -> (ndarray, ndarray):
    r"""
    The difference of the mean square nuclear charge radius between two isotopes and its uncertainty
    calculated from the Barrett radii and elastic electron scattering form factors

    $$\begin{aligned}
    \delta\!\langle r^2\rangle &= \left(\frac{R}{V_2}\right)^2
    - \left(\frac{R_\mathrm{ref}}{V_{2,\mathrm{ref}}}\right)^2\\[2ex]
    \Delta\delta\!\langle r^2\rangle &=
    \sqrt{\left(\frac{2 R\Delta R}{V_2^2}\right)^2
    + \left(\frac{2 R_\mathrm{ref}\Delta R_\mathrm{ref}}{V_{2,\mathrm{ref}}^2}\right)^2}.
    \end{aligned}$$

    If the uncertainty `dr_d` is specified, which is often much lower than `r_d` and `r_ref_d`,
    an improved formula is used that gives smaller uncertainties

    $$\begin{aligned}
    \delta\!\langle r^2\rangle &= sd,
    \quad s \coloneqq \frac{1}{V_2}\left(\frac{R}{V_2} + \frac{R_\mathrm{ref}}{V_{2,\mathrm{ref}}}\right),
    \quad d \coloneqq \delta R + R_\mathrm{ref}\left(1 - \frac{V_2}{V_{2,\mathrm{ref}}}\right)\\[2ex]
    \Delta\delta\!\langle r^2\rangle &= \sqrt{\left(s\Delta\delta R\right)^2
    + \left(\frac{d}{V_2^2}\Delta R\right)^2
    + \left(\left[\frac{d}{V_2V_{2,\mathrm{ref}}}
    + s\left(1 - \frac{V_2}{V_{2,\mathrm{ref}}}\right)\right]\Delta R_\mathrm{ref}\right)^2}.
    \end{aligned}$$

    :param r: The Barrett radius $R$ of the first isotope (arb. units).
    :param r_d: The uncertainty of the Barrett radius $\Delta R$ of the first isotope ([`r`]).
    :param r_ref: The Barrett radius $R_\mathrm{ref}$ of the second isotope ([`r`]).
    :param r_ref_d: The uncertainty of the Barrett radius $\Delta R_\mathrm{ref}$ of the second isotope ([`r`]).
    :param dr: The difference between the Barrett radii $\delta R = R - R_\mathrm{ref}$
     of the first and second isotope ([`r`]).
    :param dr_d: The uncertainty of the difference between the Barrett radii $\Delta\delta R$
     of the first and second isotope ([`r`]).
    :param v2: The shape factor $V_2$ of the first isotope.
    :param v2_ref: The shape factor $V_{2,\mathrm{ref}}$ of the second isotope.
    :returns: (dr2, dr2_d) The difference of the mean-square nuclear charge radius between two isotopes
     $\delta\!\langle r^2\rangle$ and its uncertainty $\Delta\delta\!\langle r^2\rangle$ ([`r ** 2`]).
    """
    r, r_d = np.asarray(r, dtype=float), np.asarray(r_d, dtype=float)
    r_ref, r_ref_d = np.asarray(r_ref, dtype=float), np.asarray(r_ref_d, dtype=float)
    dr, dr_d = tools.asarray_optional(dr, dtype=float), tools.asarray_optional(dr_d, dtype=float)
    v2, v2_ref = np.asarray(v2, dtype=float), np.asarray(v2_ref, dtype=float)

    if dr is None and dr_d is not None:
        dr = r - r_ref

    if dr_d is None:
        val = (r / v2) ** 2 - (r_ref / v2_ref) ** 2
        err = np.sqrt((2 * r * r_d / v2 ** 2) ** 2 + (2 * r_ref * r_ref_d / v2_ref ** 2) ** 2)
    else:
        sum_term = (r / v2 + r_ref / v2_ref) / v2
        delta_term = dr + r_ref * (1. - v2 / v2_ref)
        val = sum_term * delta_term

        err = (sum_term * dr_d) ** 2
        err += (delta_term * r_d / (v2 ** 2)) ** 2
        err += ((delta_term / (v2 * v2_ref) + sum_term * (1. - v2 / v2_ref)) * r_ref_d) ** 2

    return val, np.sqrt(err)


def delta_r4(r: array_like, r_d: array_like, r_ref: array_like, r_ref_d: array_like,
             dr: array_like = None, dr_d: array_like = None, v4: array_like = 1., v4_ref: array_like = 1.) \
        -> (ndarray, ndarray):
    r"""
    The difference of the mean square nuclear charge radius between two isotopes and its uncertainty
    calculated from the Barrett radii and elastic electron scattering form factors

    $$\begin{aligned}
    \delta\!\langle r^4\rangle &= \left(\frac{R}{V_4}\right)^4
    - \left(\frac{R_\mathrm{ref}}{V_{4,\mathrm{ref}}}\right)^4\\[2ex]
    \Delta\delta\!\langle r^4\rangle &=
    \sqrt{\left(\frac{4 R^3\Delta R}{V_4^4}\right)^2
    + \left(\frac{4 R_\mathrm{ref}^3\Delta R_\mathrm{ref}}{V_{4,\mathrm{ref}}^4}\right)^2}.
    \end{aligned}$$

    If the uncertainty `dr_d` is specified, which is often much lower than `r_d` and `r_ref_d`,
    an improved formula is used that gives smaller uncertainties

    $$\begin{aligned}
    \delta\!\langle r^4\rangle &= sd,
    \quad s \coloneqq \left(\frac{R}{V_4}\right)^2 + \left(\frac{R_\mathrm{ref}}{V_{4,\mathrm{ref}}}\right)^2,
    \quad d \coloneqq \left(\frac{R}{V_4}\right)^2 - \left(\frac{R_\mathrm{ref}}{V_{4,\mathrm{ref}}}\right)^2\\[2ex]
    \Delta\delta\!\langle r^4\rangle &= \sqrt{\left(s\Delta d\right)^2
    + \left(\frac{2dR}{V_4^2}\Delta R\right)^2
    + \left(\frac{2dR_\mathrm{ref}}{V_{4,\mathrm{ref}}^2}\Delta R_\mathrm{ref}\right)^2}.
    \end{aligned}$$

    Here $d$ and $\Delta d$ are calculated using the small-uncertainty version of the `delta_r2` function
    for the mean-square nuclear charge radius.

    :param r: The Barrett radius $R$ of the first isotope (arb. units).
    :param r_d: The uncertainty of the Barrett radius $\Delta R$ of the first isotope ([`r`]).
    :param r_ref: The Barrett radius $R_\mathrm{ref}$ of the second isotope ([`r`]).
    :param r_ref_d: The uncertainty of the Barrett radius $\Delta R_\mathrm{ref}$ of the second isotope ([`r`]).
    :param dr: The difference between the Barrett radii $\delta R = R - R_\mathrm{ref}$
     of the first and second isotope ([`r`]).
    :param dr_d: The uncertainty of the difference between the Barrett radii $\Delta\delta R$
     of the first and second isotope ([`r`]).
    :param v4: The shape factor $V_4$ of the first isotope.
    :param v4_ref: The shape factor $V_{4,\mathrm{ref}}$ of the second isotope.
    :returns: (dr4, dr4_d) The difference of the mean-quartic nuclear charge radius between two isotopes
     $\delta\!\langle r^4\rangle$ and its uncertainty $\Delta\delta\!\langle r^4\rangle$ ([`r ** 4`]).
    """
    r, r_d = np.asarray(r, dtype=float), np.asarray(r_d, dtype=float)
    r_ref, r_ref_d = np.asarray(r_ref, dtype=float), np.asarray(r_ref_d, dtype=float)
    dr, dr_d = tools.asarray_optional(dr, dtype=float), tools.asarray_optional(dr_d, dtype=float)
    v4, v4_ref = np.asarray(v4, dtype=float), np.asarray(v4_ref, dtype=float)

    if dr is None and dr_d is not None:
        dr = r - r_ref

    if dr_d is None:
        val = (r / v4) ** 4 - (r_ref / v4_ref) ** 4
        err = np.sqrt((4 * r ** 3 * r_d / v4 ** 4) ** 2 + (4 * r_ref ** 3 * r_ref_d / v4_ref ** 4) ** 2)
    else:
        sum_term = (r / v4) ** 2 + (r_ref / v4_ref) ** 2
        delta_term = delta_r2(r, r_d, r_ref, r_ref_d, dr, dr_d, v4, v4_ref)
        val = sum_term * delta_term[0]

        err = (sum_term * delta_term[1]) ** 2
        err += (2. * delta_term[0] * r * r_d / (v4 ** 2)) ** 2
        err += (2. * delta_term[0] * r_ref * r_ref_d / (v4_ref ** 2)) ** 2

    return val, np.sqrt(err)


def delta_r6(r: array_like, r_d: array_like, r_ref: array_like, r_ref_d: array_like,
             dr: array_like = None, dr_d: array_like = None, v6: array_like = 1., v6_ref: array_like = 1.) \
        -> (ndarray, ndarray):
    r"""
    The difference of the mean square nuclear charge radius between two isotopes and its uncertainty
    calculated from the Barrett radii and elastic electron scattering form factors

    $$\begin{aligned}
    \delta\!\langle r^6\rangle &= \left(\frac{R}{V_6}\right)^6
    - \left(\frac{R_\mathrm{ref}}{V_{6,\mathrm{ref}}}\right)^6\\[2ex]
    \Delta\delta\!\langle r^6\rangle &=
    \sqrt{\left(\frac{6 R^5\Delta R}{V_6^6}\right)^2
    + \left(\frac{6 R_\mathrm{ref}^5\Delta R_\mathrm{ref}}{V_{6,\mathrm{ref}}^6}\right)^2}.
    \end{aligned}$$

    If the uncertainty `dr_d` is specified, which is often much lower than `r_d` and `r_ref_d`,
    an improved formula is used that gives smaller uncertainties

    $$\begin{aligned}
    \delta\!\langle r^6\rangle &= s(d + t),
    \quad s \coloneqq \frac{V_6}{R}\left[\left(\frac{R}{V_6}\right)^3
    + \left(\frac{R_\mathrm{ref}}{V_{6,\mathrm{ref}}}\right)^3\right],\\
    \quad d &\coloneqq \left(\frac{R}{V_6}\right)^4 - \left(\frac{R_\mathrm{ref}}{V_{6,\mathrm{ref}}}\right)^4
    \quad t \coloneqq \left(\frac{R_\mathrm{ref}}{V_{6,\mathrm{ref}}}\right)^4
    \left(1 - \frac{R}{R_\mathrm{ref}}\frac{V_{6,\mathrm{ref}}}{V_6}\right)\\[2ex]
    \Delta\delta\!\langle r^6\rangle &= \sqrt{\splitfrac{\left(s\Delta d\right)^2
    + \left(\left[\left(-\frac{R_\mathrm{ref}}{V_{6,\mathrm{ref}}}\right)^3\frac{s}{V_6}
    + (d + t)\left(-\frac{s}{R} + 3\frac{R}{V_6^2}\right)\right]\Delta R\right)^2}
    {+ \left(\left[s\left(\frac{4t}{R_\mathrm{ref}} + \frac{R}{V_6}\frac{R_\mathrm{ref}^2}{V_{6,\mathrm{ref}}^3}\right)
    + 3(d + t)\frac{V_6}{R}\frac{R_\mathrm{ref}^2}{V_{6,\mathrm{ref}}^3}\right]\Delta R_\mathrm{ref}\right)^2}}.
    \end{aligned}$$

    Here $d$ and $\Delta d$ are calculated using the small-uncertainty version of the `delta_r4` function
    for the mean-quartic nuclear charge radius.

    :param r: The Barrett radius $R$ of the first isotope (arb. units).
    :param r_d: The uncertainty of the Barrett radius $\Delta R$ of the first isotope ([`r`]).
    :param r_ref: The Barrett radius $R_\mathrm{ref}$ of the second isotope ([`r`]).
    :param r_ref_d: The uncertainty of the Barrett radius $\Delta R_\mathrm{ref}$ of the second isotope ([`r`]).
    :param dr: The difference between the Barrett radii $\delta R = R - R_\mathrm{ref}$
     of the first and second isotope ([`r`]).
    :param dr_d: The uncertainty of the difference between the Barrett radii $\Delta\delta R$
     of the first and second isotope ([`r`]).
    :param v6: The shape factor $V_6$ of the first isotope.
    :param v6_ref: The shape factor $V_{6,\mathrm{ref}}$ of the second isotope.
    :returns: (dr6, dr6_d) The difference of the mean-sextic nuclear charge radius between two isotopes
     $\delta\!\langle r^6\rangle$ and its uncertainty $\Delta\delta\!\langle r^6\rangle$ ([`r**6`]).
    """
    r, r_d = np.asarray(r, dtype=float), np.asarray(r_d, dtype=float)
    r_ref, r_ref_d = np.asarray(r_ref, dtype=float), np.asarray(r_ref_d, dtype=float)
    dr, dr_d = tools.asarray_optional(dr, dtype=float), tools.asarray_optional(dr_d, dtype=float)
    v6, v6_ref = np.asarray(v6, dtype=float), np.asarray(v6_ref, dtype=float)

    if dr is None and dr_d is not None:
        dr = r - r_ref

    if dr_d is None:
        val = (r / v6) ** 6 - (r_ref / v6_ref) ** 6
        err = np.sqrt((6 * r ** 5 * r_d / v6 ** 6) ** 2 + (6 * r_ref ** 5 * r_ref_d / v6_ref ** 6) ** 2)
    else:
        sum_term = (v6 / r) * ((r / v6) ** 3 + (r_ref / v6_ref) ** 3)
        delta = delta_r4(r, r_d, r_ref, r_ref_d, dr, dr_d, v6, v6_ref)
        delta_term = delta[0] + (r_ref / v6_ref) ** 4 * (1. - (r / v6) * (v6_ref / r_ref))
        val = sum_term * delta_term

        err = (sum_term * delta[1]) ** 2
        err += ((-(r_ref / v6_ref) ** 3 * sum_term / v6
                 + delta_term * (-sum_term / r + 3. * r / (v6 ** 2))) * r_d) ** 2
        err += (((4 * r_ref ** 3 / (v6_ref ** 4) * (1. - (r / v6) * (v6_ref / r_ref))
                  + (r / v6) * r_ref ** 2 / (v6_ref ** 3)) * sum_term
                 + delta_term * 3. * (v6 / r) * r_ref ** 2 / (v6_ref ** 3)) * r_ref_d) ** 2

    return val, np.sqrt(err)


def lambda_r(r: array_like, r_d: array_like, r_ref: array_like, r_ref_d: array_like,
             dr: array_like = None, dr_d: array_like = None, v2: array_like = 1., v2_ref: array_like = 1.,
             v4: array_like = 1., v4_ref: array_like = 1., v6: array_like = 1., v6_ref: array_like = 1.,
             c2c1: array_like = 1., c3c1: array_like = 1.) -> (ndarray, ndarray):
    r"""
    The differential nuclear charge radius series up to $\mathcal{O}(r^6)$ between two isotopes and its uncertainty
    calculated from the Barrett radii, elastic electron scattering form factors and the Seltzer coefficients

    $$\begin{aligned}
    \Lambda &= \delta\!\langle r^2\rangle + \frac{C_2}{C_1}\delta\!\langle r^4\rangle
    + \frac{C_3}{C_1}\delta\!\langle r^6\rangle\\[2ex]
    \Delta\Lambda &= \sqrt{\Delta\delta\!\langle r^2\rangle^2
    + \left(\frac{C_2}{C_1}\Delta\delta\!\langle r^4\rangle\right)^2
    + \left(\frac{C_3}{C_1}\Delta\delta\!\langle r^6\rangle\right)^2}
    \end{aligned}$$
    
    The moments of the differential nuclear charge radii and their uncertainties are calculated with the functions
    `delta_r2`, `delta_r4` and `delta_r6`.

    :param r: The Barrett radius $R$ of the first isotope (arb. units).
    :param r_d: The uncertainty of the Barrett radius $\Delta R$ of the first isotope ([`r`]).
    :param r_ref: The Barrett radius $R_\mathrm{ref}$ of the second isotope ([`r`]).
    :param r_ref_d: The uncertainty of the Barrett radius $\Delta R_\mathrm{ref}$ of the second isotope ([`r`]).
    :param dr: The difference between the Barrett radii $\delta R = R - R_\mathrm{ref}$
     of the first and second isotope ([`r`]).
    :param dr_d: The uncertainty of the difference between the Barrett radii $\Delta\delta R$
     of the first and second isotope ([`r`]).
    :param v2: The shape factor $V_2$ of the first isotope.
    :param v2_ref: The shape factor $V_{2,\mathrm{ref}}$ of the second isotope.
    :param v4: The shape factor $V_4$ of the first isotope.
    :param v4_ref: The shape factor $V_{4,\mathrm{ref}}$ of the second isotope.
    :param v6: The shape factor $V_6$ of the first isotope.
    :param v6_ref: The shape factor $V_{6,\mathrm{ref}}$ of the second isotope.
    :param c2c1: Seltzer's coefficient for the quartic moment $C_2 / C_1$ ([`1 / r**2`]).
    :param c3c1: Seltzer's coefficient for the sextic moment $C_3 / C_1$ ([`1 / r**4`]).
    :returns: (Lambda, Lambda_d) The difference of the nuclear charge radius series between two isotopes $\Lambda$
     and its uncertainty $\Delta\Lambda$ ([`r**2`]).
    """
    r2 = delta_r2(r, r_d, r_ref, r_ref_d, dr, dr_d, v2, v2_ref)
    r4 = delta_r4(r, r_d, r_ref, r_ref_d, dr, dr_d, v4, v4_ref)
    r6 = delta_r6(r, r_d, r_ref, r_ref_d, dr, dr_d, v6, v6_ref)
    return lambda_rn(r2[0], r2[1], r4[0], r4[1], r6[0], r6[1], c2c1, c3c1)


def lambda_rn(r2: array_like, r2_d: array_like, r4: array_like, r4_d: array_like,
              r6: array_like, r6_d: array_like, c2c1: array_like = 1., c3c1: array_like = 1.) -> (ndarray, ndarray):
    r"""
    The differential nuclear charge radius series up to $\mathcal{O}(r^6)$ between two isotopes and its uncertainty
    calculated from the moments of the nuclear charge radii and the Seltzer coefficients

    $$\begin{aligned}
    \Lambda &= \delta\!\langle r^2\rangle + \frac{C_2}{C_1}\delta\!\langle r^4\rangle
    + \frac{C_3}{C_1}\delta\!\langle r^6\rangle\\[2ex]
    \Delta\Lambda &= \sqrt{\Delta\delta\!\langle r^2\rangle^2
    + \left(\frac{C_2}{C_1}\Delta\delta\!\langle r^4\rangle\right)^2
    + \left(\frac{C_3}{C_1}\Delta\delta\!\langle r^6\rangle\right)^2}
    \end{aligned}$$
    
    :param r2: The difference of the mean-square nuclear charge radius between two isotopes
    $\delta\!\langle r^2\rangle$.
    :param r2_d: The uncertainty of the difference of the mean-square nuclear charge radius
    $\Delta\delta\!\langle r^2\rangle$.
    :param r4: The difference of the mean-quartic nuclear charge radius between two isotopes
    $\delta\!\langle r^4\rangle$.
    :param r4_d: The uncertainty of the difference of the mean-quartic nuclear charge radius
    $\Delta\delta\!\langle r^4\rangle$.
    :param r6: The difference of the mean-sextic nuclear charge radius between two isotopes
    $\delta\!\langle r^6\rangle$.
    :param r6_d: The uncertainty of the difference of the mean-sextic nuclear charge radius
    $\Delta\delta\!\langle r^6\rangle$.
    :param c2c1: Seltzer's coefficient for the quartic moment $C_2 / C_1$ ([`1 / r**2`]).
    :param c3c1: Seltzer's coefficient for the sextic moment $C_3 / C_1$ ([`1 / r**4`]).
    :returns: (Lambda, Lambda_d) The difference of the nuclear charge radius series between two isotopes $\Lambda$
     and its uncertainty $\Delta\Lambda$ ([`r**2`]).
    """
    r2, r2_d = np.asarray(r2, dtype=float), np.asarray(r2_d, dtype=float)
    r4, r4_d = np.asarray(r4, dtype=float), np.asarray(r4_d, dtype=float)
    r6, r6_d = np.asarray(r6, dtype=float), np.asarray(r6_d, dtype=float)
    c2c1, c3c1 = np.asarray(c2c1, dtype=float), np.asarray(c3c1, dtype=float)
    val = r2 + c2c1 * r4 + c3c1 * r6
    err = r2_d ** 2
    err += (c2c1 * r4_d) ** 2
    err += (c3c1 * r6_d) ** 2
    return val, np.sqrt(err)


def schmidt_line(l: quant_like, i: quant_like, is_proton: bool) -> ndarray:
    r"""
    Calculate the single-particle Schmidt value of the nuclear magnetic moment

    $$
    \mu = \begin{cases}\frac{I}{(I + 1)}\left((L + 1)g_L - \frac{1}{2}g_s\right) & I < L \\
    Lg_L + \frac{1}{2}g_s & \text{else}\end{cases}.
    $$
    
    :param l: The orbital angular momentum quantum number of the nucleon $L$.
    :param i: The nuclear spin quantum number $I$.
    :param is_proton: Whether the contributing nucleon is a proton (`True`) or a neutron (`False`).
    :returns: (mu_schmidt) The Schmidt value of the nuclear magnetic moment $\mu$ ($\mu_\mathrm{N}$).
    """
    _g_s = gp_s if is_proton else gn_s
    _g_l = 1 if is_proton else 0

    if i < l:
        ret = i / (i + 1) * ((l + 1) * _g_l - 0.5 * _g_s)
    else:
        ret = l * _g_l + 0.5 * _g_s

    return np.array(ret, dtype=float)


""" Optics """


def sellmeier(w: array_like, a: array_like, b: array_like) -> ndarray:
    r"""
    The Sellmeier equation for calculating the refractive index of a material with the lists of coefficients
    `a` and `b` for the wavelength `w`

    $$
    n = \sqrt{1 + \sum_i \frac{A_i\lambda^2}{\lambda^2 - B_i}}.
    $$

    :param w: The wavelength $\lambda$ (&mu;m).
    :param a: A list of coefficients $A_i$.
    :param b: A list of coefficients $B_i$ (&mu;m).
    :returns: (n) The refractive index $n$.
    """
    w, a, b = np.asarray(w, dtype=float), np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    tools.check_dimension(a.shape[0], 0, b)
    sum_term = np.sum([a_i * w ** 2 / (w ** 2 - b_i) for a_i, b_i in zip(a, b)], axis=0)
    return np.sqrt(1 + sum_term)


""" 3-D kinematics """


def gamma_3d(v: array_like, axis: int = -1) -> ndarray:
    r"""
    The time-dilation/Lorentz factor

    $$
    \gamma = \frac{1}{\sqrt{1 - (\vec{v}/ c)^2}},
    $$

    with the vacuum speed of light $c$ and the velocity vector $\vec{v}$.

    :param v: The velocity 3-vector $\vec{v}$. Must have size 3 in the specified `axis` (m/s).
    :param axis: The axis along which the vector components are aligned.
    :returns: (gamma) The time-dilation/Lorentz factor $\gamma$ corresponding to the velocity vector `v`.
    :raises ValueError: `v` must have 3 components along the specified `axis`.
    """
    tools.check_dimension(3, axis, v)
    return gamma(tools.absolute(v, axis=axis))


def boost(x: array_like, v: array_like, axis: int = -1) -> ndarray:
    r"""
    The relativistic Lorentz-boosted 4-vector

    $$
    x^{\prime\mu} = \Lambda^\nu_\mu(\vec{v})x^\mu,
    $$

    with the Lorentz-boost matrix $\Lambda_\nu^\mu$.

    :param x: The 4-vector $x^\mu$ in the stationary rest frame (arb. units).
     Must have size 4 in the specified `axis`.
    :param v: The velocity 3-vector $\vec{v}$. Must have size 3 in the specified `axis` (m/s).
    :param axis: The axis along which the vector components are aligned.
    :returns: (x_boosted) The 4-vector $x^{\prime\mu}$ in the coordinate system moving with velocity `v`
    relative to the current rest frame ([`x`]).
    :raises ValueError: `x` and `v` must have 4 and 3 components along the specified `axis`, respectively.
     The shapes of `x` and `v` must be compatible.
    """
    x, v = np.asarray(x), np.asarray(v)
    tools.check_dimension(4, axis, x)
    tools.check_dimension(3, axis, v)
    bet = beta(v)
    bet_abs = beta(tools.absolute(v, axis=axis))
    tools.check_shape_like(np.sum(x, axis=axis), bet_abs, allow_scalar=False)
    bet_abs[bet_abs == 0] = 1

    gam = gamma_3d(v, axis=axis)
    b_xyz = np.array([[1. + (gam - 1.) * np.take(bet, i, axis=axis) * np.take(bet, j, axis=axis) / (bet_abs ** 2)
                       if i == j else (gam - 1.) * np.take(bet, i, axis=axis) * np.take(bet, j, axis=axis)
                       / (bet_abs ** 2) for j in range(3)] for i in range(3)])
    b = np.array([[gam, -gam * np.take(bet, 0, axis=axis), -gam * np.take(bet, 1, axis=axis),
                   -gam * np.take(bet, 2, axis=axis)],
                  [-gam * np.take(bet, 0, axis=axis), b_xyz[0, 0], b_xyz[0, 1], b_xyz[0, 2]],
                  [-gam * np.take(bet, 1, axis=axis), b_xyz[1, 0], b_xyz[1, 1], b_xyz[1, 2]],
                  [-gam * np.take(bet, 2, axis=axis), b_xyz[2, 0], b_xyz[2, 1], b_xyz[2, 2]]])
    axes = list(range(len(v.shape)))
    axes.insert(0, axes.pop(axis))
    x = np.transpose(x, axes=axes)
    y = np.array([np.sum(b[i] * x, axis=0) for i in range(4)])
    axes = list(range(1, len(axes)))
    axes.insert(axis, 0) if axis != -1 else axes.append(0)
    return np.transpose(y, axes=axes)


def doppler_3d(k: array_like, v: array_like, return_frame='atom', axis=-1) -> ndarray:
    r"""
    The 3D Doppler shift of the 3-vector $\vec{k}$

    $$
    k^{\prime\mu} = \Lambda^\nu_\mu(\pm\vec{v})k^\mu,
    $$

    with the Lorentz-boost matrix $\Lambda_\nu^\mu$.

    :param k: The 3-vector $\vec{k}$ of light. Must have size 3 in the specified `axis (arb. units)`.
    :param v: The velocity 3-vector $\vec{v}$. Must have size 3 in the specified `axis` (m/s).
    :param return_frame: The coordinate system in which the frequency is returned. Can be either `'atom'` or `'lab'`.
    :param axis: The axis along which the vector components are aligned.
    :returns: (k_boosted) the Doppler-shifted 3-vector $\vec{k}^{\prime}$ in either the rest frame of the atom
     or the laboratory frame ([`k`]).
    :raises ValueError: `rest_frame` must be either `'atom'` or `'lab'`. The shapes of `k` and `v` must be compatible.
    """
    k, v = np.asarray(k), np.asarray(v)
    tools.check_dimension(3, axis, k, v)
    k_0 = tools.absolute(k, axis=axis)
    k_4 = np.concatenate([np.expand_dims(k_0, axis=axis), k], axis=axis)
    if return_frame == 'atom':
        """ Return k in the atomic system. """
        ret = boost(k_4, v)
    elif return_frame == 'lab':
        """ Return k in the laboratory system. """
        ret = boost(k_4, -v)
    else:
        raise ValueError('rest_frame must be either \'atom\' or \'lab\'.')

    i = [slice(None)] * len(ret.shape)
    i[axis] = slice(1, None)
    return ret[i]


def gaussian_beam_3d(r: array_like, k: array_like, w0: array_like, p0: array_like,
                     r0: array_like = None, axis: int = -1) -> ndarray:
    r"""
    The Gaussian beam intensity at the position $\vec{r} - \vec{r}_0$

    $$\begin{aligned}
    I(\vec{r}) &= \fracc{2P_0}{\pi w_z^2}\exp\left[-2\left(\!\frac{\rho}{w_z}\right\!)^{\! 2}\right]\\[1ex]
    w_z &= w_0\sqrt{1 + \left(\!\frac{z}{z_0}\!\right)^{\!2}}\\[1ex]
    z_0 &= \frac{1}{2}|\vec{k}|w_0^2\\[2ex]
    z &= (\vec{r} - \vec{r}_0)\cdot\hat{k}\\[2ex]
    \rho &= \sqrt{\left[(\vec{r} - \vec{r}_0)\cdot\hat{x}\right]^2
    + \left[(\vec{r} - \vec{r}_0)\cdot\hat{y}\right]^2},
    \end{aligned}$$

    where \hat{k} is the unit vector in $\vec{k}$ direction and $\hat{x}$, $\hat{y}$
    are unit vectors orthogonal to \hat{k}.

    :param r: The position 3-vector $\vec{r}$ where to calculate the beam intensity (m).
    :param k: The 3-vector $\vec{k}$ of light, where $|\vec{k}| = \omega / c$ (rad / m).
    :param w0: The beam waist $w_0$ (m).
    :param p0: The total power $P_0$ propagated by the gaussian beam (W).
    :param r0: The position 3-vector $\vec{r}_0$ of the beam waist. If `r0` is `None`, it is `[0., 0., 0.]` (m).
    :param axis: The axis along which the vector components are aligned.
    :returns: (I_r) The intensity $I(\vec{r}) (W/m<sup>2</sup> = uW/mm<sup>2</sup>).
    :raises ValueError: `r`, `k` and `r0` must have 3 components along the specified `axis`.
     The shapes of `r`, `k`, `w0`, `r0` and `p0` must be compatible.
    """
    if r0 is None:
        r0 = 0.

    r, k, w0, p0, r0 = (np.asarray(r, dtype=float), np.asarray(k, dtype=float), np.asarray(w0, dtype=float),
                        np.asarray(p0, dtype=float), np.asarray(r0, dtype=float))
    # tools.check_dimension(3, axis, r, k)
    # tools.check_shape_like(np.sum(r, axis=axis), np.sum(k, axis=axis), w0, np.sum(r0, axis=axis), p0)

    k_abs = tools.absolute(k, axis=axis)
    e_r, e_theta, e_phi = tools.orthonormal(k)

    rho = np.sqrt(np.sum((r - r0) * e_theta, axis=axis) ** 2 + np.sum((r - r0) * e_phi, axis=axis) ** 2)
    z = np.sum((r - r0) * e_r, axis=axis)

    z0 = 0.5 * w0 ** 2 * k_abs
    w_z = w0 * np.sqrt(1. + (z / z0) ** 2)
    return 2. * p0 / (np.pi * w_z ** 2) * np.exp(-2. * (rho / w_z) ** 2)


def gaussian_doppler_3d(r: array_like, k: array_like, w0: array_like, v: array_like, r0: array_like = None,
                        axis: int = -1) -> ndarray:
    r"""
    The length |$\vec{k}^\prime$| of the Doppler-shifted 3-vector `k` in the rest frame of the atom

    $$\begin{aligned}
    |\vec{k}^\prime| &= |\vec{k}^\prime|\gamma\left[1 - \beta\cos(\alpha)\left(1 - \frac{w_0^2}{2z_+} 
    - \frac{\rho^2z_-}{2z_+^2}\right) - \beta\sin(\alpha)\rho\frac{z}{z_+}\right]\\[2ex]
    z_\pm &= z^2 \pm z_0^2\\[2ex]
    z_0 &= \frac{1}{2}|\vec{k}|w_0^2\\[2ex]
    z &= (\vec{r} - \vec{r}_0)\cdot\hat{k}\\[2ex]
    \rho &= \sqrt{\left[(\vec{r} - \vec{r}_0)\cdot\hat{x}\right]^2
    + \left[(\vec{r} - \vec{r}_0)\cdot\hat{y}\right]^2},
    \end{aligned}$$
    
    where $\beta = |\vec{v}| / c$ is the relativistic velocity, $\gamma$ the time-dilation factor and $\alpha$ the angle
    between $\vec{k}$ and $\vec{v}$.

    :param r: The position 3-vector $\vec{r}$ where to calculate the beam intensity (m).
    :param k: The 3-vector $\vec{k}$ of light, where $|\vec{k}| = \omega / c$ (rad / m).
    :param w0: The beam waist $w_0$ (m).
    :param v: The velocity 3-vector $\vec{v}$ (m/s).
    :param r0: The position 3-vector $\vec{r}_0$ of the beam waist. If `r0` is `None`, it is `[0., 0., 0.]` (m).
    :param axis: The axis along which the vector components are aligned.
    :returns: (k_abs) The length |$\vec{k}^\prime$| (rad / m).
    :raises ValueError: `r`, `k`, `v` and `r0` must have 3 components along the specified `axis`.
     The shapes of `r`, `k`, `w0`, `v` and `r0` must be compatible.
    """
    if r0 is None:
        r0 = np.zeros_like(r)
    r, r0, k, v = np.asarray(r), np.asarray(r0), np.asarray(k), np.asarray(v)
    tools.check_dimension(3, axis, r, r0, k, v)
    tools.check_shape_like(np.sum(r, axis=axis), np.sum(k, axis=axis), np.array(w0),
                           np.sum(v, axis=axis), np.sum(r0, axis=axis))
    k_abs = tools.absolute(k, axis=axis)
    e_r, e_theta, e_phi = tools.orthonormal(k)
    rho = np.sqrt(np.sum((r - r0) * e_theta, axis=axis) ** 2 + np.sum((r - r0) * e_phi, axis=axis) ** 2)
    z = np.sum((r - r0) * e_r, axis=axis)
    z_0 = 0.5 * w0 ** 2 * k_abs
    z_plus = z ** 2 + z_0 ** 2
    z_minus = z ** 2 - z_0 ** 2
    alpha = tools.angle(v, k, axis=axis)
    bet_abs = beta(tools.absolute(v, axis=axis))
    return k_abs * gamma_3d(v) * (1. - bet_abs * np.cos(alpha) * (1. - 0.5 * w0 ** 2 / z_plus
                                                                  - 0.5 * rho ** 2 * z_minus / (z_plus ** 2))
                                  - bet_abs * np.sin(alpha) * rho * z / z_plus)


""" Probability distributions """


def t_xi(xi: array_like, f: array_like, u: array_like, q: array_like, m: array_like) -> ndarray:
    r"""
    The temperature of an ensemble of ions with acceleration/bunching parameter `xi`

    $$
    T = \xi\frac{\sqrt{8qUmc^2}}{\gamma k_\mathrm{B}f},
    $$

    where $\gamma$ is the relativistic time-dilation factor.

    :param xi: The acceleration/bunching parameter $\xi$ used as a parameter in `source_energy_pdf`.
     `xi` is usually derived from fitting to asymmetric spectral lines (MHz).
    :param f: The rest-frame transition frequency (MHz).
    :param u: The acceleration voltage (V).
    :param q: The electric charge of the ions (e).
    :param m: The mass of the ions (u).
    :returns: (T) The temperature $T$ (K).
    """
    xi, f, u, q, m = (np.asarray(xi, dtype=float), np.asarray(f, dtype=float), np.asarray(u, dtype=float),
                      np.asarray(q, dtype=float), np.asarray(m, dtype=float))
    return xi * np.sqrt(8 * q * sc.e * u * m * sc.u * sc.c ** 2) / (sc.k * f * gamma_e_kin(q * u, m))


def normal_vx_pdf(vx: array_like, m: array_like, t: array_like) -> ndarray:
    r"""
    The Gaussian probability density of a velocity component $v_x$

    $$
    \rho(v_x) = \sqrt{\frac{m}{2\pi\k_\mathrm{B}T}}\exp\left(-\frac{mv_x^2}{2\k_\mathrm{B}T}\right),
    $$

    for a mass $m$ particle in thermal equilibrium at temperature $T$.

    :param vx: The velocity quantiles $v_x$ (m/s).
    :param m: The mass $m$ of the particle (u).
    :param t: The temperature $T$ of the environment (K).
    :returns: (rho_vx) The probability density in thermal equilibrium at the velocity `vx` (s/m).
    """
    vx, m, t = np.asarray(vx, dtype=float), np.asarray(m, dtype=float), np.asarray(t, dtype=float)
    scale = np.sqrt(sc.k * t / (m * sc.atomic_mass))
    return st.norm.pdf(vx, scale=scale)


def normal_vx_rvs(m: array_like, t: array_like, size: Union[int, tuple] = 1) -> ndarray:
    r"""
    Random sample velocity components $v_x$ from the Gaussian probability density function

    $$
    \rho(v_x) = \sqrt{\frac{m}{2\pi\k_\mathrm{B}T}}\exp\left(-\frac{mv_x^2}{2\k_\mathrm{B}T}\right),
    $$

    for a mass $m$ particle in thermal equilibrium at temperature $T$.

    :param m: The mass $m$ of the particle (u).
    :param t: The temperature $T$ of the environment (K).
    :param size: Either the size (`int`) or shape (`tuple`) of the returned velocity array.
     If `m` or `t` is an iterable/array, their common shape must be appended to the desired shape of the random samples.
    :returns: (vx) Random velocities $v_x$ (m/s).
    """
    m, t = np.asarray(m, dtype=float), np.asarray(t, dtype=float)
    scale = np.sqrt(sc.k * t / (m * sc.atomic_mass))
    return st.norm.rvs(scale=scale, size=size)


def chi2_ex_pdf(e: array_like, t: array_like) -> ndarray:
    r"""
    The $\chi^2_1$ probability density of an energy component $E_x$

    $$
    \rho(E_x) = \sqrt{\frac{1}{\piE_x\k_\mathrm{B}T}}\exp\left(-\frac{E_x}{\k_\mathrm{B}T}\right),
    $$

    in thermal equilibrium at temperature $T$.

    :param e: The energy quantiles $E_x$ (eV).
    :param t: The temperature $T$ of the environment (K).
    :returns: The probability density in thermal equilibrium at the energy `e` (1/eV).
    """
    e, t = np.asarray(e, dtype=float), np.asarray(t, dtype=float)
    scale = 0.5 * sc.k * t / E_NORM
    return st.chi2.pdf(e, 1, scale=scale)


def chi2_ex_rvs(t: array_like, size: Union[int, tuple] = 1) -> ndarray:
    r"""
    A random sample energy components $E_x$ from the $\chi^2_1$ probability density function

    $$
    \rho(E_x) = \sqrt{\frac{1}{\piE_x\k_\mathrm{B}T}}\exp\left(-\frac{E_x}{\k_\mathrm{B}T}\right),
    $$

    in thermal equilibrium at temperature $T$.

    :param t: The temperature $T$ of the environment (K).
    :param size: Either the size (`int`) or shape (`tuple`) of the returned velocity array.
     If `m` or `t` is an iterable/array, their common shape must be appended to the desired shape of the random samples.
    :returns: (vx) Random energies $E_x$ (m/s).
    """
    t = np.asarray(t, dtype=float)
    scale = 0.5 * sc.k * t / E_NORM
    return st.chi2.rvs(1, scale=scale, size=size)


def convolved_boltzmann_norm_pdf(e: array_like, t: array_like, scale_e: array_like, e0: array_like = 0) -> ndarray:
    """
    :param e: energy quantiles (eV).
    :param t: The temperature of the ensemble (K).
    :param scale_e: The standard deviation of the normal distribution (eV).
    :param e0: The mean energy of the normal distribution (eV).
    :returns: The probability density at the energy e, distributed according
     to a convolution of the boltzmann and a normal distribution (1/eV).
    """
    e, t, scale_e, e0 = np.asarray(e), np.asarray(t), np.asarray(scale_e), np.asarray(e0)
    t /= E_NORM
    scale = scale_e / (sc.k * t)
    loc = (e - e0) / (sc.k * t) - scale ** 2
    nonzero = loc.astype(bool)
    loc = loc[nonzero]
    norm = np.exp(-0.5 * scale ** 2) \
        / (np.sqrt(2.) * np.pi * scale * sc.k * t)
    x = (loc / (2. * scale)) ** 2
    main = np.full(e.shape, np.sqrt(LEMNISCATE * np.sqrt(np.pi) * scale))
    main_nonzero = np.empty_like(e[nonzero], dtype=float)
    mask = loc < 0.
    main_nonzero[mask] = np.sqrt(-loc[mask] / 2.) * np.exp(-loc[mask]) \
        * sp.kv(0.25, x[mask]) * np.exp(-x[mask])
    main_nonzero[~mask] = np.pi / 2. * np.sqrt(loc[~mask]) * np.exp(-loc[~mask]) \
        * (sp.ive(0.25, x[~mask]) + sp.ive(-0.25, x[~mask]))
    main[nonzero] = main_nonzero * norm
    return main


def convolved_thermal_norm_v_pdf(v: array_like, m: array_like, t: array_like,
                                 scale_e: array_like, e0: array_like = 0, relativistic=True) -> ndarray:
    """
    :param v: velocity quantiles. All values must have the same sign (m/s).
    :param m: The mass of the ensembles bodies (amu).
    :param t: The temperature of the ensemble (K).
    :param scale_e: The standard deviation of the normal distribution (eV).
    :param e0: The mean energy of the normal distribution (eV).
    :param relativistic: The calculation is performed either relativistically or classically.
    :returns: The probability density at the velocity v, corresponding to the kinetic energy, distributed according
     to a convolution of the boltzmann and a normal distribution (s/m).
    """
    v, m, t, scale_e, e0 = np.asarray(v), np.asarray(m), np.asarray(t), np.asarray(scale_e), np.asarray(e0)
    if np.any(v < 0.) and np.any(v > 0.):
        raise ValueError('This pdf can only describe the case where all velocities have the same sign.')
    energy = e_kin(v, m, relativistic)
    tr = m * sc.atomic_mass * np.abs(v)
    if relativistic:
        tr *= gamma(v) ** 3
    return convolved_boltzmann_norm_pdf(energy, t, scale_e, e0=e0) * tr / E_NORM


def convolved_thermal_norm_f_pdf(f: array_like, f_lab: array_like, alpha: array_like, m: array_like, t: array_like,
                                 scale_e: array_like, e0: array_like = 0, relativistic=True) -> ndarray:
    """
    :param f: Frequency quantiles (arb. units).
    :param f_lab: Laser frequency in the laboratory frame ([f]).
    :param alpha: Angle between the laser and the atoms velocity direction (rad).
    :param m: The mass of the ensembles bodies (amu).
    :param t: The temperature of the ensemble (K).
    :param scale_e: The standard deviation of the normal distribution (eV).
    :param e0: The mean energy of the normal distribution (eV).
    :param relativistic: The calculation is performed either relativistically or classically.
    :returns: The probability density at the frequency 'f' in the atoms rest frame,
     related to the kinetic energy via the laser frequency 'f_lab' and the Doppler effect.
     The kinetic energies are distributed according to a convolution of the boltzmann and a normal distribution (1/MHz).
    """
    f, f_lab = np.asarray(f), np.asarray(f_lab)
    m, t, scale_e, e0 = np.asarray(m), np.asarray(t), np.asarray(scale_e), np.asarray(e0)

    v = inverse_doppler(f, f_lab, alpha, mode='isnan-small')
    tr = np.abs(inverse_doppler_d1(f, f_lab, alpha, mode='isnan-small'))
    mask = np.isnan(v)
    ret = np.zeros(f.shape)
    ret[~mask] = convolved_thermal_norm_v_pdf(v[~mask], m, t, scale_e, e0=e0, relativistic=relativistic) * tr[~mask]
    return ret


def convolved_thermal_norm_f_lin_pdf(f: array_like, xi: array_like, sigma: array_like, col=True) -> ndarray:
    """
    :param f: Frequency quantiles (arb. units).
    :param xi: The proportionality constant between kinetic energy differences and frequency differences ([f]).
    :param sigma: The standard deviation of the underlying normal distribution in frequency units ([f]).
    :param col: Col/Acol alignment of the laser relative to the atom beam.
    :returns: The probability density at the frequency 'f' in the atoms rest frame,
     related to differences in kinetic energy via the proportionality constant 'xi'.
     The kinetic energies are distributed according to a convolution of the boltzmann and a normal distribution (1/[f]).
    """
    pm = 1. if col else -1.
    f, xi, sigma = np.asarray(f), np.asarray(xi), np.asarray(sigma)
    scalar_true = tools.check_shape((), f, xi, sigma, return_mode=True)
    if scalar_true:
        f = np.array([f])
    sig = (0.5 * sigma / xi) ** 2
    norm = np.exp(-0.5 * sig) / (np.sqrt(2.) * np.pi * sigma)
    mu = -0.5 * pm * f / xi - sig
    b_arg = 0.25 * mu ** 2 / sig

    nonzero = mu.astype(bool)
    mu = mu[nonzero]
    b_arg = b_arg[nonzero]
    main = np.full(f.shape, np.sqrt(LEMNISCATE * np.sqrt(sig * np.pi)))
    main_nonzero = np.empty_like(f[nonzero], dtype=float)
    mask = mu < 0.

    main_nonzero[mask] = np.sqrt(-0.5 * mu[mask]) * np.exp(-mu[mask]) \
        * np.exp(-b_arg[mask]) * sp.kv(0.25, b_arg[mask])
    main_nonzero[~mask] = 0.5 * np.pi * np.sqrt(mu[~mask]) * np.exp(-mu[~mask]) \
        * (sp.ive(0.25, b_arg[~mask]) + sp.ive(-0.25, b_arg[~mask]))
    main[nonzero] = main_nonzero
    if scalar_true:
        return main[0] * norm
    return main * norm


def source_energy_pdf(f, f0, sigma, xi, collinear=True):
    """
    :param f: Frequency quantiles (arb. units).
    :param f0: Frequency offset (arb. units).
    :param sigma: The standard deviation of the underlying normal distribution in frequency units ([f]).
    :param xi: The proportionality constant between kinetic energy differences and frequency differences ([f]).
    :param collinear:
    :returns: PDF of rest frame frequencies after acceleration of thermally and normally distributed kinetic energies.
    """
    pm = 1. if collinear else -1.
    f = np.asarray(f)
    sig = (sigma / (2. * xi)) ** 2
    _norm = np.exp(-0.5 * sig) / (sigma * np.sqrt(2. * np.pi))
    mu = -pm * (f - f0) / (2. * xi) - sig
    nonzero = mu.astype(bool)
    mu = mu[nonzero]
    b_arg = mu ** 2 / (4. * sig)
    main = np.full(f.shape, np.sqrt(LEMNISCATE * np.sqrt(sig / np.pi)))
    main_nonzero = np.empty_like(f[nonzero], dtype=float)
    mask = mu < 0.

    main_nonzero[mask] = np.sqrt(-0.5 * mu[mask] / np.pi) * np.exp(-mu[mask]) \
        * np.exp(-b_arg[mask]) * sp.kv(0.25, b_arg[mask])
    main_nonzero[~mask] = 0.5 * np.sqrt(mu[~mask] * np.pi) * np.exp(-mu[~mask]) \
        * (sp.ive(0.25, b_arg[~mask]) + sp.ive(-0.25, b_arg[~mask]))
    main[nonzero] = main_nonzero
    return main * _norm
