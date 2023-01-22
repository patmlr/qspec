# -*- coding: utf-8 -*-
"""
pycol.Physics

Created on 01.04.2020

@author: Patrick Mueller

Module including physical functions useful for CLS.
"""

import string
import numpy as np
import scipy.constants as sc
import scipy.stats as st
import scipy.special as sp

from .types import *
from . import tools


L_LABEL = ['S', 'P', 'D', ] + list(string.ascii_uppercase[5:])
E_NORM = sc.e
LEMNISCATE = 2.6220575543
mu_N = sc.physical_constants['nuclear magneton'][0]
mu_B = sc.physical_constants['Bohr magneton'][0]
g_s = sc.physical_constants['electron g factor'][0]
m_e_u = sc.physical_constants['electron mass in u'][0]
m_e_u_d = sc.physical_constants['electron mass in u'][2]
gp_s = sc.physical_constants['proton g factor'][0]
gn_s = sc.physical_constants['neutron g factor'][0]


""" Units """


def inv_cm_to_freq(k: array_like):
    """
    :param k: The wavenumber k of a transition (cm ** -1)
    :returns: The frequency corresponding to the wavenumber k (MHz).
    """
    return k * sc.c * 1e-4


def freq_to_inv_cm(freq: array_like):
    """
    :param freq: The frequency f of a transition (MHz)
    :returns: The wavenumber k corresponding to the frequency freq (MHz).
    """
    return freq / sc.c * 1e4


def wavelength_to_freq(lam: array_like):
    """
    :param lam: The wavelength lambda of a transition (um)
    :returns: The frequency corresponding to the wavelength lam (MHz).
    """
    return sc.c / lam


def freq_to_wavelength(freq: array_like):
    """
    :param freq: The frequency f of a transition (MHz)
    :returns: The wavelength corresponding to the frequency freq (MHz).
    """
    return sc.c / freq


""" 1-D kinematics """


def beta(v: array_like) -> array_like:
    """
    :param v: The velocity of a body (m/s).
    :returns: The velocity v relative to light speed.
    """
    v = np.asarray(v)
    return v / sc.c


def gamma(v: array_like) -> array_like:
    """
    :param v: The velocity of a body (m/s).
    :returns: The time-dilation/Lorentz factor corresponding to the velocity v.
    """
    return 1. / np.sqrt(1. - beta(v) ** 2)


def gamma_e(e: array_like, m: array_like) -> array_like:
    """
    :param e: The total energy of a body, including the energy of the rest mass (eV).
    :param m: The mass of the body (amu).
    :returns: The time-dilation/Lorentz factor corresponding to the total energy e of a body with mass m.
    """
    e = np.asarray(e)
    return e / e_rest(m)


def gamma_e_kin(e: array_like, m: array_like) -> array_like:
    """
    :param e: The kinetic energy of a body (eV).
    :param m: The mass of the body (amu).
    :returns: The time-dilation/Lorentz factor corresponding to the kinetic energy e of a body with mass m.
    """
    return 1. + gamma_e(e, m)


def e_rest(m: array_like) -> array_like:
    """
    :param m: The mass of a body (amu).
    :returns: The resting energy of the body with mass m (eV).
    """
    m = np.asarray(m)
    return m * sc.atomic_mass * sc.c ** 2 / E_NORM


def e_kin(v: array_like, m: array_like, relativistic=True) -> array_like:
    """
    :param v: The velocity of a body (m/s).
    :param m: The mass of the body (amu).
    :param relativistic: The calculation is performed either relativistically or classically.
    :returns: The kinetic energy of a body with velocity v and mass m (eV).
    """
    if relativistic:
        return (gamma(v) - 1.) * e_rest(m)
    else:
        v, m = np.asarray(v), np.asarray(m)
        return m * sc.atomic_mass * v ** 2 / 2. / E_NORM


def e_total(v: array_like, m: array_like) -> array_like:
    """
    :param v: The velocity of a body (m/s).
    :param m: The mass of the body (amu).
    :returns: The total energy of a body with velocity v and mass m (eV). """
    return gamma(v) * e_rest(m)


def e_el(u: array_like, q: array_like) -> array_like:
    """
    :param u: The electric potential difference (V).
    :param q: The charge of a body (e).
    :returns: The potential energy difference of a body with charge q inside an electric potential with voltage u (eV).
    """
    q, u = np.asarray(q), np.asarray(u)
    return q * u


def v_e(e: array_like, m: array_like, v0: array_like = 0, relativistic=True) -> array_like:
    """
    :param e: Energy which is added to the kinetic energy of a body with velocity v0 (eV).
    :param m: The mass of the body (amu).
    :param v0: The initial velocity of the body (m/s).
    :param relativistic: The calculation is performed either relativistically or classically.
    :returns: The velocity of a body with mass m and velocity v0 after the addition of the kinetic energy e (m/s).
    """
    if relativistic:
        return sc.c * np.sqrt(1. - (1. / (gamma(v0) + gamma_e(e, m))) ** 2)
    else:
        v0, e, m = np.asarray(v0), np.asarray(e), np.asarray(m)
        return np.sqrt(v0 ** 2 + 2. * e * E_NORM / (m * sc.atomic_mass))


def v_e_d1(e: array_like, m: array_like, v0: array_like = 0, relativistic=True) -> array_like:
    """
    :param e: Energy which is added to the kinetic energy of a body with velocity v0 (eV).
    :param m: The mass of the body (amu).
    :param v0: The initial velocity of the body (m/s).
    :param relativistic: The calculation is performed either relativistically or classically.
    :returns: The first derivative of 'v_e' regarding 'e' (m/(s eV)).
    """
    m = np.asarray(m)
    dv = 1. / (m * sc.atomic_mass * v_e(e, m, v0=v0, relativistic=relativistic))
    if relativistic:
        dv /= (gamma(v0) + gamma_e(e, m)) ** 3
    return dv * E_NORM


def v_el(u: array_like, q: array_like, m: array_like, v0: array_like = 0, relativistic=True) -> array_like:
    """
    :param u: The electric potential difference added to the kinetic energy of a body with velocity v0 (V).
    :param q: The charge of a body (e).
    :param m: The mass of the body (amu).
    :param v0: The initial velocity of the body (m/s).
    :param relativistic: The calculation is performed either relativistically or classically.
    :returns: The velocity of a body with starting velocity v0, charge q and mass m
     after electrostatic acceleration with voltage u (m/s).
    """
    return v_e(e_el(u, q), m, v0=v0, relativistic=relativistic)


def v_el_d1(u: array_like, q: array_like, m: array_like, v0: array_like = 0, relativistic=True) -> array_like:
    """
    :param u: The electric potential difference added to the kinetic energy of a body with velocity v0 (V).
    :param q: The charge of a body (e).
    :param m: The mass of the body (amu).
    :param v0: The initial velocity of the body (m/s).
    :param relativistic: The calculation is performed either relativistically or classically.
    :returns: The first derivative of 'v_el' regarding 'u' (m/(s V)).
    """
    return v_e_d1(e_el(u, q), m, v0=v0, relativistic=relativistic) * q


def p_v(v: array_like, m: array_like, relativistic=True) -> array_like:
    """
    :param v: The velocity of a body (m/s).
    :param m: The mass of the body (amu).
    :param relativistic: The calculation is performed either relativistically or classically.
    :returns: The momentum of a body with velocity v and mass m.
    """
    v, m = np.asarray(v), np.asarray(m)
    if relativistic:
        return gamma(v) * m * sc.atomic_mass * v
    else:
        return m * sc.atomic_mass * v


def p_e(e: array_like, m: array_like, p0: array_like = 0, relativistic=True) -> array_like:
    """
    :param e: Energy which is added to the kinetic energy of a body with velocity v0 (eV).
    :param m: The mass of the body (amu).
    :param p0: The initial momentum of the body (amu m/s).
    :param relativistic: The calculation is performed either relativistically or classically.
    :returns: The momentum of a body with starting momentum p0 and mass m
     after the addition of the kinetic energy e (amu m/s).
    """
    e, p0 = np.asarray(e), np.asarray(p0)
    if relativistic:
        pc_square = (p0 * sc.c) ** 2 / E_NORM
        return np.sqrt(e ** 2 + pc_square + 2 * e * np.sqrt(pc_square + e_rest(m))) / sc.c
    else:
        m = np.asarray(m)
        return np.sqrt(p0 ** 2 + 2 * m * sc.atomic_mass * e * E_NORM)


def p_el(u: array_like, q: array_like, m: array_like, p0: array_like = 0, relativistic=True) -> array_like:
    """
    :param u: The electric potential difference added to the kinetic energy of a body with velocity v0 (V).
    :param q: The charge of a body (e).
    :param m: The mass of the body (amu).
    :param p0: The initial momentum of the body (amu m/s).
    :param relativistic: The calculation is performed either relativistically or classically.
    :returns: The momentum of a body with starting momentum p0, charge q and mass m
     after electrostatic acceleration with voltage u.
    """
    return p_e(e_el(u, q), m, p0, relativistic=relativistic)


def doppler(f: array_like, v: array_like, alpha: array_like, return_frame='atom') -> array_like:
    """
    :param f: The frequency of light (arb. units).
    :param v: The velocity of a body (m/s).
    :param alpha: The angle between the velocity- and the wave-vector in the laboratory frame (rad).
    :param return_frame: The coordinate system in which the frequency is returned. Can be either 'atom' or 'lab'.
    :returns: the Doppler-shifted frequency in either the rest frame of the atom or the laboratory frame ([f]).
    :raises ValueError: rest_frame must be either 'atom' or 'lab'.
    """
    f, alpha = np.asarray(f), np.asarray(alpha)
    if return_frame == 'atom':
        """ Return freq in the atomic system, alpha=0 == Col, alpha in laboratory system """
        return f * gamma(v) * (1. - beta(v) * np.cos(alpha))
    elif return_frame == 'lab':
        """ Return freq in the laboratory system, alpha=0 == Col, alpha in laboratory system """
        return f / (gamma(v) * (1. - beta(v) * np.cos(alpha)))
    else:
        raise ValueError('rest_frame must be either "atom" or "lab".')


def doppler_d1(f: array_like, v: array_like, alpha: array_like, return_frame='atom') -> array_like:
    """
    :param f: The frequency of light (arb. units).
    :param v: The velocity of a body (m/s).
    :param alpha: The angle between the velocity- and the wave-vector in the laboratory frame (rad).
    :param return_frame: The coordinate system in which the frequency is returned. Can be either 'atom' or 'lab'.
    :returns: the first derivative of the 'doppler' formula regarding 'v' ([f] s/m).
    :raises ValueError: rest_frame must be either 'atom' or 'lab'.
    """
    if return_frame == 'atom':
        """ Return df/dv in the atomic system, alpha=0 == Col, alpha in laboratory system. """
        f, alpha = np.asarray(f), np.asarray(alpha)
        return f * gamma(v) ** 3 * (beta(v) - np.cos(alpha)) / sc.c
    elif return_frame == 'lab':
        """ Return df/dv in the laboratory system, alpha=0 == Col, alpha in laboratory system. """
        f_lab = doppler(f, v, alpha, return_frame='lab')
        return -f_lab / f * doppler_d1(f_lab, v, alpha, return_frame='atom')
    else:
        raise ValueError('rest_frame must be either "atom" or "lab".')


def doppler_e_d1(f: array_like, alpha: array_like, e: array_like, m: array_like,
                 v0: array_like = 0, return_frame='atom', relativistic=True) -> array_like:
    """
    :param f: The frequency of light (arb. units).
    :param alpha: The angle between the velocity- and the wave-vector in the laboratory frame (rad).
    :param e: Energy which is added to the kinetic energy of a body with velocity v0 (eV).
    :param m: The mass of the body (amu).
    :param v0: The initial velocity of the body (m/s).
    :param return_frame: The coordinate system in which the frequency is returned. Can be either 'atom' or 'lab'.
    :param relativistic: The calculation is performed either relativistically or classically.
    :returns: the first derivative of the 'doppler' formula regarding 'e' ([f]/eV).
    :raises ValueError: rest_frame must be either 'atom' or 'lab'.
    """
    v = v_e(e, m, v0=v0, relativistic=relativistic)
    return doppler_d1(f, v, alpha, return_frame=return_frame) * v_e_d1(e, m, v0=v0, relativistic=relativistic)


def doppler_el_d1(f: array_like, alpha: array_like, u: array_like, q: array_like, m: array_like,
                  v0: array_like = 0., return_frame='atom', relativistic=True) -> array_like:
    """
    :param f: The frequency of light (arb. units).
    :param alpha: The angle between the velocity- and the wave-vector in the laboratory frame (rad).
    :param u: The electric potential difference added to the kinetic energy of a body with velocity v0 (V).
    :param q: The charge of a body (e).
    :param m: The mass of the body (amu).
    :param v0: The initial velocity of the body (m/s).
    :param return_frame: The coordinate system in which the frequency is returned. Can be either 'atom' or 'lab'.
    :param relativistic: The calculation is performed either relativistically or classically.
    :returns: the first derivative of the 'doppler' formula regarding 'u' ([f]/V).
    :raises ValueError: rest_frame must be either 'atom' or 'lab'.
    """
    v = v_el(u, q, m, v0=v0, relativistic=relativistic)
    return doppler_d1(f, v, alpha, return_frame=return_frame) * v_el_d1(u, q, m, v0=v0, relativistic=relativistic)


def inverse_doppler(f_atom: array_like, f_lab: array_like, alpha: array_like,
                    mode='raise-raise', return_mask=False) -> array_like:
    """
    :param f_lab: The frequency of light in the laboratory frame (arb. units).
    :param f_atom: The frequency of light in the atoms rest frame ([f_lab]).
    :param alpha: The angle between the velocity- and the wave-vector in the laboratory frame (rad).
    :param mode: The mode how to handle numpy.nans and ambiguous velocities. Available options are:
        * 'raise-raise': Raise an error if there are numpy.nans and if the velocity is ambiguous.
        * 'raise-small': Raise an error if there are numpy.nans and return the smaller velocity.
        * 'raise-large': Raise an error if there are numpy.nans and return the larger velocity.
        * 'isnan-raise': Ignore numpy.nans and raise an error if the velocity is ambiguous.
        * 'isnan-small': Ignore numpy.nans and return the smaller velocity.
        * 'isnan-large': Ignore numpy.nans and return the larger velocity.
    :param return_mask: Whether the mask where the velocity is ambiguous is returned as a second argument.
    :returns: the velocity required to shift f_lab to f_atom (m/s)
     and optionally the mask where the velocity is ambiguous.
    """
    modes = ['raise-raise', 'raise-small', 'raise-large', 'isnan-raise', 'isnan-small', 'isnan-large']
    if mode not in modes:
        raise ValueError('mode must be in {}.'.format(modes))
    f_lab, f_atom, alpha = np.asarray(f_lab), np.asarray(f_atom), np.asarray(alpha)
    scalar_true = tools.check_shape((), f_lab, f_atom, alpha, return_mode=True)
    if scalar_true:
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
        # print('WARNING: Situation is physically impossible for at least one argument.')
    if np.any(ambiguous):
        if mode[6:] == 'raise':
            raise ValueError('Situation allows two different velocities.')
        # print('WARNING: Situation allows two different velocities. Taking the {} velocity.'.format(mode[6:]))
    if return_mask:
        if scalar_true:
            return (bet * sc.c)[0], ambiguous
        return bet * sc.c, ambiguous
    if scalar_true:
        return (bet * sc.c)[0]
    return bet * sc.c


def inverse_doppler_d1(f_atom: array_like, f_lab: array_like, alpha: array_like,
                       mode='raise-raise', return_mask=False) -> array_like:
    """
    :param f_lab: The frequency of light in the laboratory frame (arb. units).
    :param f_atom: The frequency of light in the atoms rest frame ([f_lab]).
    :param alpha: The angle between the velocity- and the wave-vector in the laboratory frame (rad).
    :param mode: The mode how to handle numpy.nans and ambiguous velocities. Available options are:
        * 'raise-raise': Raise an error if there are numpy.nans and if the velocity is ambiguous.
        * 'raise-small': Raise an error if there are numpy.nans and return the smaller velocity.
        * 'raise-large': Raise an error if there are numpy.nans and return the larger velocity.
        * 'isnan-raise': Ignore numpy.nans and raise an error if the velocity is ambiguous.
        * 'isnan-small': Ignore numpy.nans and return the smaller velocity.
        * 'isnan-large': Ignore numpy.nans and return the larger velocity.
    :param return_mask: Whether the mask where the velocity is ambiguous is returned as a second argument.
    :returns: the first derivative of the 'inverse_doppler' formula regarding f_atom (m/(s MHz)).
    """
    f_lab, f_atom, alpha = np.asarray(f_lab), np.asarray(f_atom), np.asarray(alpha)
    scalar_true = tools.check_shape((), f_lab, f_atom, alpha, return_mode=True)
    if scalar_true:
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
    if return_mask:
        if scalar_true:
            return (bet * sc.c / (f_lab * square_sum))[0], ambiguous
        return bet * sc.c / (f_lab * square_sum), ambiguous
    if scalar_true:
        return (bet * sc.c / (f_lab * square_sum))[0]
    return bet * sc.c / (f_lab * square_sum)


def alpha_atom(alpha: array_like, v: array_like) -> array_like:
    """
    :param alpha: The angle between a velocity- and a wave-vector in the laboratory frame (rad).
    :param v: The velocity of a body (m/s).
    :returns: The angle between the velocity- and the wave-vector in the atoms rest frame (rad).
    """
    alpha = np.asarray(alpha)
    cos = np.cos(alpha)
    arg = (beta(v) + cos) / (1. + beta(v) * cos)
    return np.arccos(arg)


def v_rec(f, m) -> array_like:
    """
    :param f: The frequency of light in the atoms rest frame (MHz).
    :param m: The mass of a body (amu).
    :returns: The change of velocity of an atom at rest
     due to the absorption of a photon with frequency f (m/s).
    """
    f, m = np.asarray(f), np.asarray(m)
    return sc.h * f / (m * sc.atomic_mass * sc.c) * 1e6


def photon_recoil(f: array_like, m: array_like) -> array_like:
    """
    :param f: The frequency of light in the atoms rest frame (MHz).
    :param m: The mass of a body (amu).
    :returns: The change of a transition frequency of an atom at rest
     due to the absorption of a photon with frequency f (MHz).
    """
    f, m = np.asarray(f), np.asarray(m)
    return (sc.h * (f * 1e6) ** 2) / (2 * m * sc.atomic_mass * sc.c ** 2) * 1e-6


def photon_recoil_v(v: array_like, alpha: array_like, f_lab: array_like, m: array_like) -> array_like:
    """
    :param v: The velocity of a body (m/s).
    :param alpha: The angle between a velocity- and a wave-vector in the laboratory frame (rad).
    :param f_lab: The frequency of light in the laboratory frame (MHz).
    :param m: The mass of a body (amu).
    :returns: The change of a transition frequency of an atom moving with velocity v
     due to the absorption of a laser photon with frequency f (MHz).
    """
    f = doppler(f_lab, v, alpha, return_frame='atom')
    return photon_recoil(f, m)


""" Atomic physics """


def get_f(i: scalar, j: scalar):
    """
    :param i: The nuclear spin quantum number I.
    :param j: The electronic total angular momentum quantum number J.
    :returns: All possible f quantum numbers.
    """
    return [k + abs(i - j) for k in range(int(i + j + 1 - abs(i - j)))]


def get_m(f: scalar):
    """
    :param f: The total angular momentum quantum number F (= J if I == 0).
    :returns: All possible zeeman substates of the specified quantum number.
    """
    return [k - f for k in range(int(2 * f + 1))]


def hyperfine(i: float, j: float, f: float, a: array_like, b: array_like = None) -> array_like:
    """
    :param i: The nuclear spin quantum number I.
    :param j: The electronic total angular momentum quantum number J.
    :param f: The total angular momentum quantum number F.
    :param a: The magnetic dipole hyperfine constant A (arb. units).
    :param b: The electric quadrupole hyperfine constant B ([a]).
    :returns: The hyperfine shift of an atomic state (i, j, f) with hyperfine constants a and b ([a]).
    """
    a = np.asarray(a)
    if b is None:
        b = np.zeros_like(a)
    if i < 0. or j < 0. or f < 0.:
        print('Either i >= 0, j >= 0 or f >= 0 is not fulfilled.')
        raise ValueError
    if f < abs(i - j) or f > i + j:
        print('|i - j| <= f <= i + j must be fulfilled.')
        raise ValueError
    if i == 0. or j == 0.:
        return 0.
    k = f * (f + 1) - i * (i + 1) - j * (j + 1)
    shift = a * k / 2
    if i > 0.5 and j > 0.5:
        k_2 = 3 * k * (k + 1) / 2 - 2 * i * (i + 1) * j * (j + 1)
        k_2 /= i * (2 * i - 1) * j * (2 * j - 1)
        shift += b * k_2 / 4
    return shift


def lande_n(gyro: array_like) -> array_like:
    """
    :param gyro: The gyromagnetic ratio (MHz).
    :returns: The nuclear g-factor.
    """
    gyro = np.asarray(gyro)
    return gyro * sc.h / mu_N


def lande_j(s: float, ll: float, j: float, approx_g_s: bool = False) -> float:
    """
    :param s: The electron spin quantum number S.
    :param ll: The electronic angular momentum quantum number L.
    :param j: The electronic total angular momentum quantum number J.
    :param approx_g_s: Whether to use g_s = -2 or the QED result g_s = -2.0023... .
    :returns: The electronic g-factor.
    """
    if j == 0:
        return 0.
    g = -2 if approx_g_s else g_s
    jj = j * (j + 1)
    ls = ll * (ll + 1) - s * (s + 1)
    val = -(jj + ls) / (2 * jj)
    val += (jj - ls) / (2 * jj) * g
    return val


def lande_f(i: float, j: float, f: float, g_n: float, g_j: float) -> float:
    """
    :param i: The nuclear spin quantum number I.
    :param j: The electronic total angular momentum quantum number J.
    :param f: The total angular momentum quantum number F.
    :param g_n: The nuclear g-factor.
    :param g_j: The electronic g-factor.
    :returns: The hyperfine structure g-factor.
    """
    ff = f * (f + 1.)
    ji = j * (j + 1.) - i * (i + 1.)
    val = (ff + ji) / (2 * ff) * g_j
    val += (ff - ji) / (2 * ff) * g_n * mu_N / mu_B
    return val


def zeeman(m: float, b: array_like, g: float, as_freq=True) -> array_like:
    """
    :param m: The B-field-axis component quantum number m of the total angular momentum.
    :param b: The B-field (T).
    :param g: The g-factor.
    :param as_freq: The zeeman shift can be returned in energy or frequency units.
    :returns: The energy shift of an atomic state due to the zeeman effect in energy or frequency units (eV or MHz).
    """
    b = np.asarray(b)
    delta = -g * m * mu_B * b / E_NORM
    if as_freq:
        delta /= sc.h * 1e6 / E_NORM
    return delta


def hyper_zeeman(i: float, s: float, ll: float, j: float, f: float, m: float, g_n: float,
                 a_hyper: array_like, b_hyper: array_like, b: array_like,
                 g_n_as_gyro: bool = False, as_freq: bool = True) -> array_like:
    """
    :param i: The nuclear spin quantum number I.
    :param s: The electron spin quantum number S.
    :param ll: The electronic angular momentum quantum number L.
    :param j: The electronic total angular momentum quantum number J.
    :param f: The total angular momentum quantum number F.
    :param m: The B-field-axis component quantum number m of the total angular momentum.
    :param g_n: The nuclear g-factor or the gyromagnetic ratio if g_n_as_gyro == True.
    :param a_hyper: The magnetic dipole hyperfine constant A (eV or MHz).
    :param b_hyper: The electric quadrupole hyperfine constant B ([a]).
    :param b: The B-field (T).
    :param g_n_as_gyro: Whether g_n is the nuclear g-factor or the gyromagnetic ratio.
    :param as_freq: The shift can be returned in energy or frequency units.
    :returns: The total energy shift of an atomic state due to the hyperfine splitting
     and the zeeman effect in energy or frequency units (eV or MHz).
    """
    a_hyper, b_hyper, b = np.asarray(a_hyper), np.asarray(b_hyper), np.asarray(b)
    g_j = lande_j(s, ll, j)
    g_f = lande_f(i, j, f, lande_n(g_n) if g_n_as_gyro else g_n, g_j) if i != 0 or j != 0 else 0.
    shift = hyperfine(i, j, f, a_hyper, b_hyper)
    shift += zeeman(m, b, g_f, as_freq=as_freq)
    return shift


def a_hyper_mu(i: scalar, j: scalar, mu: array_like, b: array_like):
    """
    :param i: The nuclear spin quantum number I.
    :param j: The electronic total angular momentum quantum number J.
    :param mu: The magnetic moment of the nucleus in units of the nuclear magneton (mu_N).
    :param b: The B-field of the atomic electrons at the nucleus (T).
    :returns: The hyperfine structure constant A (MHz).
    """
    mu, b = np.asarray(mu), np.asarray(b)
    if i == 0 or j == 0:
        return np.zeros_like(mu * b)
    return mu * b / np.sqrt(i * (i + 1) * j * (j + 1)) / sc.h


def saturation_intensity(f: array_like, a: array_like, a_dipole: array_like):
    """
    :param f: The frequency of the transition.
    :param a: The Einstein A coefficient (MHz).
    :param a_dipole: The reduced dipole coefficient of the transition (see algebra.a_dipole).
    :returns: The saturation intensity.
    """
    f, a, a_dipole = np.asarray(f), np.asarray(a), np.asarray(a_dipole)
    return np.pi * f ** 3 * sc.h * a / (3 * sc.c ** 2 * a_dipole) * 1e24


def saturation(i: array_like, f: array_like, a: array_like, a_dipole: array_like):
    """
    :param i: The intensity of the laser.
    :param f: The frequency of the transition.
    :param a: The Einstein A coefficient (MHz).
    :param a_dipole: The reduced dipole coefficient of the transition (see algebra.a_dipole).
    :returns: The saturation parameter.
    """
    i, f, a, a_dipole = np.asarray(i), np.asarray(f), np.asarray(a), np.asarray(a_dipole)
    i0 = np.pi * f ** 3 * sc.h * a / (3 * sc.c ** 2 * a_dipole) * 1e24
    return i / i0


def rabi(a: array_like, s: array_like):
    """
    :param a: The Einstein A coefficient (MHz).
    :param s: The saturation parameter.
    :returns: The rabi frequency.
    """
    a, s = np.asarray(a), np.asarray(s)
    return a * np.sqrt(s / 2.)


def scattering_rate(df: array_like, a: array_like, s: array_like):
    """
    :param df: The detuning of to be scattered light from the transition.
     This must be differences of real frequencies, such that w = 2 pi * df (MHz).
    :param a: The Einstein A coefficient (MHz).
    :param s: The saturation parameter.
    :returns: The 2-state-equilibrium scattering-rate of an electronic transition.
    """
    df, a, s = np.asarray(df), np.asarray(a), np.asarray(s)
    return 0.125 * s * a ** 3 / (0.25 * (1 + s) * a ** 2 + (2 * np.pi * df) ** 2)


def mass_factor(m: array_like, m_ref: array_like, m_d: array_like = 0, m_ref_d: array_like = 0, k_inf: bool = True) \
        -> (ndarray, ndarray):
    """
    :param m: The mass of the isotope (amu).
    :param m_ref: The mass of the reference isotope (amu). Must be a scalar or have the same shape as 'm'.
    :param m_d: The uncertainty of the mass of the isotope (amu). Must be a scalar or have the same shape as 'm'.
    :param m_ref_d: The uncertainty of the mass of the reference isotope (amu).
     Must be a scalar or have the same shape as 'm'.
    :param k_inf: Whether the normal mass-shift factor K(NMS) is defined mass independently
     as m_e * T(inf) (= True) or as m_e * T(A_ref) (= False). Compare (6.4) with (3.17)
     in [W. H. King, Isotope shifts in atomic spectra (1984)].
    :returns: the mass factor and its uncertainty needed to calculate modified isotope shifts or charge radii.
    """
    m, m_d, m_ref, m_ref_d = np.asarray(m), np.asarray(m_d), np.asarray(m_ref), np.asarray(m_ref_d)
    if k_inf:
        mu = (m + m_e_u) * (m_ref + m_e_u) / (m - m_ref)
        if np.all(m_d) == 0 and np.all(m_ref_d) == 0:
            return mu, np.zeros_like(mu)
        mu_d = ((mu / (m + m_e_u) - mu / (m - m_ref)) * m_d) ** 2
        mu_d += ((mu / (m_ref + m_e_u) + mu / (m - m_ref)) * m_ref_d) ** 2
        mu_d += ((mu / (m + m_e_u) + mu / (m_ref + m_e_u)) * m_e_u_d) ** 2
    else:
        mu = (m + m_e_u) * m_ref / (m - m_ref)
        if np.all(m_d) == 0 and np.all(m_ref_d) == 0:
            return mu, np.zeros_like(mu)
        mu_d = (-m_ref * (m_ref + m_e_u) / ((m - m_ref) ** 2) * m_d) ** 2
        mu_d += (m * (m + m_e_u) / ((m - m_ref) ** 2) * m_ref_d) ** 2
        mu_d += (m_ref / (m - m_ref) * m_e_u_d) ** 2
    return mu, np.sqrt(mu_d)


def delta_r2(r: array_like, r_d: array_like, r_ref: array_like, r_ref_d: array_like,
             delta_r: array_like, delta_r_d: array_like, v2: array_like, v2_ref: array_like):
    """
    :param r: The Barrett radius of an isotope.
    :param r_d: The uncertainty of the Barrett radius.
    :param r_ref: The Barrett radius of a reference isotope.
    :param r_ref_d: The uncertainty of the Barrett radius of the reference isotope.
    :param delta_r: The difference between the Barrett radius of the isotope and the reference isotope.
    :param delta_r_d: The uncertainty of the difference between
     the Barrett radius of the isotope and the reference isotope.
    :param v2: The V2 factor of the isotope.
    :param v2_ref: The V2 factor of the reference isotope.
    :returns: The difference of the mean square nuclear charge radius between two isotopes and its uncertainty.
    """
    r, r_d = np.asarray(r, dtype=float), np.asarray(r_d, dtype=float)
    r_ref, r_ref_d = np.asarray(r_ref, dtype=float), np.asarray(r_ref_d, dtype=float)
    delta_r, delta_r_d = np.asarray(delta_r, dtype=float), np.asarray(delta_r_d, dtype=float)
    v2, v2_ref = np.asarray(v2, dtype=float), np.asarray(v2_ref, dtype=float)

    sum_term = (r / v2 + r_ref / v2_ref) / v2
    delta_term = delta_r + r_ref * (1. - v2 / v2_ref)
    val = sum_term * delta_term  # (r/v2)**2 - (r_ref/v2_ref)**2

    err = (sum_term * delta_r_d) ** 2
    err += (delta_term * r_d / (v2 ** 2)) ** 2
    err += ((delta_term / (v2 * v2_ref) + sum_term * (1. - v2 / v2_ref)) * r_ref_d) ** 2
    return val, np.sqrt(err)


def delta_r4(r: array_like, r_d: array_like, r_ref: array_like, r_ref_d: array_like,
             delta_r: array_like, delta_r_d: array_like, v4: array_like, v4_ref: array_like):
    """
    :param r: The Barrett radius of an isotope.
    :param r_d: The uncertainty of the Barrett radius.
    :param r_ref: The Barrett radius of a reference isotope.
    :param r_ref_d: The uncertainty of the Barrett radius of the reference isotope.
    :param delta_r: The difference between the Barrett radius of the isotope and the reference isotope.
    :param delta_r_d: The uncertainty of the difference between
     the Barrett radius of the isotope and the reference isotope.
    :param v4: The V4 factor of the isotope.
    :param v4_ref: The V4 factor of the reference isotope.
    :returns: The difference of the mean quartic nuclear charge radius between two isotopes and its uncertainty.
    """
    r, r_d = np.asarray(r, dtype=float), np.asarray(r_d, dtype=float)
    r_ref, r_ref_d = np.asarray(r_ref, dtype=float), np.asarray(r_ref_d, dtype=float)
    delta_r, delta_r_d = np.asarray(delta_r, dtype=float), np.asarray(delta_r_d, dtype=float)
    v4, v4_ref = np.asarray(v4, dtype=float), np.asarray(v4_ref, dtype=float)

    sum_term = (r / v4) ** 2 + (r_ref / v4_ref) ** 2
    delta_term = delta_r2(r, r_d, r_ref, r_ref_d, delta_r, delta_r_d, v4, v4_ref)
    val = sum_term * delta_term[0]  # (r/v4)**4 - (r_ref/v4_ref)**4

    err = (sum_term * delta_term[1]) ** 2
    err += (2. * delta_term[0] * r * r_d / (v4 ** 2)) ** 2
    err += (2. * delta_term[0] * r_ref * r_ref_d / (v4_ref ** 2)) ** 2
    return val, np.sqrt(err)


def delta_r6(r: array_like, r_d: array_like, r_ref: array_like, r_ref_d: array_like,
             delta_r: array_like, delta_r_d: array_like, v6: array_like, v6_ref: array_like):
    """
    :param r: The Barrett radius of an isotope.
    :param r_d: The uncertainty of the Barrett radius.
    :param r_ref: The Barrett radius of a reference isotope.
    :param r_ref_d: The uncertainty of the Barrett radius of the reference isotope.
    :param delta_r: The difference between the Barrett radius of the isotope and the reference isotope.
    :param delta_r_d: The uncertainty of the difference between
     the Barrett radius of the isotope and the reference isotope.
    :param v6: The V6 factor of the isotope.
    :param v6_ref: The V6 factor of the reference isotope.
    :returns: The difference of the mean sextic nuclear charge radius between two isotopes and its uncertainty.
    """
    r, r_d = np.asarray(r, dtype=float), np.asarray(r_d, dtype=float)
    r_ref, r_ref_d = np.asarray(r_ref, dtype=float), np.asarray(r_ref_d, dtype=float)
    delta_r, delta_r_d = np.asarray(delta_r, dtype=float), np.asarray(delta_r_d, dtype=float)
    v6, v6_ref = np.asarray(v6, dtype=float), np.asarray(v6_ref, dtype=float)

    sum_term = (v6 / r) * ((r / v6) ** 3 + (r_ref / v6_ref) ** 3)
    delta = delta_r4(r, r_d, r_ref, r_ref_d, delta_r, delta_r_d, v6, v6_ref)
    delta_term = delta[0] + (r_ref / v6_ref) ** 4 * (1. - (r / v6) * (v6_ref / r_ref))
    val = sum_term * delta_term  # (r/v6)**6 - (r_ref/v6_ref)**6

    err = (sum_term * delta[1]) ** 2
    err += ((-(r_ref / v6_ref) ** 3 * sum_term / v6
             + delta_term * (-sum_term / r + 3. * r / (v6 ** 2))) * r_d) ** 2
    err += (((4 * r_ref ** 3 / (v6_ref ** 4) * (1. - (r / v6) * (v6_ref / r_ref))
              + (r / v6) * r_ref ** 2 / (v6_ref ** 3)) * sum_term
             + delta_term * 3. * (v6 / r) * r_ref ** 2 / (v6_ref ** 3)) * r_ref_d) ** 2
    return val, np.sqrt(err)


def lambda_r(r: float, r_d: float, r_ref: float, r_ref_d: float, delta_r: float, delta_r_d: float,
             v2: float, v2_ref: float, v4: float, v4_ref: float, v6: float, v6_ref: float, c2c1: float, c3c1: float):
    """
    :param r: The Barrett radius of an isotope.
    :param r_d: The uncertainty of the Barrett radius.
    :param r_ref: The Barrett radius of a reference isotope.
    :param r_ref_d: The uncertainty of the Barrett radius of the reference isotope.
    :param delta_r: The difference between the Barrett radius of the isotope and the reference isotope.
    :param delta_r_d: The uncertainty of the difference between
     the Barrett radius of the isotope and the reference isotope.
    :param v2: The V2 factor of the isotope.
    :param v2_ref: The V2 factor of the reference isotope.
    :param v4: The V4 factor of the isotope.
    :param v4_ref: The V4 factor of the reference isotope.
    :param v6: The V6 factor of the isotope.
    :param v6_ref: The V6 factor of the reference isotope.
    :param c2c1: Seltzer's coefficient for the quartic moment.
    :param c3c1: Seltzer's coefficient for the sextic moment.
    :returns: The difference of the mean sextic nuclear charge radius between two isotopes and its uncertainty.
    """
    r2 = delta_r2(r, r_d, r_ref, r_ref_d, delta_r, delta_r_d, v2, v2_ref)
    r4 = delta_r4(r, r_d, r_ref, r_ref_d, delta_r, delta_r_d, v4, v4_ref)
    r6 = delta_r6(r, r_d, r_ref, r_ref_d, delta_r, delta_r_d, v6, v6_ref)
    return lambda_rn(r2[0], r2[1], r4[0], r4[1], r6[0], r6[1], c2c1, c3c1)


def lambda_rn(r_2: float, r_2_d: float, r_4: float, r_4_d: float, r_6: float, r_6_d: float, c2c1: float, c3c1: float):
    """
    :param r_2: The difference of the mean square nuclear charge radius between two isotopes.
    :param r_2_d: The uncertainty of the difference of the mean square nuclear charge radius.
    :param r_4: The difference of the mean quartic nuclear charge radius between two isotopes.
    :param r_4_d: The uncertainty of the difference of the mean quartic nuclear charge radius.
    :param r_6: The difference of the mean sextic nuclear charge radius between two isotopes.
    :param r_6_d: The uncertainty of the difference of the mean sextic nuclear charge radius.
    :param c2c1: Seltzer's coefficient for the quartic moment.
    :param c3c1: Seltzer's coefficient for the sextic moment.
    :returns: the Lambda observable for the given differences in mean square, quartic and sextic nuclear charge radii
     and its uncertainty.
    """
    val = r_2 + c2c1 * r_4 + c3c1 * r_6
    err = r_2_d ** 2
    err += (c2c1 * r_4_d) ** 2
    err += (c3c1 * r_6_d) ** 2
    return val, np.sqrt(err)


def schmidt_line(ll, j, is_proton):
    _g_s = gp_s if is_proton else gn_s
    _g_l = 1 if is_proton else 0
    if j < ll:
        return j / (j + 1) * ((ll + 1) * _g_l - 0.5 * _g_s)
    return ll * _g_l + 0.5 * _g_s


""" Optics """


def sellmeier(w: array_like, a: array_iter, b: array_iter):
    """
    :param w: The wavelength in µm.
    :param a: The a coefficients.
    :param b: The b coefficients.
    :return: The index of refraction for the wavelength w and the given material.
    """
    a, b = np.asarray(a), np.asarray(b)
    tools.check_dimension(a.shape[0], 0, b)
    sum_term = np.sum([a_i * w ** 2 / (w ** 2 - b_i) for a_i, b_i in zip(a, b)], axis=0)
    return np.sqrt(1 + sum_term)


""" 3-D kinematics """


def gamma_3d(v: array_like, axis=-1) -> array_like:
    """
    :param v: The velocity 3-vector (m/s).
    :param axis: The axis along which the vector components are aligned.
    :returns: The time-dilation/Lorentz factor corresponding to the velocity vector v.
    :raises ValueError: v must have 3 components along the specified axis.
    """
    tools.check_dimension(3, axis, v)
    return gamma(tools.absolute(v, axis=axis))


def boost(x: array_like, v: array_like, axis=-1) -> array_like:
    """
    :param x: The 4-vector x in the current rest frame (arb. units).
    :param v: The velocity 3-vector (m/s).
    :param axis: The axis along which the vector components are aligned.
    :returns: The 4-vector x in the coordinate system moving with velocity v relative to the current rest frame ([x]).
    :raises ValueError: x and v must have 4 and 3 components along the specified axis, respectively.
     The shapes of x and v must be compatible.
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


def doppler_3d(k: array_like, v: array_like, return_frame='atom', axis=-1) -> array_like:
    """
    :param k: The k-wave-3-vector of light (arb. units).
    :param v: The velocity 3-vector (m/s).
    :param return_frame: The coordinate system in which the frequency is returned. Can be either 'atom' or 'lab'.
    :param axis: The axis along which the vector components are aligned.
    :returns: the Doppler-shifted k-wave-4-vector in either the rest frame of the atom or the laboratory frame ([k]).
    :raises ValueError: rest_frame must be either 'atom' or 'lab'. The shapes of k and v must be compatible.
    """
    k, v = np.asarray(k), np.asarray(v)
    tools.check_dimension(3, axis, k, v)
    k_0 = tools.absolute(k, axis=axis)
    k_4 = np.concatenate([np.expand_dims(k_0, axis=axis), k], axis=axis)
    if return_frame == 'atom':
        """ Return k in the atomic system. """
        return boost(k_4, v)
    elif return_frame == 'lab':
        """ Return k in the laboratory system. """
        return boost(k_4, -v)
    else:
        raise ValueError('rest_frame must be either "atom" or "lab".')


def gaussian_beam_3d(r: array_like, k: array_like, w0: array_like,
                     r0: array_like = None, p0: array_like = None, axis: int = -1) -> array_like:
    """
    :param r: The position 3-vector where to calculate the beam intensity (m).
    :param k: The k-wave-3-vector of light (rad / m).
    :param w0: The beam waist (m).
    :param r0: The position 3-vector of the beam waist. Is (0m, 0m, 0m) if r0 is not specified (m).
    :param p0: The total power propagated by the gaussian beam. Is 1W if p0 is not specified (W).
    :param axis: The axis along which the vector components are aligned.
    :returns: The intensity of a gaussian beam with k-wave-vector k at the position r - r0 (W/m**2 == µW/mm**2).
    :raises ValueError: r, k and r0 must have 3 components along the specified axis.
     The shapes of r, k, w0, r0 and p0 must be compatible.
    """
    if r0 is None:
        r0 = np.zeros_like(r)
    if p0 is None:
        p0 = np.ones_like(w0)
    r, r0, k = np.asarray(r, dtype=float), np.asarray(r0, dtype=float), np.asarray(k, dtype=float)
    tools.check_dimension(3, axis, r, r0, k)
    # tools.check_shape_like(np.sum(r, axis=axis), np.sum(k, axis=axis), w0, np.sum(r0, axis=axis), p0)
    k_abs = tools.absolute(k, axis=axis)
    e_r, e_theta, e_phi = tools.orthonormal(k)
    rho = np.sqrt(np.sum((r - r0) * e_theta, axis=axis) ** 2 + np.sum((r - r0) * e_phi, axis=axis) ** 2)
    z = np.sum((r - r0) * e_r, axis=axis)
    z0 = 0.5 * w0 ** 2 * k_abs
    w_z = w0 * np.sqrt(1. + (z / z0) ** 2)
    return 2. * p0 / (np.pi * w_z ** 2) * np.exp(-2. * (rho / w_z) ** 2)


def gaussian_doppler_3d(r: array_like, k: array_like, w0: array_like, v: array_like, r0=None, axis=-1) -> array_like:
    """
    :param r: The position 3-vector relative to 'r0' where to calculate the doppler-shifted wave number (m).
    :param k: The k-wave-3-vector of light (rad / m).
    :param w0: The beam waist (m).
    :param v: The velocity 3-vector (m/s).
    :param r0: The position 3-vector of the beam waist. Is (0m, 0m, 0m) if r0 is not specified (m).
    :param axis: The axis along which the vector components are aligned.
    :returns: The length of the k-wave-3-vector in the atoms rest frame (rad / m).
    :raises ValueError: r, k, v and r0 must have 3 components along the specified axis.
     The shapes of r, k, w0, v and r0 must be compatible.
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
    return k_abs * gamma_3d(v) * (1. - bet_abs * np.cos(alpha) * (1. - w0 ** 2 / 2. / z_plus
                                                                  - rho ** 2 / 2. * z_minus / (z_plus ** 2))
                                  - bet_abs * np.sin(alpha) * rho * z / z_plus)


""" Probability distributions """


def thermal_v_pdf(v: array_like, m: array_like, t: array_like) -> array_like:
    """
    :param v: velocity quantiles (m/s).
    :param m: The mass of the ensembles bodies (amu).
    :param t: The temperature of the ensemble (K).
    :returns: The probability density in thermal equilibrium at the velocity v (s/m).
    """
    v, m, t = np.asarray(v), np.asarray(m), np.asarray(t)
    scale = np.sqrt(sc.k * t / (m * sc.atomic_mass))
    return st.norm.pdf(v, scale=scale)


def thermal_v_rvs(m: array_like, t: array_like, size: Union[int, tuple] = 1) -> array_like:
    """
    :param m: The mass of the ensembles bodies (amu).
    :param t: The temperature of the ensemble (K).
    :param size: Either the size (int) or shape (tuple) of the returned velocity array.
     If 'm' or 't' is an iterable/array, their common shape must be appended to the desired shape of the random samples.
    :returns: Random velocities according to the thermal equilibrium distribution (m/s).
    """
    m, t = np.asarray(m), np.asarray(t)
    scale = np.sqrt(sc.k * t / (m * sc.atomic_mass))
    return st.norm.rvs(scale=scale, size=size)


def thermal_e_pdf(e: array_like, t: array_like) -> array_like:
    """
    :param e: energy quantiles (eV).
    :param t: The temperature of the ensemble (K).
    :returns: The probability density at the energy e, distributed according to a boltzmann distribution (1/eV).
    """
    e, t = np.asarray(e), np.asarray(t)
    scale = sc.k * t / 2. / E_NORM
    return st.chi2.pdf(e, 1, scale=scale)


def thermal_e_rvs(t: array_like, size: Union[int, tuple] = 1) -> array_like:
    """
    :param t: The temperature of the ensemble (K).
    :param size: Either the size (int) or shape (tuple) of the returned energy array.
     If 't' is an iterable/array, its shape must be appended to the desired shape of the random samples.
    :returns: Random energies according to the boltzmann distribution (m/s).
    """
    t = np.asarray(t)
    scale = sc.k * t / 2. / E_NORM
    return st.chi2.rvs(1, scale=scale, size=size)


def convolved_boltzmann_norm_pdf(e: array_like, t: array_like, scale_e: array_like, e0: array_like = 0) -> array_like:
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
                                 scale_e: array_like, e0: array_like = 0, relativistic=True) -> array_like:
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
                                 scale_e: array_like, e0: array_like = 0, relativistic=True) -> array_like:
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


def convolved_thermal_norm_f_lin_pdf(f: array_like, xi: array_like, sigma: array_like, col=True) -> array_like:
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
