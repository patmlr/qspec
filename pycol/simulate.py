# -*- coding: utf-8 -*-
"""
pycol.simulate

Created on 05.05.2020

@author: Patrick Mueller

Module including classes and methods to simulate fluorescence spectra.
"""

from time import time
import scipy.integrate as si
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d.axes3d import Axes3D

import pycol.algebra as al
from pycol._simulate import *


try:
    matplotlib.use('Qt5Agg')
except ImportError:
    pass
# config = tools.get_config_dict()
# plt.rcParams['animation.ffmpeg_path'] = config['ffmpeg']

LINESTYLES = ['-', ':', '--', '-.']


def ct_markov_analytic(t: array_like, n: array_like, rates: array_like, r: array_like = 1.):
    """
    Computes the analytic solution of the evolution of a linear continuous-time markov chain.

    CAUTION! Computation of analytic solution problematic
    (Multiplication of very small numbers with very large numbers).

    :param t: The time at which the probability is returned (us).
    :param n: The number of the state for which the probability is returned. The first state corresponds to 'n' = 0.
    :param rates: The rate(s) at which the markov chain is transferring population from state i to state i + 1.
     If 'rates' is a scalar, this rate is assumed for all 0 &#8804; i &#8804; 'n'. If 'rates' is an Iterable,
     it must have length max('n') + 1 and individual rates 'rates[i]' are assumed for the states
     0 &#8804; i &#8804; 'n'.
    :param r: The ratio of the population which is transferred from state i to state i + 1. Then 1 - r is the ratio
     of the population that is lost during the transition from state i to state i + 1.
    :returns: The probability for the markov chain to be in state 'n' after the time 't' for P(t=0, n=0) = 1.
     The normalization condition is sum_n(P(n, t0)) = 1 for every single point in time t0.
     If 't' and 'n' are Iterables, a 2-d array is returned
     with shape (t.size, n.size), containing all probabilities P(t, n).
    """
    t = np.asarray(t)
    t_scalar = False
    if len(t.shape) == 0:
        t_scalar = True
    t = np.expand_dims(t.flatten(), axis=(1, 2))

    n = np.asarray(n, dtype=int)
    n_zero = n == 0
    n_max = np.max(n)
    n_scalar = False
    if len(n.shape) == 0:
        n_scalar = True
        # return np.array([ct_markov(t, n_i, rates[:(int(n_i)+1)], r=r) for n_i in n.flatten()]).T
    n = np.expand_dims(n.flatten(), axis=(0, 1))

    rates = np.asarray(rates)
    r = np.asarray(r)
    if len(r.shape) > 0 or np.abs(r - 0.5) > 0.5:
        raise ValueError('\'r\' must be a scalar between 0 and 1.')

    if len(rates.shape) == 0:
        ret = (r * rates * t[:, 0, :]) ** n[:, 0, :] / tools.factorial(n[:, 0, :]) * np.exp(-rates * t[:, 0, :])
    else:
        b = np.full((n_max, n_max), np.nan)

        def _b(_i, _n):
            _b_k = b[_i, _n - 1]
            if not np.isnan(_b_k):
                return _b_k
            if _i == _n - 1:
                if _n == 1:
                    return r * rates[0] / (rates[1] - rates[0])
                return r * rates[_i] / (rates[_i] - rates[_n]) * np.sum([_b(_j, _i) for _j in range(_n - 1)])
            elif _i <= _n - 2:
                return r * rates[_n - 1] / (rates[_n] - rates[_i]) * _b(_i, _n - 1)
            return 0.

        for n_i in range(1, n_max + 1, 1):
            for i_i in range(n_i):
                b[i_i, n_i - 1] = _b(i_i, n_i)

        b[np.isnan(b)] = 0.
        b = np.expand_dims(b, axis=0)
        b_map = n.flatten()[~n_zero] - 1

        ret = np.zeros((t.size, n.size))
        ret[:, n_zero] = np.exp(-rates[0] * t[:, 0, :])
        rates = np.expand_dims(rates, axis=(0, 2))

        ret[:, ~n_zero] = np.sum(b[:, :, b_map] * np.exp(-rates[:, :-1, :] * t), axis=1) \
            - np.sum(b[:, :, b_map], axis=1) \
            * np.exp(-np.transpose(rates[:, b_map + 1, :], axes=[0, 2, 1]) * t[:, 0, :])

    if t_scalar:
        if n_scalar:
            return ret[0, 0]
        return ret[0, :]
    if n_scalar:
        return ret[:, 0]
    return ret


def ct_markov_dgl(t: array_like, n: int, rates: array_like, r: array_like = 1., p0: array_like = None,
                  time_resolved: bool = False, show: bool = False):
    """
    Computes the evolution of a linear continuous-time markov chain numerically by solving the underlying ODE system.

    :param t: The time after which the probability is returned.
     If all times from 0 to 't' are required, use 'time_resolved'=True (us).
    :param n: The maximum state number to compute. The first state corresponds to 'n' = 0.
    :param rates: The rate(s) at which the markov chain is transferring population from state i to state i + 1.
     If 'rates' is a scalar, this rate is assumed for all 0 &#8804; i &#8804; 'n'. If 'rates' is an Iterable,
     individual rates 'rates[i]' are assumed for the states i and 'rates' must have size 'n' + 1.
    :param r: The ratio of the population which is transferred from state i to state i + 1. Here 1 - r is the ratio
     of the population that is lost during the transition from state i to state i + 1.
    :param p0: The initial population distribution. Must be an Iterable of length max('n') + 1.
     If None, 'p0' is set to [1, 0, 0, ..., 0].
    :param time_resolved: Whether to return the complete history of the result.
     If True, a 2-tuple similar to (time, population) is returned, were population has shape (time.size, n + 1).
     time will be an array of equally spaced times, such that numerical integrations can be performed easily.
    :param show: Whether to plot the result.
    :returns: The probability distribution of the states 0 &#8804; i &#8804; 'n' after the time 't'
     for P(t=0, n=i) = p0[i] The normalization condition is sum_n(P(n, t0)) = 1 for every single point in time t0.
    """
    t = np.asarray(t, dtype=float)
    if len(t.shape) > 0:
        raise ValueError('\'t\' must be a scalar.')
    n_dim = int(n) + 1
    rates = np.asarray(rates)
    if len(rates.shape) == 0:
        rates = np.full(n_dim, rates)
    else:
        rates = rates.flatten()
        if rates.size != n_dim:
            raise ValueError('\'rates\' must have size max(\'n\') + 1, but has size {}.'.format(rates.size))
    r = np.asarray(r)
    if len(r.shape) > 0 or np.abs(r - 0.5) > 0.5:
        raise ValueError('\'r\' must be a scalar between 0 and 1.')

    if p0 is None:
        p0 = np.zeros(n + 1)
        p0[0] = 1.
    else:
        p0 = np.asarray(p0, dtype=float).flatten()
        if p0.size != n_dim:
            raise ValueError('\'p0\' must have size max(\'n\') + 1, but has size {}.'.format(p0.size))

    def _f(_t, _y):
        return np.array([-rates[0] * _y[0], ] + [r * _ri * _yi - _rj * _yj for _ri, _yi, _rj, _yj
                                                 in zip(rates[:-1], _y[:-1], rates[1:], _y[1:])])

    jac = np.zeros((n_dim, n_dim))
    for i, ri in enumerate(rates):
        jac[i, i] = -ri
        if i > 0:
            jac[i, i - 1] = r * rates[i - 1]

    def _df(_t, _y):
        return jac

    dt = 0.05 / np.max(rates)
    t_array = np.linspace(0., t, tools.odd(t / dt))
    ode = si.ode(_f, jac=_df)
    ode.set_initial_value(p0, 0.)
    # noinspection PyTypeChecker
    y_array = np.array([p0, ] + [ode.integrate(ti) for ti in t_array[1:]])
    # result = si.solve_ivp(_f, (0., t), p0, method='RK45', max_step=dt, t_eval=t_array, vectorized=True)
    # # noinspection PyUnresolvedReferences
    # t_array, y_array = result.t, result.y.T

    if show:
        for ni, yi in enumerate(y_array.T):
            plt.plot(t_array, yi, label='n={}'.format(ni))
        plt.plot(t_array, np.sum(y_array, axis=1), 'k--', label=r'$N$')
        if y_array.shape[1] < 10:
            plt.legend()
        plt.show()

    if time_resolved:
        return t_array, y_array
    return y_array[-1, :]


def lambda_states(t: array_like, delta_1: array_like, delta_2: array_like, a_ge: array_like, a_me: array_like,
                  s_1: array_like, s_2: array_like, lw_1: array_like = 0., lw_2: array_like = 0., p0: array_like = None,
                  time_resolved: bool = False, show: bool = False):
    """
    Computes the evolution of Lambda-systems such as the alkali metals or the singly-charged alkaline-earth metals.
    The state vector is defined as (g, m, e), where g is the first end of the Lambda,
    m the second end and e the intermediate state.

    :param t: The time after which the probability is returned.
     If all times from 0 to 't' are required, use 'time_resolved'=True (us).
    :param delta_1: The detuning of the first laser relative to the g->e transition.
    :param delta_2: The detuning of the second laser relative to the m->e transition.
    :param a_ge: The Einstein coefficient of the e->g transition.
    :param a_me: The Einstein coefficient of the e->m transition.
    :param s_1: The saturation parameter of the g->e transition.
    :param s_2: The saturation parameter of the m->e transition.
    :param lw_1: The frequency width of the first laser.
    :param lw_2: The frequency width of the second laser.
    :param p0: The initial density matrix. Must have shape (6, ), containing the elements [gg, mm, ee, gm, eg, em].
     If None, initially all population is in the g state.
    :param time_resolved: Whether to return the complete history of the result.
    :param show: Whether to plot the result.
    :returns: The density matrix elements after the time 't'. If time_resolved is True,
     a 2-tuple similar to (time, density matrix) is returned, were the density matrix has shape (time.size, 6).
     time will be an array of equally spaced times, such that numerical integrations can be performed easily.
    """
    t = np.asarray(t, dtype=float)
    if len(t.shape) > 0:
        raise ValueError('\'t\' must be a scalar.')

    if p0 is None:
        p0 = np.zeros(6, dtype=complex)
        p0[0] = 1.
    else:
        p0 = np.asarray(p0, dtype=complex).flatten()
        if p0.size != 6:
            raise ValueError('\'p0\' must have size 6, but has size {}.'.format(p0.size))

    _delta_1 = 2 * np.pi * delta_1
    _delta_2 = 2 * np.pi * delta_2
    rabi_1 = a_ge * np.sqrt(s_1 / 2.)
    rabi_2 = a_me * np.sqrt(s_2 / 2.)

    def _f(_t, _y):
        i_gg = _y[0]
        i_mm = _y[1]
        i_ee = _y[2]

        i_gm = _y[3]
        i_eg = _y[4]
        i_em = _y[5]

        gg = a_ge * i_ee + 1j * rabi_1 * (np.conj(i_eg) - i_eg) / 2.
        mm = a_me * i_ee + 1j * rabi_2 * (np.conj(i_em) - i_em) / 2.
        ee = -(a_ge + a_me) * i_ee - 1j * rabi_1 * (np.conj(i_eg) - i_eg) / 2. \
            - 1j * rabi_2 * (np.conj(i_em) - i_em) / 2.

        gm = (1j * (_delta_2 - _delta_1) - (lw_1 + lw_2) / 2.) * i_gm \
            + 1j * rabi_2 * np.conj(i_eg) / 2. - 1j * rabi_1 * i_em / 2.
        eg = (1j * _delta_1 - (a_ge + a_me + lw_1) / 2.) * i_eg \
            + 1j * rabi_1 * (i_ee - i_gg) / 2. - 1j * rabi_2 * np.conj(i_gm) / 2.
        em = (1j * _delta_2 - (a_ge + a_me + lw_2) / 2.) * i_em \
            + 1j * rabi_2 * (i_ee - i_mm) / 2. - 1j * rabi_1 * i_gm / 2.

        return np.array([gg, mm, ee, gm, eg, em], dtype=complex)

    dt = 0.1 / np.max([a_ge + rabi_1 + lw_1, a_me + rabi_2 + lw_2])
    t_array = np.linspace(0., t, tools.odd(t / dt))
    ode = si.complex_ode(_f)
    ode.set_initial_value(p0, 0.).set_integrator('vode', method='bdf')
    # noinspection PyTypeChecker
    y_array = np.array([p0, ] + [ode.integrate(ti) for ti in t_array[1:]])
    # result = si.solve_ivp(_f, (0., t), p0, method='RK45', max_step=dt, t_eval=t_array, vectorized=True)
    # # noinspection PyUnresolvedReferences
    # t_array, y_array = result.t, result.y.T
    if show:
        plt.plot(t_array, y_array[:, 0].real, label=r'$\rho_\mathrm{gg}$')
        plt.plot(t_array, y_array[:, 1].real, label=r'$\rho_\mathrm{mm}$')
        plt.plot(t_array, y_array[:, 2].real, label=r'$\rho_\mathrm{ee}$')
        plt.legend(loc=5)
        plt.show()

    if time_resolved:
        return t_array, y_array
    return y_array[-1, :]


def lambda_ge_rec(t: array_like, n: array_like, delta: array_like, a_ge: array_like, a_me: array_like, s: array_like,
                  f: array_like, m: array_like, p0: array_like = None, dt: float = None, time_resolved: bool = False,
                  show: bool = False):
    """
    Computes the evolution of Lambda-systems such as the alkali metals or the singly-charged alkaline-earth metals,
    taking into account photon recoils. The system is driven by a single laser.
    The state vector is defined as (g0, m0, e0, g1, m1, e1, ..., gn, mn, en), where g is the first end of the Lambda,
    m the second end and e the intermediate state.
    The photon recoils are modeled using a discrete subspace and not the continuous momentum space.
    The number of photon recoils increases by one when the system is excited from the g state to the e state.
    Vice-versa, the number of photon recoils decreases by one when the system is deexcited
    from the e state to the g state via the process of stimulated emission. When the system decays into the g state
    via the dissipative mechanism described by the Einstein coefficient 'a_ge',
    the number of photon recoils does not change.

    :param t: The time after which the probability is returned.
     If all times from 0 to 't' are required, use 'time_resolved'=True (us).
    :param n: The maximum number of photon recoils to consider.
    :param delta: The detuning of the laser relative to the g->e transition (MHz).
    :param a_ge: The Einstein coefficient of the e->g transition (MHz).
    :param a_me: The Einstein coefficient of the e->m transition (MHz).
    :param s: The saturation parameter of the g->e transition.
    :param f: The transition frequency of the g->e transition (MHz).
    :param m: The mass number of the element (u).
    :param p0: The initial density matrix. Must have shape (6, ), containing the elements [gg, mm, ee, gm, eg, em](0)
     or be the full density matrix with all the recoil information.
    :param dt: The width of the time steps.
    :param time_resolved: Whether to return the complete history of the result.
    :param show: Whether to plot the result.
    :returns: The diagonal density matrix elements after the time 't'. If time_resolved is True,
     a 2-tuple similar to (time, density matrix) is returned, were the density matrix has shape (time.size, 3(n+1)).
     time will be an array of equally spaced times, such that numerical integrations can be performed easily.
    """
    t = np.asarray(t, dtype=float)
    if len(t.shape) > 0:
        raise ValueError('\'t\' must be a scalar.')
    if p0 is None:
        y0 = np.zeros(int(3 * (n + 1) * (3 * (n + 1) + 1) / 2), dtype=complex)
        y0[0] = 1. + 0.j
    else:
        p0 = np.asarray(p0, dtype=float).flatten()
        if p0.size == 6:
            y0 = np.zeros(int(3 * (n + 1) * (3 * (n + 1) + 1) / 2), dtype=complex)
            y0[:6] = p0
        elif p0.size == int(3 * (n + 1) * (3 * (n + 1) + 1) / 2):
            y0 = p0
        else:
            raise ValueError('\'p0\' must have size {}, but has size {}.'
                             .format(int(3 * (n + 1) * (3 * (n + 1) + 1) / 2), p0.size))

    _delta = 2 * np.pi * delta
    rabi = a_ge * np.sqrt(s / 2.)
    f_rec = 2 * np.pi * ph.photon_recoil(f, m)

    def hamiltonian(_t, n1, n2):  # without hbar
        gg, mm, ee = 0. + 0.j, 0. + 0.j, 0. + 0.j
        gm, eg, me = 0. + 0.j, 0. + 0.j, 0. + 0.j
        mg, ge, em = 0. + 0.j, 0. + 0.j, 0. + 0.j
        if n1 == n2:
            _e_kin = n1 ** 2 * f_rec
            # _delta includes one recoil, _delta := (w_eg - w_L +- k*v + k*v_rec)
            gg = -(_delta + 2 * n1 * f_rec) + _e_kin
            mm, ee = _e_kin, _e_kin
        elif n1 == n2 - 1:
            ge = rabi / 2.
        elif n1 == n2 + 1:
            eg = rabi / 2.
        else:
            return np.zeros((3, 3), dtype=complex)
        return np.array([[gg, gm, ge], [mg, mm, me], [eg, em, ee]], dtype=complex)

    def rho(_y):
        ret = np.zeros((3 * (n + 1), 3 * (n + 1)), dtype=complex)
        np.fill_diagonal(ret, _y[:(3 * (n + 1))])
        for _i in range(3 * (n + 1) - 1):
            i_start = 3 * (n + 1) + sum([3 * (n + 1) - _j - 1 for _j in range(_i)])
            i_end = i_start + sum([3 * (n + 1) - _j - 1 for _j in range(_i + 1)])
            np.fill_diagonal(ret[(_i+1):], _y[i_start:i_end].conjugate())
            np.fill_diagonal(ret[:, (_i+1):], _y[i_start:i_end])
        return ret

    def _f(_t, _y):
        _rho = rho(_y)
        h = np.block([[hamiltonian(_t, n1, n2) for n2 in range(n + 1)] for n1 in range(n + 1)])
        ret = -1.j * (h @ _rho - _rho @ h)

        zero = np.zeros((3, 3))
        sigma = np.array([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]])
        sigma_eg = np.block([[sigma if _i == _j else zero for _j in range(n + 1)] for _i in range(n + 1)])
        sigma = np.array([[0., 0., 0.], [0., 0., 0.], [0., 1., 0.]])
        sigma_em = np.block([[sigma if _i == _j else zero for _j in range(n + 1)] for _i in range(n + 1)])
        lindbladian_ge = sigma_eg.T @ _rho @ sigma_eg \
            - 0.5 * (sigma_eg @ sigma_eg.T @ _rho + _rho @ sigma_eg @ sigma_eg.T)
        lindbladian_me = sigma_em.T @ _rho @ sigma_em \
            - 0.5 * (sigma_em @ sigma_em.T @ _rho + _rho @ sigma_em @ sigma_em.T)
        lindbladian_ge *= a_ge
        lindbladian_me *= a_me
        ret += lindbladian_ge + lindbladian_me
        return np.concatenate(tuple(np.diagonal(ret, offset=_i) for _i in range(3 * (n + 1))), axis=0)

    dt = 0.1 / (a_ge + rabi) if dt is None else dt
    t_array = np.linspace(0., t, tools.odd(t / dt))
    ode = si.complex_ode(_f)
    ode.set_initial_value(y0, 0.).set_integrator('vode', method='bdf')
    # noinspection PyTypeChecker
    y_array = np.array([y0, ] + [ode.integrate(ti) for ti in t_array[1:]])
    # result = si.solve_ivp(_f, (0., t), y0, method='RK45', max_step=dt, t_eval=t_array, vectorized=True)
    # # noinspection PyUnresolvedReferences
    # t_array, y_array = result.t, result.y.T
    if show:
        plt.plot(t_array, np.sum([y_array[:, 3 * i].real for i in range(n + 1)], axis=0),
                 'C0--', label=r'$N_\mathrm{gg}$')
        plt.plot(t_array, np.sum([y_array[:, 3 * i + 1].real for i in range(n + 1)], axis=0),
                 'C1--', label=r'$N_\mathrm{mm}$')
        plt.plot(t_array, np.sum([y_array[:, 3 * i + 2].real for i in range(n + 1)], axis=0),
                 'C2--', label=r'$N_\mathrm{ee}$')
        plt.plot(t_array, np.sum([y_array[:, 3 * i].real + y_array[:, 3 * i + 1].real + y_array[:, 3 * i + 2].real
                                  for i in range(n + 1)], axis=0), 'k--', label=r'$N$')
        pg, pm, pe = None, None, None
        for ni, yi in enumerate(y_array.T[:(3 * (n + 1))]):
            if ni % 3 == 0:
                pg, = plt.plot(t_array, yi.real, 'b-')
            if ni % 3 == 1:
                pm, = plt.plot(t_array, yi.real, 'r-')
            if ni % 3 == 2:
                pe, = plt.plot(t_array, yi.real, 'g-')
        pg.set_label(r'$\rho_\mathrm{gg}^n$')
        pm.set_label(r'$\rho_\mathrm{mm}^n$')
        pe.set_label(r'$\rho_\mathrm{ee}^n$')
        plt.xlabel('Time (us)')
        plt.xlabel('Population')
        plt.legend(loc=7)
        plt.show()

        plt.bar(np.arange(n + 1), [y_array[-1, 3 * i + 0].real for i in range(n + 1)])
        plt.show()

    if time_resolved:
        return t_array, y_array[:, :(3 * (n + 1))].real
    return y_array[-1, :(3 * (n + 1))].real


class Geometry:
    """
    Class representing a fluorescence detection geometry. The solid angle over which fluorescence light is detected
    can be defined through intervals of the two angles 'theta' and 'phi'.
    With these, every spacial direction can be addressed using an orthonormal system defined by

    .. math::

        \\hat{e}_r &:= \\begin{pmatrix}\\sin(\\theta) \\\\
                       \\cos(\\theta)\\sin(\\phi) \\\\
                       \\cos(\\theta)\\cos(\\phi)\\end{pmatrix}.

    If the user specifies a rotation object with unitary matrix :math:`R`,
    the new system is :math:`\\hat{e}_r^\\prime = R \\hat{e}_r`
    The entire two-dimensional interval is defined through the cartesian product
    :math:`\\bigcup_i \\theta_i \\times \\bigcup_i \\phi_i`.
    For every disjoint interval a weight can be defined through a 'weights' matrix.
    A probability distribution function (pdf) can be defined to have continuous angle weights.
    A rotation matrix can be defined to rotate the entire coordinate systems/detection geometry.
    A sample of angle pairs from the defined intervals can be generated using the 'integration_sample' method.
    """
    def __init__(self):
        """
        Initializing the different attributes. The standard interval is
        :math:`\\theta\\in [-\\pi/2, \\pi/2]` and \\phi\\in [0, 2\\pi].
        """
        self.theta_intervals = np.array([[-np.pi / 2., np.pi / 2.]])
        self.phi_intervals = np.array([[0., 2. * np.pi]])
        self.solid_angle = 4 * np.pi
        self.step = np.pi / 32.
        self.pdf = None
        self.weights = None
        self.rotation = tools.Rotation()

    def set_intervals(self, theta: array_iter, phi: array_iter):
        """
        :param theta: An interval or a list of intervals for the angle 'theta'.
        :param phi: An interval or a list of intervals for the angle 'phi'.
        :returns: None. Merges overlapping intervals and calculates the solid angle
         corresponding to the defined intervals.
        """
        theta, phi = np.asarray(theta), np.asarray(phi)
        if len(theta.shape) == 1:
            theta = np.expand_dims(theta, axis=0)
        if len(phi.shape) == 1:
            phi = np.expand_dims(phi, axis=0)
        if len(theta.shape) != 2 or len(phi.shape) != 2:
            raise ValueError('theta and phi must be intervals or lists of intervals.')
        self.theta_intervals = tools.merge_intervals(theta)
        self.phi_intervals = tools.merge_intervals(phi)
        y_int = 0.
        theta_int, phi_int = self.integration_sample()
        for t in theta_int:
            y = np.cos(t)
            y_int += si.simps(y, t)
        y_int *= np.sum([p[-1] - p[0] for p in phi_int])
        self.solid_angle = y_int

    def set_weights(self, weights: Union[array_iter, None]):
        """
        :param weights: None or a matrix of weights for the defined disjoint intervals.
         The shape of the weights must fulfill weights.shape == (len(self.theta_intervals), len(self.phi_intervals)).
        :returns: None. Sets the 'weights' attribute of the geometry object.
        """
        if weights is not None:
            weights = np.asarray(weights)
            shape = (self.theta_intervals.shape[0], self.phi_intervals.shape[0])
            if weights.shape != shape:
                raise ValueError('Weights must have shape {}, but have shape {}.'.format(shape, weights.shape))
        self.weights = weights

    def set_pdf(self, pdf: Union[Callable, None]):
        """
        :param pdf: None or a callable which accepts two arguments (theta, phi).
        :returns: None. Sets the 'pdf' attribute of the Geometry object.
        """
        if pdf is not None:
            try:
                pdf(self.theta_intervals[0, 0], self.phi_intervals[0, 0])
            except AttributeError:
                raise AttributeError('The pdf must be a callable which takes two arguments *(theta, phi).')
        self.pdf = pdf

    def set_rotation(self, r: tools.Rotation = None):
        """
        :param r: None or a 3x3 matrix defining a rotation.Therefore, dot(r, r.T) = I and Det(r) = 1 must be fulfilled.
         If r == None, the Identity matrix is used.
        :returns: None. Sets the 'R' attribute of the Geometry object.
        """
        if r is None:
            r = tools.Rotation()
        elif not isinstance(r, tools.Rotation):
            raise TypeError('r must be a \'Rotation\' object but is of type {}'.format(type(r)))
        self.rotation = r

    def integration_sample(self, step: scalar = None) -> (list, list):
        """
        :param step: None or a scalar which defines the approximate spacing
         between the equidistant values of the integration sample.
         If step == None, the currently defined step size is used. The standard value is pi/32.
        :returns: Two lists of arrays with equidistant values
         in the defined intervals for 'theta' and 'phi', respectively.
         Note that all intervals are exactly covered by an odd number of values (for Simpson integration)
         under the expense of not matching the 'step' size exactly.
        """
        if step is None:
            step = self.step
        theta = [np.linspace(t[0], t[1], tools.odd((t[1] - t[0]) / step + 1.)) for t in self.theta_intervals]
        phi = [np.linspace(p[0], p[1], tools.odd((p[1] - p[0]) / step + 1.)) for p in self.phi_intervals]
        return theta, phi

    def plot(self, show: bool = True):
        """
        Shows a 3d plot of the angular range covered by the geometry object.
        :returns: The axes object.
        """
        theta, phi = self.integration_sample(step=0.02)
        r = np.array([[tools.e_r(t, p) for p in phi[0]] for t in theta[0]])
        r = tools.transform(np.expand_dims(self.rotation.R, axis=(0, 1)), r)
        fig = plt.figure(num=1, figsize=[8, 8], clear=True)
        ax = fig.add_subplot(111, projection='3d')
        plt.xlabel(r'$x$ / arb. units')
        plt.ylabel(r'$y$ / arb. units')
        ax.set_zlabel(r'$z$ / arb. units')

        lim = 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ticks = [-1., -0.5, 0., 0.5, 1.]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_zticks(ticks)
        ax.pbaspect = [1., 1., 1.]
        radius = 0.8
        r *= radius
        surf = ax.plot_surface(r[:, :, 0], r[:, :, 1], r[:, :, 2], zorder=2, rcount=40, ccount=40,
                               cmap=plt.get_cmap('cividis'), antialiased=False, alpha=0.3)
        surf.set_clim([-1, 1])
        # cb = plt.colorbar(surf, ax=ax, shrink=0.5)
        # cb.set_label('z / arb. units')
        # cb.set_ticks([-1, 0, 1])

        plt.style.use('seaborn')
        plt.tight_layout()
        if show:
            plt.show()
            plt.style.use('default')

        return fig, ax


class ScatteringRate:
    """
    Class representing the perturbative scattering rate of a closed electronic transition.
    The spectrum is excited by a laser with the specified 'polarization'
    and recorded with a FDR defined by 'geometry'. An external magnetic field can be defined through 'b'.
    """
    def __init__(self, atom: Atom, i_decay: int = 0, geometry: Geometry = None,
                 polarization: Polarization = None, b: array_like = None):
        """
        :param atom: The investigated atom.
        :param geometry: The geometry of the FDR.
        :param polarization: The polarization of the incident laser beam.
        :param b: The external magnetic field. Can be a vector or a scalar.
         In the latter case, the magnetic field is aligned with the z-axis.
        """
        self.environment = Environment()
        self.atom = atom
        self.i_decay = i_decay
        self.state_l, self.state_u = None, None
        self.i_state_l, self.i_state_u = None, None
        self.freq_0 = None
        self.v = None
        self.gamma = None
        self.A, self.a_const = None, None
        self.x0 = None
        self.indices = None
        self.x0_array, self.A_i, self.A_f, self.counts = None, None, None, None
        self.b, self.b_abs = np.array([0., 0., 0.]), 0.
        self.e_r_b = np.array([0., 0., 1.])
        self.e_l = np.array([0., 0., 1.])
        self.R_b, self.R = np.identity(3, dtype=float), np.identity(3, dtype=float)
        self.geometry, self.polarization = Geometry(), Polarization(self.e_l)
        self.set_states()
        self.check_atom()
        self.set_geometry(geometry)
        self.set_polarization(polarization)
        self.generate_dipoles()
        self.set_b(b)  # TODO: Implement Environments in C++.
    
    def check_atom(self):
        """
        Check whether the structure of the atom fulfills all requirements to calculate the scattering rate.

        :raises ValueError: The atom has to consist of a single closed transition between two fine-structure states.
         Their common nucleus can have an arbitrary spin.
        :returns: None.
        """
        s, l, j, i = self.state_l[0].s, self.state_l[0].l, self.state_l[0].j, self.state_l[0].i
        if any(state_l.s != s or state_l.l != l or state_l.j != j or state_l.i != i for state_l in self.state_l):
            raise ValueError('All lower states must be part of the same fine-structure state.')
        s, l, j, i = self.state_u[0].s, self.state_u[0].l, self.state_u[0].j, self.state_u[0].i
        if any(state_u.s != s or state_u.l != l or state_u.j != j or state_u.i != i for state_u in self.state_u):
            raise ValueError('All upper states must be part of the same fine-structure state.')
        if self.state_l[0].i != self.state_u[0].i:
            raise ValueError('The lower and upper states must be part of the same nuclear state.')

    def set_geometry(self, geometry: Geometry = None):
        """
        :param geometry: The geometry object. If None, the entire solid angle is covered by the detector.
        :return: None. Sets the geometry object.
        """
        self.geometry = geometry
        if self.geometry is None:
            self.geometry = Geometry()
        self.R = np.dot(self.geometry.rotation.R, self.R_b)

    def set_polarization(self, polarization: Polarization = None):
        """
        :param polarization: The polarization object. If None, the light is linearly polarized along the z-axis.
        :returns: None. Sets the polarization object and the local polarization vector 'e_l'.
        """
        self.polarization = polarization
        if self.polarization is None:
            self.polarization = Polarization([0, 0, 1])
        self.e_l = np.dot(self.R_b, self.polarization.x)
        self.polarization.def_q_axis(self.e_r_b)

    def set_states(self):
        """
        Define lower and upper states.

        :returns: None.
        """
        state_l = np.array([state for state in self.atom if state.label == self.atom.decay_map.labels[self.i_decay][0]])
        state_u = np.array([state for state in self.atom if state.label == self.atom.decay_map.labels[self.i_decay][1]])
        self.freq_0 = np.mean([s_u.freq_j for s_u in state_u]) - np.mean([s_l.freq_j for s_l in state_l])
        if self.freq_0 > 0:
            self.state_u = state_u
            self.state_l = state_l
        else:
            self.freq_0 *= -1
            self.state_u = state_l
            self.state_l = state_u

    def set_x0(self):
        """
        :returns: None. Updates the resonance positions.
        """
        self.x0 = [[state_u.freq - state_l.freq - self.freq_0 for state_u in self.state_u] for state_l in self.state_l]
        self.x0_array = np.array([self.x0[col[0]][col[1]] for row in self.indices for col in row], dtype=float)

    def generate_dipoles(self):
        """
        :returns: None. Generates an array of frequencies which cover the entire spectrum
         and saves it to the x attribute of the Spectrum object.
        """
        i = self.state_l[0].i
        self.gamma = self.atom.decay_map.a[self.i_decay] / (2 * np.pi)
        j_l = self.state_l[0].j
        j_u = self.state_u[0].j

        self.A = [[np.zeros(3, dtype=complex) if abs(u.f - ll.f) > 1 or abs(u.m - ll.m) > 1
                   else al.a_dipole_cart(i, j_l, ll.f, ll.m, j_u, u.f, u.m)
                   for u in self.state_u] for ll in self.state_l]

        self.a_const = [[0. if abs(u.f - ll.f) > 1 or abs(u.m - ll.m) > 1
                         else al.a_dipole(i, j_l, ll.f, ll.m, j_u, u.f, u.m, u.m - ll.m, as_sympy=False)
                         for u in self.state_u] for ll in self.state_l]

        self.indices = [[[i_i, i_u, i_f] for i_u, u in enumerate(self.state_u)
                         if abs(u.f - i.f) <= 1 and abs(u.m - i.m) <= 1
                         and abs(u.f - f.f) <= 1 and abs(u.m - f.m) <= 1]
                        for i_i, i in enumerate(self.state_l) for i_f, f in enumerate(self.state_l)]
        self.indices = [arg for arg in self.indices if arg]
        self.A_i = np.array([self.A[col[0]][col[1]] for row in self.indices for col in row], dtype=complex)
        self.A_f = np.array([self.A[col[2]][col[1]] for row in self.indices for col in row], dtype=complex)
        self.counts = np.array([len(row) for row in self.indices], dtype=int)
        self.set_x0()

    def set_b(self, b: array_like):
        """
        :param b: The magnetic field vector.
        :returns: None. Sets the new magnetic field for all states
         and updates the quantization axis and the resonance positions.
        """
        if b is None:
            self.b = np.array([0., 0., 0.])
            self.b_abs = 0.
            self.e_r_b = np.array([0., 0., 1.])
        else:
            self.b = np.asarray(b)
            if len(self.b.shape) == 0:
                self.b = np.array([0., 0., self.b])
            self.b_abs = tools.absolute(self.b)
            self.e_r_b = np.array([0., 0., 1.])
            if not self.b_abs == 0.:
                self.e_r_b = self.b / self.b_abs
        z_axis = np.array([0., 0., 1.])
        self.R_b = tools.rotation_to_vector(self.e_r_b, z_axis).R
        self.R = np.dot(self.geometry.rotation.R, self.R_b)
        self.e_l = np.dot(self.R_b, self.polarization.x)
        self.polarization.def_q_axis(self.e_r_b)
        self.environment = Environment(B=self.b)
        for state in self.atom:
            state.update(self.environment)
        self.set_x0()

    def generate_x(self, width: scalar = 20., step: scalar = None):
        """
        :param width: The covered width around resonances in natural linewidths.
        :param step: The step size between generated x values.
         Note that between resonances, there might be a larger step.
        :returns: An array of frequencies which covers the entire spectrum.
        """
        intervals = [[x0 - width * self.gamma / 2, x0 + width * self.gamma / 2] for x0 in self.x0_array]
        intervals = tools.merge_intervals(intervals)
        if step is None:
            step = self.gamma / 30.
        return np.concatenate([np.linspace(i[0], i[1], tools.odd((i[1] - i[0]) / step + 1.))
                               for i in intervals], axis=0)

    def generate_y0(self, x: array_like) -> array_like:
        """
        :param x: The frequency of light in an atoms rest frame (MHz).
        :returns: The spectrum for the case of unpolarized light
         and the detection of the complete solid angle (4 pi).
        """
        y0 = np.zeros_like(x, dtype=float)
        for x0_list, a_const_list in zip(self.x0, self.a_const):
            for x0, a_const in zip(x0_list, a_const_list):
                denominator = (x - x0) ** 2 + 0.25 * self.gamma ** 2
                y0 += a_const ** 2 / denominator
        return y0 * (self.gamma / 2.) ** 3 / len(self.x0) / 3 * 2 * np.pi

    def generate_y_4pi(self, x: array_like) -> array_like:
        """
        :param x: The frequency of light in an atoms rest frame (MHz).
        :returns: The spectrum for the detection of the complete solid angle (4 pi).
        """
        y0 = np.zeros_like(x, dtype=float)
        for x0_list, a_const_list, ll in zip(self.x0, self.a_const, self.state_l):
            for x0, a_const, u in zip(x0_list, a_const_list, self.state_u):
                if abs(u.f - ll.f) > 1 or abs(u.m - ll.m) > 1:
                    continue
                denominator = (x - x0) ** 2 + 0.25 * self.gamma ** 2
                y0 += a_const ** 2 / denominator * tools.absolute_complex(self.polarization.q[int(u.m - ll.m) + 1]) ** 2
        return y0 * (self.gamma / 2.) ** 3 / len(self.x0) * 2 * np.pi

    def generate_y(self, x: array_like, theta: array_like, phi: array_like, decimals: int = 8):
        """
        :param x: The frequency of light in an atoms rest frame (MHz).
        :param theta: The angle between the emission direction of the fluorescence light and the x-axis + 90°.
        :param phi: The mixing angle between the y- and z-axis (sin(phi), cos(phi)).
        :param decimals: The precision of the vector calculus in considered decimal places.
        :returns: The fluorescence spectrum for incident light with frequency 'x'
         for the direction of emission defined by 'theta' and 'phi'.
        """
        x, theta, phi = np.asarray(x).flatten(), np.asarray(theta).flatten(), np.asarray(phi).flatten()
        t, p = np.meshgrid(theta, phi, indexing='ij')
        t, p = t.flatten(), p.flatten()
        e_theta = np.around(tools.transform(np.expand_dims(self.R, axis=0), tools.e_theta(t, p)), decimals=decimals)
        e_phi = np.around(tools.transform(np.expand_dims(self.R, axis=0), tools.e_phi(t, p)), decimals=decimals)
        e_theta /= np.expand_dims(tools.absolute(e_theta), axis=-1)
        e_phi /= np.expand_dims(tools.absolute(e_phi), axis=-1)
        # Lorentz Boost TODO
        # if self.v is not None:
        #     e_theta = ph.boost(np.concatenate([[1.], e_theta], axis=0), self.v)[1:]
        #     e_phi = ph.boost(np.concatenate([[1.], e_phi], axis=0), self.v)[1:]
        #     e_theta /= tools.absolute(e_theta)
        #     e_phi /= tools.absolute(e_phi)

        norm = 3. / (8. * np.pi) * (self.gamma / 2.) ** 3 / len(self.x0) * 2 * np.pi
        denominator = 1. / (-np.expand_dims(x, axis=1) + self.x0_array + 0.5j * self.gamma)

        i_l = np.around(np.sum(self.e_l * self.A_i, axis=-1), decimals=decimals)
        f_theta = np.around(np.sum(np.expand_dims(e_theta, axis=1)
                                   * np.expand_dims(self.A_f, axis=0), axis=-1) * i_l,
                            decimals=decimals)
        f_phi = np.around(np.sum(np.expand_dims(e_phi, axis=1)
                                 * np.expand_dims(self.A_f, axis=0), axis=-1) * i_l,
                          decimals=decimals)
        shape = np.array([denominator.shape[0], f_theta.shape[0], self.counts.shape[0]], dtype=int)
        denominator, f_theta, f_phi = denominator.flatten(), f_theta.flatten(), f_phi.flatten()
        t0 = time()
        ret = sr_generate_y(denominator, f_theta, f_phi, self.counts, shape).reshape((x.size, theta.size, phi.size))
        t0 = time() - t0
        print('Time: {}s'.format(t0))
        return ret * norm

    def integrate_y(self, x: array_like, step: scalar = None):
        """
        :param x: The frequency of light in an atoms rest frame (MHz).
        :param step: The step size used for the integration over the angles theta and phi.
        :returns: The fluorescence spectrum for incident light with frequencies 'x'
         integrated over the directions of emission 'theta' and 'phi'.
        """
        x = np.asarray(x)
        theta, phi = self.geometry.integration_sample(step=step)
        y_int = np.zeros(x.shape)
        for i, t in enumerate(theta):
            for j, p in enumerate(phi):
                y = self.generate_y(x, t, p) * np.expand_dims(np.expand_dims(np.cos(t), axis=0), axis=-1)
                if self.geometry.pdf is not None:
                    y *= np.expand_dims(self.geometry.pdf(t, p), axis=0)
                y = si.simps(y, x=t, axis=1)
                y = si.simps(y, x=p)
                if self.geometry.weights is not None:
                    y *= self.geometry.weights[i, j]
                y_int += y
        return y_int

    def plot_spectrum(self, norm_to_4pi: bool = False, step: scalar = None):
        """

        :param norm_to_4pi: Whether to renormalize the spectrum defined by geometry to have the same maximum
         as the spectrum for the detection with the complete solid angle 4 pi.
         If True, the difference between the two spectra is plotted as well. Default is False.
        :param step: The step size for the integration of the scattering rate.
        :returns: None. Plots the spectrum for the given geometry
         and that for the detection with the complete solid angle 4 pi.
        """
        plt.figure(figsize=[8, 4])
        x = self.generate_x()
        y_4pi = self.generate_y_4pi(x)
        y = self.integrate_y(x, step=step)
        if norm_to_4pi:
            y *= np.max(y_4pi) / np.max(y)
        plt.xlabel(r'$\nu - \nu_0$ (MHz)')
        plt.ylabel(r'Scattering rate (MHz / sat. parameter)')
        plt.plot(x, y_4pi, label=r'$y(4\pi)$')
        plt.plot(x, y, label=r'$y($Geometry$)$')
        if norm_to_4pi:
            plt.plot(x, y - y_4pi, label='Residuals')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_angular_distribution(self, x: scalar = None, n: int = 5, theta: array_iter = None, phi: array_iter = None,
                                  mode: str = None, show: bool = True, save: str = None):
        """
        :param x: The frequency of light in an atoms rest frame (MHz).
         If None, the position of the first maximum is taken (self.x0[0][0]).
        :param n: 2 ** n + 1 samples for 'theta' and 2 ** (n + 1) + 1 samples for 'phi' are drawn,
         giving a total of 2 ** (2n + 1) + 3 * 2 ** n + 1 points to draw.
        :param theta: A custom array for theta. Overwrites 'n'.
        :param phi: A custom array for phi. Overwrites 'n'.
        :param mode: Either '3d' for a surface plot or anything else for a color-mesh plot.
        :param save: The path where to save the plot. If None, the plot will not be saved.
        :param show: Whether to show the plot.
        :returns: theta, phi and z. Optionally, a plot is drawn and saved.
        """
        if theta is not None:
            theta = np.asarray(theta)
        else:
            theta = np.linspace(-np.pi / 2., np.pi / 2., 2 ** n + 1)
        if phi is not None:
            phi = np.asarray(phi)
        else:
            phi = np.linspace(0., 2. * np.pi, 2 ** (n + 1) + 1)
        y = np.squeeze(self.generate_y(self.x0[0][0] if x is None else x, theta, phi))

        if mode is None:
            mode = '2d'
        if mode == '3d':
            r = np.array([[tools.transform(self.geometry.rotation.R, tools.e_r(t, p)) for p in phi] for t in theta])
            fig = plt.figure(num=1, figsize=[8, 8], clear=True)
            ax = fig.add_subplot(111, projection='3d', label='Surface')
            rad = y[:, :]
            rad = np.expand_dims(rad, axis=-1)
            r *= rad
            r -= np.min(tools.absolute(r, axis=-1)) * (r / np.expand_dims(tools.absolute(r, axis=-1), axis=-1))
            r /= np.max(tools.absolute(r, axis=-1))
            plt.xlabel(r'$I_x$ (arb. units)')
            plt.ylabel(r'$I_y$ (arb. units)')
            ax.set_zlabel(r'$I_z$ (arb. units)')
            lim = 1.1
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
            ticks = [-1., -0.5, 0., 0.5, 1.]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_zticks(ticks)
            ax.pbaspect = [1., 1., 1.]
            
            x_color = tools.absolute(r, axis=-1)
            cm = plt.cm.ScalarMappable(cmap='viridis')
            facecolors = cm.to_rgba(x_color)
            surf = ax.plot_surface(r[:, :, 0], r[:, :, 1], r[:, :, 2], rcount=80, ccount=80,
                                   facecolors=facecolors, antialiased=True, alpha=1)
            surf.set_clim([-1, 1])
            cb = plt.colorbar(cm, ax=ax, shrink=0.5)
            cb.set_label('Intensity (arb. units)')
            cb.set_ticks(np.arange(0., 1.1, 0.2))
            plt.gca().axis('off')
            plt.style.use('seaborn')
            plt.tight_layout()
        else:
            # import matplotlib
            font = {'family': 'Arial',
                    'size': 12}
            matplotlib.rc('font', **font)
            plt.figure(figsize=[3, 4.5])
            theta, phi = np.meshgrid(theta, phi, indexing='ij')
            ax = plt.gca()
            ax.set_aspect('equal')
            plt.xlabel(r'$\theta$ / rad')
            plt.ylabel(r'$\varphi$ / rad')
            cmesh = plt.pcolormesh(theta, phi, y[:, :], cmap='viridis')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='4%', pad=0.05)
            cbar = plt.colorbar(cmesh, cax=cax)
            cbar.set_label(label='Intensity (arb. units)')

        if save is not None:
            plt.savefig(save, dpi=300)
        if show:
            plt.show()
        plt.cla()
        plt.clf()
        plt.close()
        plt.style.use('default')
        return theta, phi, y

    def plot_mixed_distribution(self, n: int = 5, mode: str = None, show: bool = True, save: str = None):
        """
        :param n: 2 ** n + 1 samples for 'theta' and 2 ** (n + 1) + 1 samples for 'phi' are drawn,
         giving a total of (2 ** n + 1) * len(self.x) and (2 ** (n + 1) + 1) * len(self.x)
         points to draw for theta and phi, respectively.
        :param mode: Either '3d' for a surface plot or anything else for a color-mesh plot.
        :param save: The path where to save the plot. If None, the plot will not be saved.
        :param show: Whether to show the plot.
        :returns: None. Optionally, two plots are drawn for theta and phi as the y-axes and saved.
        """
        x = self.generate_x()
        theta = np.linspace(-np.pi / 2., np.pi / 2., 2 ** n + 1)
        phi = np.linspace(0., 2. * np.pi, 2 ** (n + 1) + 1)
        y = self.generate_y(x, theta, phi)
        x_theta, theta = np.meshgrid(x, theta, indexing='ij')
        x_phi, phi = np.meshgrid(x, phi, indexing='ij')
        plt.xlabel(r'$\nu - \nu_0$ / MHz')
        plt.ylabel(r'$\theta$ / rad')

        if mode is None:
            mode = '2d'
        if mode == '3d':
            fig = plt.figure(num=1, figsize=[16, 9], clear=True)
            ax = fig.add_subplot(111, projection='3d', label='Surface')
            ax.set_zlabel(r'$I$ / arb. units')
            ax.plot_surface(x_theta, theta, y[:, :, 0], rcount=200/2, ccount=200/2, cmap=plt.get_cmap('viridis'),
                            antialiased=True, alpha=1)
            if save is not None:
                plt.savefig(save, dpi=300)
                plt.close()
            if show:
                plt.show()
            fig = plt.figure(num=1, figsize=[16, 9], clear=True)
            ax = fig.add_subplot(111, projection='3d', label='Surface')
            plt.xlabel(r'$\nu - \nu_0$ / MHz')
            plt.ylabel(r'$\phi$ / rad')
            ax.set_zlabel(r'$I$ / arb. units')
            ax.plot_surface(x_phi, phi, y[:, 2 ** (n - 1), :], rcount=200/2, ccount=200/2, cmap=plt.get_cmap('viridis'),
                            antialiased=True, alpha=1)
            if save is not None:
                plt.savefig(save, dpi=300)
                plt.close()
            if show:
                plt.show()
        else:
            ax = plt.gca()
            cmesh = plt.pcolormesh(x_theta, theta, y[:, :, 0], cmap='viridis')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.1)
            cbar = plt.colorbar(cmesh, cax=cax)
            cbar.set_label(label=r'$I$ / arb. units')
            if save is not None:
                plt.savefig(save, dpi=300)
                plt.close()
            if show:
                plt.show()
            plt.xlabel(r'$\nu - \nu_0$ / MHz')
            plt.ylabel(r'$\phi$ / rad')
            cmesh = plt.pcolormesh(x_phi,  phi, y[:, 2 ** (n - 1), :], cmap='viridis')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.1)
            cbar = plt.colorbar(cmesh, cax=cax)
            cbar.set_label(label=r'$I$ / arb. units')
            if save is not None:
                plt.savefig(save, dpi=300)
                plt.close()
            if show:
                plt.show()

    def plot_setup(self):
        """
        :returns: None. Creates a 3D-plot of the physical situation, including the axis
         of the magnetic field (quantization axis), the real and imaginary axis of the polarization
         and the are which is covered by the detector geometry.
        """
        fig, ax = self.geometry.plot(show=False)
        x, y, z = np.zeros(1), np.zeros(1), np.zeros(1)

        # Laser

        # Polarization
        # Real
        u, v, w = self.polarization.x.real[:, 0], self.polarization.x.real[:, 1], self.polarization.x.real[:, 2]
        ax.quiver(x, y, z, u, v, w, zorder=3, color='b', length=1.,
                  arrow_length_ratio=0.1, label='Real(Pol.)')
        # Imag
        u, v, w = np.meshgrid(self.polarization.x.imag[:, 0], self.polarization.x.imag[:, 1],
                              self.polarization.x.imag[:, 2])
        ax.quiver(x, y, z, u, v, w, zorder=3, color='tab:orange', length=1.,
                  arrow_length_ratio=0.1, label='Imag(Pol.)')

        # B-field / Quantization axis
        u, v, w = np.meshgrid(self.e_r_b[0], self.e_r_b[1], self.e_r_b[2])
        ax.quiver(x, y, z, u, v, w, zorder=2, color='g', length=1.,
                  arrow_length_ratio=0.1, label='B-field /\nQuant. axis')

        # Velocity
        if self.v is not None:
            u, v, w = np.meshgrid(self.v[0], self.v[1], self.v[2])
            ax.quiver(x, y, z, u, v, w, zorder=10, color='r', length=1.,
                      arrow_length_ratio=0.1, label='Velocity')

        plt.legend()
        plt.show()
        plt.style.use('default')
