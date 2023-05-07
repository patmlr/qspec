# -*- coding: utf-8 -*-
"""
pycol.models.splitter

Created on 14.03.2022

@author: Patrick Mueller

Splitter classes for lineshape models.
"""

from string import ascii_uppercase
import numpy as np

from pycol.tools import merge_intervals
from pycol.algebra import wigner_6j, a
from pycol.models.base import Model, Summed


def gen_splitter_model(qi: bool = False, hf_mixing: bool = False):
    if qi and hf_mixing:
        pass
    elif qi:
        pass
    elif hf_mixing:
        return HyperfineMixed
    else:
        return Hyperfine
    raise ValueError('Specified splitter model not available.')


def get_all_f(i, j):
    i, j = np.array(i, float).flatten(), np.array(j, float).flatten()
    return sorted(set(f + abs(_i - _j) for _i in i for _j in j for f in range(int(_i + _j - abs(_i - _j) + 1))))


def hf_coeff(i, j, f):
    """ Return the tuple of hyperfine coefficients for A and B-factor for a given quantum state """
    # First and third order are taken from https://journals.aps.org/pra/abstract/10.1103/PhysRevA.103.032826.
    # Second order from https://link.springer.com/referencework/10.1007/978-0-387-26308-3.
    if i < 0.5 or j < 0.5:
        return tuple()

    # Magnetic dipole, the corresponding hyperfine constant is A = mu / (IJ) * <T_1>.
    k = f * (f + 1) - i * (i + 1) - j * (j + 1)
    co_a = 0.5 * k
    if i < 1 or j < 1:
        return co_a,

    # Electric quadrupole, the corresponding hyperfine constant is B = 2eQ * <T_2>.
    co_b = (0.75 * k * (k + 1) - j * (j + 1) * i * (i + 1)) / (2 * i * (2 * i - 1) * j * (2 * j - 1))
    if i < 1.5 or j < 1.5:
        return co_a, co_b

    # Magnetic octupole, the corresponding hyperfine constant is C = -Omega * <T_3>.
    co_c = k ** 3 + 4 * k ** 2 + 0.8 * k * (-3 * i * (i + 1) * j * (j + 1) + i * (i + 1) + j * (j + 1) + 3) \
        - 4 * i * (i + 1) * j * (j + 1)
    co_c /= i * (i - 1) * (2 * i - 1) * j * (j - 1) * (2 * j - 1)
    co_c *= 1.25
    if i < 2 or j < 2:
        return co_a, co_b, co_c
    return co_a, co_b, co_c  # Highest implemented order.


def hf_trans(i, j_l, j_u):
    """
    Calculate all allowed hyperfine transitions and their hyperfine coefficients.
    Returns (f_l, f_u, coAl, coBl, coAu, coBu)
    """
    return [[(f_l, f_u), hf_coeff(i, j_l, f_l), hf_coeff(i, j_u, f_u)]
            for f_l in np.arange(abs(i - j_l), (i + j_l + 0.5))
            for f_u in np.arange(abs(i - j_u), (i + j_u + 0.5))
            if abs(f_l - f_u) == 1 or (f_l - f_u == 0 and f_l != 0 and f_u != 0)]


def hf_shift(hyper_l, hyper_u, coeff_l, coeff_u):
    """
    :param hyper_l: The hyperfine structure constants of the lower state (Al, Bl, Cl, ...).
    :param hyper_u: The hyperfine structure constants of the upper state (Au, Bu, Cu, ...).
    :param coeff_l: The coefficients of the lower state to be multiplied by the constants (coAl, coBl, coCl, ...).
    :param coeff_u: The coefficients of the lower state to be multiplied by the constants (coAu, coBu, coCu, ...).
    :returns: The hyperfine structure shift of an optical transition.
    """
    return sum(const * coeff for const, coeff in zip(hyper_u, coeff_u)) \
        - sum(const * coeff for const, coeff in zip(hyper_l, coeff_l))


def hf_int(i, j_l, j_u, transitions):
    """ Calculate relative line intensities """
    return [np.around((2 * f_u + 1) * (2 * f_l + 1) * (wigner_6j(j_l, f_l, i, f_u, j_u, 1, as_sympy=False) ** 2),
                      decimals=9) for (f_l, f_u), *r in transitions]


class Splitter(Model):
    def __init__(self, model, i, j_l, j_u, name):
        super().__init__(model=model)
        self.type = 'Splitter'

        self.i = i
        self.j_l = j_l
        self.j_u = j_u
        self.name = name

        self.racah_indices = []
        self.racah_intensities = []

    def racah(self):
        for i, intensity in zip(self.racah_indices, self.racah_intensities):
            self.vals[i] = intensity


class SplitterSummed(Summed):
    def __init__(self, splitter_models):
        if any(not isinstance(model, Splitter) for model in splitter_models):
            raise TypeError('All models passed to \'SplitterSummed\' must have type \'Splitter\'.')
        super().__init__(splitter_models, labels=['({})'.format(model.name) for model in splitter_models]
                         if len(splitter_models) > 1 else None)

    def racah(self):
        i0 = 0
        for model in self.models:
            for i, intensity in zip(model.racah_indices, model.racah_intensities):
                self.set_val(i0 + i, intensity, force=True)
            i0 += model.size
        self.set_vals(self.vals, force=True)


class Hyperfine(Splitter):
    def __init__(self, model, i, j_l, j_u, name):
        super().__init__(model, i, j_l, j_u, name)
        self.type = 'Hyperfine'

        self.transitions = hf_trans(self.i, self.j_l, self.j_u)
        self.racah_intensities = hf_int(self.i, self.j_l, self.j_u, self.transitions)

        self.n_l = len(self.transitions[0][1])
        self.n_u = len(self.transitions[0][2])
        for i in range(self.n_l):
            self._add_arg('{}l'.format(ascii_uppercase[i]), 0., False, False)
        for i in range(self.n_u):
            self._add_arg('{}u'.format(ascii_uppercase[i]), 0., False, False)

        for i in range(min([self.n_l, self.n_u])):
            fix = '{} / {}'.format('{}u'.format(ascii_uppercase[i]), '{}l'.format(ascii_uppercase[i]))
            self._add_arg('{}_ratio'.format(ascii_uppercase[i]), 0., fix, False)

        for i, (t, intensity) in enumerate(zip(self.transitions, self.racah_intensities)):
            self.racah_indices.append(self._index)
            self._add_arg('int({}, {})'.format(t[0][0], t[0][1]), intensity, i == 0, False)

    def evaluate(self, x, *args, **kwargs):
        const_l = tuple(args[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(args[self.model.size + self.n_l + i] for i in range(self.n_u))
        return np.sum([args[i] * self.model.evaluate(x - hf_shift(const_l, const_u, t[1], t[2]), *args, **kwargs)
                       for i, t in zip(self.racah_indices, self.transitions)], axis=0)

    def min(self):
        const_l = tuple(self.vals[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(self.vals[self.model.size + self.n_l + i] for i in range(self.n_u))
        return self.model.min() + min(hf_shift(const_l, const_u, t[1], t[2]) for t in self.transitions)

    def max(self):
        const_l = tuple(self.vals[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(self.vals[self.model.size + self.n_l + i] for i in range(self.n_u))
        return self.model.max() + max(hf_shift(const_l, const_u, t[1], t[2]) for t in self.transitions)

    def intervals(self):
        const_l = tuple(self.vals[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(self.vals[self.model.size + self.n_l + i] for i in range(self.n_u))
        shifts = [hf_shift(const_l, const_u, t[1], t[2]) for t in self.transitions]
        return merge_intervals([[self.model.min() + shift, self.model.max() + shift] for shift in shifts])

    def racah(self):
        for i, intensity in zip(self.racah_indices, self.racah_intensities):
            self.vals[i] = intensity


class HyperfineMixed(Splitter):
    """
    Hyperfine-mixing model based on https://doi.org/10.1103/PhysRevA.55.2728 [1].
    """
    def __init__(self, model, i, j_l, j_u, name, config):
        super().__init__(model, i, j_l, j_u, name)
        self.type = 'HyperfineMixed'
        self.config = config
        self.states = ['l', 'u']
        to_mhz = 13074.70

        self.enabled = {s: self.config['enabled_{}'.format(s)] for s in self.states}

        self.J = {s: np.array(self.config['J{}'.format(s)]) for s in self.states}

        self.F = {s: get_all_f(self.i, self.J[s]) for s in self.states}

        self.T = {s: np.array(self.config['T{}'.format(s)], dtype=float) * to_mhz for s in self.states}

        self.M = 0. if self.i == 0 else np.sqrt((2 * self.i + 1) * (self.i + 1) / self.i) * self.config['mu']

        self.fs = {s: np.array(self.config['f{}'.format(s)]).flatten() for s in self.states}

        self.mask_J = {s: [np.array([i for i, j in enumerate(self.config['J{}'.format(s)])
                                     if abs(self.i - j) - 0.1 < f < self.i + j + 0.1], dtype=int) for f in self.F[s]]
                       for s in self.states}

        self.W = {s: [np.array([[(-1) ** (self.i + self.J[s][i] + f)
                                 * wigner_6j(self.i, self.J[s][i], f, self.J[s][j], self.i, 1)
                                for j in self.mask_J[s][k]] for i in self.mask_J[s][k]], dtype=float)
                      * self.T[s][np.ix_(self.mask_J[s][k], self.mask_J[s][k])] * self.M
                      for k, f in enumerate(self.F[s])] for s in self.states}  # Eq. (2.5) in [1] without diagonal.

        self.transitions = [[(_m0, _m1), (self.J['l'][_m0], self.J['u'][_m1]), (f0, f1),
                             hf_coeff(self.i, self.J['l'][_m0], f0),
                             hf_coeff(self.i, self.J['u'][_m1], f1)]
                            for i, (m0, f0) in enumerate(zip(self.mask_J['l'], self.F['l']))
                            for j, (m1, f1) in enumerate(zip(self.mask_J['u'], self.F['u']))
                            for _m0 in m0 for _m1 in m1
                            if abs(f1 - f0) < 1.1 and abs(self.J['u'][_m1] - self.J['l'][_m0]) < 1.1]

        self.wt_map = {s: np.array([[(1 - 2 * int(s == 'l')) * int(t[0][int(s == 'u')] == _m
                                                                   and t[2][int(s == 'u')] == f)
                                     for m, f in zip(self.mask_J[s], self.F[s]) for _m in m]
                                    for t in self.transitions], dtype=float) for s in self.states}

        self.racah_intensities = [a(self.i, t[1][0], t[2][0], t[1][1], t[2][1]) for t in self.transitions]

        self.order = {s: [int(min((j // 0.5, self.i // 0.5))) for j in self.J[s]] for s in self.states}

        self.hf_args_map = [[] for _ in self.transitions]
        for s in self.states:
            if self.enabled[s] and self.i > 0:
                for i, j in enumerate(self.J[s]):
                    self._add_arg('FS_{}{}({})'.format(s, i, j), self.fs[s][i], i == 0, False)
            else:
                for i, j in enumerate(self.J[s]):
                    for k in range(self.order[s][i]):
                        for t, m in zip(self.transitions, self.hf_args_map):
                            if t[0][int(s == 'u')] == i:
                                m.append(self._index)
                        self._add_arg('{}_{}{}({})'.format(ascii_uppercase[k], s, i, j), 0., False, False)

        for i, (t, intensity) in enumerate(zip(self.transitions, self.racah_intensities)):
            self.racah_indices.append(self._index)
            self._add_arg('int{}([{}, {}] -> [{}, {}])'.format(i, t[1][0], t[2][0], t[1][1], t[2][1]),
                          intensity, i == 0, False)

    def x0(self, *args):
        x0 = np.zeros(len(self.transitions))
        for s in self.states:
            if self.enabled[s]:
                _x0 = [np.linalg.eigh(  # Eigenvalues and eigenvectors of Eq. (2.5) in [1].
                    np.diag([args[self.p['FS_{}{}({})'.format(s, _m, self.J[s][_m])]] for _m in m]) + w)
                       for m, w in zip(self.mask_J[s], self.W[s])]
                # Eigenvalues are returned in ascending order! Sort based on eigenvectors.
                # Invert order
                # a =               [a, b, c]
                # order =           [2, 0, 1]
                # desired a =       [b, c, a]
                # required order =  [1, 2, 0]
                orders = [np.argmax(np.abs(w[1]), axis=0) for w in _x0]
                orders = [np.array([j for i in range(order.size) for j, o in enumerate(order) if i == o])
                          for w, order in zip(_x0, orders)]
                _x0 = np.concatenate(tuple(w[0][order] for w, order in zip(_x0, orders)))
                _x0 = self.wt_map[s] @ _x0
            else:
                _x0 = np.array([sum(args[i] * coeff for i, coeff in zip(m, t[3 + int(s == 'u')]))
                                for m, t in zip(self.hf_args_map, self.transitions)])
                _x0 *= (1 - 2 * int(s == 'l'))
            x0 += _x0
        # print(f'[{", ".join([str(_x0) for _x0 in x0])}]')
        return x0

    def evaluate(self, x, *args, **kwargs):
        return np.sum([args[i] * self.model.evaluate(x - _x0, *args, **kwargs)
                       for i, _x0 in zip(self.racah_indices, self.x0(*args))], axis=0)

    def min(self):
        return self.model.min() + np.min(self.x0(*self.vals))

    def max(self):
        return self.model.max() + np.max(self.x0(*self.vals))

    def intervals(self):
        return merge_intervals([[self.model.min() + _x0, self.model.max() + _x0] for _x0 in self.x0(*self.vals)])

    def racah(self):
        for i, intensity in zip(self.racah_indices, self.racah_intensities):
            self.vals[i] = intensity
