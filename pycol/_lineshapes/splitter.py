# -*- coding: utf-8 -*-
"""
pycol._lineshapes.splitter

Created on 14.03.2022

@author: Patrick Mueller

Splitter classes for lineshape models.
"""

from string import ascii_uppercase
import numpy as np

from pycol.tools import merge_intervals
from pycol.algebra import wigner_6j
from .base import Model, Summed


def gen_splitter_model(qi: bool = False, hf_mixing: bool = False):
    if qi and hf_mixing:
        pass
    elif qi:
        pass
    elif hf_mixing:
        pass
    else:
        return Hyperfine
    raise ValueError('Specified splitter model not available.')


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
