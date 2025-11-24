# -*- coding: utf-8 -*-
"""
qspec.models._splitter
======================

Splitter classes for lineshape models.
"""

import os
from string import ascii_uppercase
import numpy as np

from qspec.qtypes import *
from qspec.tools import merge_intervals, unit_vector
from qspec.physics import get_f, get_m, lande_f, zeeman_linear, hyper_zeeman_num
from qspec.algebra import clebsch_gordan, wigner_6j, a, b, c
from qspec.models._base import Model, Summed
from qspec.models._spectrum import LorentzQI

__all__ = ['gen_splitter_model', 'get_all_f', 'hf_coeff', 'hf_trans', 'hf_shift', 'hf_int', 'Splitter',
           'SplitterSummed', 'Hyperfine', 'HyperfineQI', 'HyperfineMixed', 'HyperfineZeeman']


def gen_splitter_model(qi: bool = False, hf_mixing: bool = False) -> type['Splitter']:
    """
    :param qi: Whether to consider QI effects.
    :param hf_mixing: Whether to consider HF mixing.
    :returns: The appropriate 'Splitter' model.
    """
    if qi and hf_mixing:
        pass
    elif qi:
        return HyperfineQI
    elif hf_mixing:
        return HyperfineMixed
    else:
        return Hyperfine
    raise ValueError('Specified splitter model not available.')


def get_all_f(i: quant_like, j: quant_like) -> list[quant]:
    """
    :param i: The nuclear spin $I$.
    :param j: The total electron angular momentum $J$.
    :returns: All possible F quantum numbers of an electronic finestructure state.
    """
    i, j = np.array(i, float).flatten(), np.array(j, float).flatten()
    return sorted(set(quant(f + abs(_i - _j)) for _i in i for _j in j for f in range(int(_i + _j - abs(_i - _j) + 1))))


def hf_coeff(i: quant_like, j: quant_like, f: quant_like):
    """
    :param i: The nuclear spin $I$.
    :param j: The total electron angular momentum $J$.
    :param f: The total angular momentum $F$.
    :returns: The tuple of hyperfine coefficients for A and B-factors of a given quantum state.
    """
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


def hf_trans(i: quant_like, j_l: quant_like, j_u: quant_like) -> list[list[tuple]]:
    r"""
    Calculate all allowed hyperfine transitions and their hyperfine coefficients.

    :param i: The nuclear spin $I$.
    :param j_l: The total lower state electron angular momentum $J$.
    :param j_u: The total upper state electron angular momentum $J^\prime$.
    :returns: (transitions)
    """
    return [[(f_l, f_u), hf_coeff(i, j_l, f_l), hf_coeff(i, j_u, f_u)]
            for f_l in get_f(i, j_l) for f_u in get_f(i, j_u)
            if abs(f_u - f_l) < 1.1 and not f_u == f_l == 0]


def hf_shift(hyper_l: Iterable[scalar_like], hyper_u: Iterable[scalar_like],
             coeff_l: Iterable[scalar_like], coeff_u: Iterable[scalar_like]):
    r"""
    :param hyper_l: The hyperfine structure constants of the lower state (Al, Bl, Cl, ...).
    :param hyper_u: The hyperfine structure constants of the upper state (Au, Bu, Cu, ...).
    :param coeff_l: The coefficients of the lower state to be multiplied by the constants (coAl, coBl, coCl, ...).
    :param coeff_u: The coefficients of the lower state to be multiplied by the constants (coAu, coBu, coCu, ...).
    :returns: The hyperfine structure shift of an optical transition.
    """
    return (sum(const * coeff for const, coeff in zip(hyper_u, coeff_u))
            - sum(const * coeff for const, coeff in zip(hyper_l, coeff_l)))


def hfm_lin_shift(hyper_l: Iterable[scalar_like], hyper_u: Iterable[scalar_like],
                  coeff_l: Iterable[scalar_like], coeff_u: Iterable[scalar_like],
                  m_l, m_u, g_l, g_u, b_field):
    r"""
    :param hyper_l: The hyperfine structure constants of the lower state (Al, Bl, Cl, ...).
    :param hyper_u: The hyperfine structure constants of the upper state (Au, Bu, Cu, ...).
    :param coeff_l: The coefficients of the lower state to be multiplied by the constants (coAl, coBl, coCl, ...).
    :param coeff_u: The coefficients of the lower state to be multiplied by the constants (coAu, coBu, coCu, ...).
    :param m_l: The lower state magnetic quantum number.
    :param m_u: The upper state magnetic quantum number.
    :param g_l: The lower F state Lande g-factor.
    :param g_u: The upper F state Lande g-factor.
    :param b_field: The magnetic flux density $B$.
    :returns: The hyperfine structure and Zeeman shift of an optical transition.
    """
    return (sum(const * coeff for const, coeff in zip(hyper_u, coeff_u))
            - sum(const * coeff for const, coeff in zip(hyper_l, coeff_l))
            + zeeman_linear(m_u, g_u, b_field, as_freq=True) - zeeman_linear(m_l, g_l, b_field, as_freq=True))


def hf_int(i: quant_like, j_l: quant_like, j_u: quant_like, transitions: Iterable) -> list[float_like]:
    r"""
    :param i: The nuclear spin $I$.
    :param j_l: The total lower state electron angular momentum $J$.
    :param j_u: The total upper state electron angular momentum $J^\prime$.
    :param transitions: A list of electronic transition properties. The first two properties of each entry should be
     f_l and f_u the total angular momenta of lower and upper state.
    :returns: The relative line intensities.
    """
    return [np.around((2 * f_u + 1) * (2 * f_l + 1) * (wigner_6j(j_l, f_l, i, f_u, j_u, 1, as_sympy=False) ** 2),
                      decimals=9) for (f_l, f_u), *r in transitions]


class Splitter(Model):
    def __init__(self, model: Model, i: quant_like, j_l: quant_like, j_u: quant_like, label: str = None):
        r"""
        The abstract base class for Hyperfine structure models.
        
        :param model: A submodel whose parameters are adopted by this model.
        :param i: The nuclear spin $I$.
        :param j_l: The total lower state electron angular momentum $J$.
        :param j_u: The total upper state electron angular momentum $J^\prime$.
        :param label: A label for the `Splitter` (isotope / isomer / transition).
        """
        super().__init__(model=model)
        self.type = 'Splitter'

        self.i = i
        self.j_l = j_l
        self.j_u = j_u
        if label is None:
            label = 'None'
        self.label = label

        self.racah_indices = []
        self.racah_intensities = []

    def racah(self):
        """
        Set the intensity values to the Racah intensities.

        :returns:
        """
        for i, intensity in zip(self.racah_indices, self.racah_intensities):
            self.vals[i] = intensity

    def racah_fixed(self, fix):
        for i in self.racah_indices:
            self.fixes[i] = fix


class SplitterSummed(Summed):
    def __init__(self, splitter_models: list[Splitter]):
        """
        A `Summed` model of `Splitter` submodels.

        :param splitter_models: The `Splitter` submodels to sum over.
        """
        if any(not isinstance(model, Splitter) for model in splitter_models):
            raise TypeError('All models passed to \'SplitterSummed\' must have type \'Splitter\'.')
        super().__init__(splitter_models, labels=['({})'.format(i if model.label is None else model.label)
                                                  for i, model in enumerate(splitter_models)]
                         if len(splitter_models) > 1 else None)

    def racah(self):
        """
        Set the intensity values of all submodels to the Racah intensities.

        :returns:
        """
        i0 = 0
        for model in self.models:
            for i, intensity in zip(model.racah_indices, model.racah_intensities):
                self.set_val(i0 + i, intensity, force=True)
            i0 += model.size
        self.set_vals(self.vals, force=True)


class Hyperfine(Splitter):
    def __init__(self, model: Model, i: quant_like, j_l: quant_like, j_u: quant_like, label: str = None):
        r"""
        :param model: A submodel whose parameters are adopted by this model.
        :param i: The nuclear spin $I$.
        :param j_l: The total lower state electron angular momentum $J$.
        :param j_u: The total upper state electron angular momentum $J^\prime$.
        :param label: A label for the `Splitter` (isotope / isomer / transition).
        """
        super().__init__(model, i, j_l, j_u, label=label)
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

    def evaluate(self, x: array_like, *args: array_iter, **kwargs: dict) -> ndarray:
        const_l = tuple(args[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(args[self.model.size + self.n_l + i] for i in range(self.n_u))
        return np.sum([args[i] * self.model.evaluate(x - hf_shift(const_l, const_u, t[1], t[2]), *args, **kwargs)
                       for i, t in zip(self.racah_indices, self.transitions)], axis=0)

    def min(self) -> float:
        """
        :returns: A hint for an x-axis minimum for a complete display of the model.
        """
        const_l = tuple(self.vals[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(self.vals[self.model.size + self.n_l + i] for i in range(self.n_u))
        return self.model.min() + min(hf_shift(const_l, const_u, t[1], t[2]) for t in self.transitions)

    def max(self) -> float:
        """
        :returns: A hint for an x-axis maximum for a complete display of the model.
        """
        const_l = tuple(self.vals[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(self.vals[self.model.size + self.n_l + i] for i in range(self.n_u))
        return self.model.max() + max(hf_shift(const_l, const_u, t[1], t[2]) for t in self.transitions)

    def intervals(self) -> list[list[float]]:
        """
        :returns: A list of x-axis intervals for a complete display of the model.
        """
        const_l = tuple(self.vals[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(self.vals[self.model.size + self.n_l + i] for i in range(self.n_u))
        shifts = [hf_shift(const_l, const_u, t[1], t[2]) for t in self.transitions]
        return merge_intervals([[self.model.min() + shift, self.model.max() + shift] for shift in shifts]).tolist()

    def racah(self):
        """
        Set the intensity values to the Racah intensities.

        :returns:
        """
        for i, intensity in zip(self.racah_indices, self.racah_intensities):
            self.vals[i] = intensity


def load_qi(filepath):
    """
    :param filepath: The path to the saved model coefficients.
    :returns: The QI model coefficients A, B and C.
    """
    if not os.path.isfile(filepath):
        return None
    with open(os.path.join(filepath), 'r') as file:
        ret = eval(file.read())
    return ret['a'], ret['b'], ret['c']


class HyperfineQI(Splitter):
    def __init__(self, _: Model, i: quant_like, j_l: quant_like, j_u: quant_like, label: str = None, qi_path: str = None):
        r"""
        A perturbative quantum interference (QI) hyperfine-structure model
        based on [<a href="https://doi.org/10.1103/PhysRevA.87.032504">
        R. C. Brown <i>et al.</i>, Phys. Rev. A <b>87</b>, 032504 (2013)</a>].
        
        :param _: Empty `model` parameter. This hyperfine-structure model always uses `Lorentz` lineshapes.
         To use other lineshapes, combine it with the `Convolve` model.
        :param i: The nuclear spin $I$.
        :param j_l: The total lower state electron angular momentum $J$.
        :param j_u: The total upper state electron angular momentum $J^\prime$.
        :param label: A label for the `Splitter` (isotope / isomer / transition).
        :param qi_path: Specify a directory to save and load the calculated geometric parts of the
         dipole matrix elements for faster repeated calculations.
         These depend only on the quantum numbers `i`, `j_l` and `j_u`.
        """
        super().__init__(LorentzQI(), i, j_l, j_u, label)
        self.type = 'HyperfineQI'
        self.qi_path = qi_path
        self.file = f'qi_{label}.txt'

        self.transitions = hf_trans(self.i, self.j_l, self.j_u)
        self.racah_intensities = [0., ]

        save = False
        if self.qi_path is None or not os.path.isfile(os.path.join(self.qi_path, self.file)):
            print(f'Calculating QI A-matrix of isotope {self.label} ... ')
            self.a_qi = [a(self.i, self.j_l, f_l, self.j_u, f_u, as_sympy=False)
                         for f_l in get_f(self.i, self.j_l) for f_u in get_f(self.i, self.j_u)
                         if abs(f_u - f_l) < 1.1 and not f_u == f_l == 0]
            print(f'Calculating QI B-matrix of isotope {self.label} ... ')
            self.b_qi = [b(self.i, self.j_l, f_l, self.j_u, f_u, as_sympy=False)
                         for f_l in get_f(self.i, self.j_l) for f_u in get_f(self.i, self.j_u)
                         if abs(f_u - f_l) < 1.1 and not f_u == f_l == 0]
            print(f'Calculating QI C-matrix of isotope {self.label} ... ')
            self.c_qi = [[c(self.i, self.j_l, f_l, self.j_u, f1_u, f2_u, as_sympy=False)
                          for i2, f2_u in enumerate(get_f(self.i, self.j_u)) if i2 > i1
                          if abs(f1_u - f_l) < 1.1 and abs(f2_u - f_l) < 1.1
                          and not f1_u == f_l == 0 and not f2_u == f_l == 0]
                         for f_l in get_f(self.i, self.j_l) for i1, f1_u in enumerate(get_f(self.i, self.j_u))]
            if qi_path is not None:
                save = True
        else:
            self.a_qi, self.b_qi, self.c_qi = load_qi(os.path.join(self.qi_path, self.file))
        if save:
            qi_dict = {'a': self.a_qi, 'b': self.b_qi, 'c': self.c_qi}
            with open(os.path.join(self.qi_path, self.file), 'w') as file:
                file.write(str(qi_dict))

        self.transitions_qi = [[[(f_l, f2_u), hf_coeff(self.i, self.j_l, f_l), hf_coeff(self.i, self.j_u, f2_u)]
                                for i2, f2_u in enumerate(get_f(self.i, self.j_u)) if i2 > i1
                                if abs(f1_u - f_l) < 1.1 and abs(f2_u - f_l) < 1.1
                                and not f1_u == f_l == 0 and not f2_u == f_l == 0]
                               for f_l in get_f(self.i, self.j_l) for i1, f1_u in enumerate(get_f(self.i, self.j_u))]

        self.n_l = len(self.transitions[0][1])
        self.n_u = len(self.transitions[0][2])
        for i in range(self.n_l):
            self._add_arg('{}l'.format(ascii_uppercase[i]), 0., False, False)
        for i in range(self.n_u):
            self._add_arg('{}u'.format(ascii_uppercase[i]), 0., False, False)

        for i in range(min([self.n_l, self.n_u])):
            fix = '{} / {}'.format('{}u'.format(ascii_uppercase[i]), '{}l'.format(ascii_uppercase[i]))
            self._add_arg('{}_ratio'.format(ascii_uppercase[i]), 0., fix, False)

        self.racah_indices.append(self._index)
        self._add_arg('geo', 0., False, False)

    def evaluate(self, x: array_like, *args: array_iter, **kwargs: dict) -> ndarray:
        const_l = tuple(args[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(args[self.model.size + self.n_l + i] for i in range(self.n_u))
        return np.sum([(_a + _b * args[self.racah_indices[0]])
                       * self.model.evaluate(x - hf_shift(const_l, const_u, t[1], t[2]), *args, **kwargs)
                       + np.sum([_c * args[self.racah_indices[0]]
                                 * self.model.evaluate_qi(x - hf_shift(const_l, const_u, t[1], t[2]),
                                                          hf_shift(const_l, const_u, _t[1], _t[2])
                                                          - hf_shift(const_l, const_u, t[1], t[2]), *args, **kwargs)
                                 for _t, _c in zip(t_list, c_list)], axis=0)
                       for t, _a, _b, t_list, c_list in zip(
                self.transitions, self.a_qi, self.b_qi, self.transitions_qi, self.c_qi)], axis=0) / np.max(self.a_qi)

    def min(self) -> float:
        """
        :returns: A hint for an x-axis minimum for a complete display of the model.
        """
        const_l = tuple(self.vals[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(self.vals[self.model.size + self.n_l + i] for i in range(self.n_u))
        return self.model.min() + min(hf_shift(const_l, const_u, t[1], t[2]) for t in self.transitions)

    def max(self) -> float:
        """
        :returns: A hint for an x-axis maximum for a complete display of the model.
        """
        const_l = tuple(self.vals[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(self.vals[self.model.size + self.n_l + i] for i in range(self.n_u))
        return self.model.max() + max(hf_shift(const_l, const_u, t[1], t[2]) for t in self.transitions)

    def intervals(self) -> list[list[float]]:
        """
        :returns: A list of x-axis intervals for a complete display of the model.
        """
        const_l = tuple(self.vals[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(self.vals[self.model.size + self.n_l + i] for i in range(self.n_u))
        shifts = [hf_shift(const_l, const_u, t[1], t[2]) for t in self.transitions]
        return merge_intervals([[self.model.min() + shift, self.model.max() + shift] for shift in shifts]).tolist()

    def racah(self):
        """
        Set the intensity values to the Racah intensities.

        :returns:
        """
        self.vals[self.racah_indices[0]] = 0.


class HyperfineMixed(Splitter):
    def __init__(self, model: Model, i: quant_like, j_l: quant_like, j_u: quant_like,
                 label: str = None, config: dict = None):
        r"""
        Hyperfine-mixing model based on [<a href="https://doi.org/10.1103/PhysRevA.55.2728">
        W. R. Johnson <i>et al.</i>, Phys. Rev. A <b>55</b>, 2728 (1997)</a>].
        
        :param model: A submodel whose parameters are adopted by this model.
        :param i: The nuclear spin $I$.
        :param j_l: The total lower state electron angular momentum $J$.
        :param j_u: The total upper state electron angular momentum $J^\prime$.
        :param label: A label for the `Splitter` (isotope / isomer / transition).
        :param config: The configuration of the hyperfine-induced mixing. This is a dictionary such as
         `{'enabled_l'=False, 'enabled_u'=False, 'Jl'=[0.5, ], 'Ju'=[0.5, ],
           'Tl'=[[1.]], 'Tu'=[[1.]], 'fl'=[0., ], 'fu'=[0., ], 'mu'=0.}`,
         where 'enabled_l'/'enabled_u' decide if the corresponding lower/upper J mix, 'Jl'/'Ju' are lists of lower/upper
         state J, including `j_l` and `j_u`, 'Tl'/'Tu' are matrix representations of
         the electronic magnetic dipole operator, see [1], 'fl'/'fu' are lists of initial fine-structure energies
         (can be omitted) and 'mu' is the magnetic dipole moment of the nucleus.
        """
        super().__init__(model, i, j_l, j_u, label)
        self.type = 'HyperfineMixed'
        if config is None:
            config = {'enabled_l':False, 'enabled_u':False, 'Jl':[self.j_l, ], 'Ju':[self.j_u, ],
                      'Tl':[[1.]], 'Tu':[[1.]], 'fl':[0., ], 'fu':[0., ], 'mu':0.}
        self.config = config
        self.states = ['l', 'u']
        to_mhz = 13074.70

        self.enabled = {s: self.config['enabled_{}'.format(s)] for s in self.states}

        self.J = {s: np.array(self.config['J{}'.format(s)]) for s in self.states}

        self.F = {s: get_all_f(self.i, self.J[s]) for s in self.states}

        self.T = {s: np.array(self.config['T{}'.format(s)], dtype=float) * to_mhz for s in self.states}

        self.M = 0. if self.i == 0 else np.sqrt((2 * self.i + 1) * (self.i + 1) / self.i) * self.config['mu']

        self.fs = {s: np.array(self.config.get('f{}'.format(s), np.zeros(self.J[s].shape)))
                   for s in self.states}

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

    def x0(self, *args: array_iter):
        """
        :param args: The function parameters.
        :returns: The peak positions.
        """
        x0 = np.zeros(len(self.transitions))
        for s in self.states:
            if self.enabled[s]:
                _x0 = [np.linalg.eigh(  # Eigenvalues and eigenvectors of Eq. (2.5) in [1].
                    np.diag([args[self.p['FS_{}{}({})'.format(s, _m, self.J[s][_m])]] for _m in m]) + w)
                       for m, w in zip(self.mask_J[s], self.W[s])]
                # Eigenvalues are returned in ascending order! Sort based on eigenvectors.
                # Invert order, e.g.,
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

    def evaluate(self, x: array_like, *args: array_iter, **kwargs: dict) -> ndarray:
        return np.sum([args[i] * self.model.evaluate(x - _x0, *args, **kwargs)
                       for i, _x0 in zip(self.racah_indices, self.x0(*args))], axis=0)

    def min(self) -> float:
        """
        :returns: A hint for an x-axis minimum for a complete display of the model.
        """
        return self.model.min() + np.min(self.x0(*self.vals))

    def max(self) -> float:
        """
        :returns: A hint for an x-axis maximum for a complete display of the model.
        """
        return self.model.max() + np.max(self.x0(*self.vals))

    def intervals(self) -> list[list[float]]:
        """
        :returns: A list of x-axis intervals for a complete display of the model.
        """
        return merge_intervals([[self.model.min() + _x0, self.model.max() + _x0]
                                for _x0 in self.x0(*self.vals)]).tolist()

    def racah(self):
        """
        Set the intensity values to the Racah intensities.

        :returns:
        """
        for i, intensity in zip(self.racah_indices, self.racah_intensities):
            self.vals[i] = intensity


class HyperfineZeeman(Splitter):
    def __init__(self, model: Model, i: quant_like, j_l: quant_like, j_u: quant_like,
                 gi: scalar_like, gj_l: scalar_like, gj_u: scalar_like,
                 f_l: quant_like = None, f_u: quant_like = None, q: array_like = None,
                 rho_ml: array_like = None, rho_as_par: bool = False,
                 scale_par_with_shift: str = None, linear: bool = False, label: str = None):
        r"""
        :param model: A submodel whose parameters are adopted by this model.
        :param i: The nuclear spin $I$.
        :param j_l: The total lower state electron angular momentum $J$.
        :param j_u: The total upper state electron angular momentum $J^\prime$.
        :param gi: The nuclear Lande g-factor.
        :param gj_l: The lower state Lande gj-factor.
        :param gj_u: The upper state Lande gj-factor.
        :param f_l: The total lower state angular momenta $F$.
        :param f_u: The total upper state angular momenta $F^\prime$.
        :param q: The polarization vector $\vec{q}$.
        :param rho_ml: The lower state population. Must be a list of arrays. If `None`, the default list is given by
        `rho_ml = [np.ones(2 * f + 1) for f in f_l]`.
        :param scale_par_with_shift: Choose a parameter that is scaled with the Zeeman shift
         such as the Lorentz width `'Gamma'`.
        :param linear: Whether the Zeeman shift is linear (`True`) or nonlinear (`False`). If nonlinear,
         the Zeeman shift is calculated numerically, decreasing computing speed.
        :param label: A label for the `Splitter` (isotope / isomer / transition).
        """
        super().__init__(model, i, j_l, j_u, label=label)
        self.type = 'HyperfineZeeman'

        self.gi = gi
        self.gj_l = gj_l
        self.gj_u = gj_u

        self.linear = linear

        if f_l is None:
            self.f_l = get_f(self.i, self.j_l)
        elif hasattr(f_l, '__getitem__'):
            self.f_l = np.array(f_l, dtype=float)
        else:
            self.f_l = np.array([f_l], dtype=float)

        if f_u is None:
            self.f_u = get_f(self.i, self.j_u)
        elif hasattr(f_u, '__getitem__'):
            self.f_u = np.array(f_u, dtype=float)
        else:
            self.f_u = np.array([f_u], dtype=float)

        if q is None:
            self.q = np.array([1., 1., 1.])
        elif hasattr(q, '__getitem__'):
            self.q = np.array(q, dtype=float)
        else:
            self.q = unit_vector(int(q) + 1, 3, dtype=float)
        self.q /= np.sum(self.q)

        if rho_ml is None:
            self.rho_ml = [np.ones(int(2 * f + 1), dtype=float) for f in self.f_l]
        else:
            rho_ml = [np.asarray(r, dtype=float) for r in rho_ml]
            self.rho_ml = [r / np.sum(r) * (2 * f + 1) for f, r in zip(self.f_l, rho_ml)]

        self.rho_as_par = rho_as_par
        self.rho_indexes = []

        self.transitions = hf_trans(self.i, self.j_l, self.j_u)
        self.transitions_m = []
        self._racah_intensities = hf_int(self.i, self.j_l, self.j_u, self.transitions)
        self.racah_intensities = []

        self.n_l = len(self.transitions[0][1])
        self.n_u = len(self.transitions[0][2])
        for i in range(self.n_l):
            self._add_arg('{}l'.format(ascii_uppercase[i]), 0., False, False)
        for i in range(self.n_u):
            self._add_arg('{}u'.format(ascii_uppercase[i]), 0., False, False)

        for i in range(min([self.n_l, self.n_u])):
            fix = '{} / {}'.format('{}u'.format(ascii_uppercase[i]), '{}l'.format(ascii_uppercase[i]))
            self._add_arg('{}_ratio'.format(ascii_uppercase[i]), 0., fix, False)

        self._add_arg('B_field', 0., False, False)

        for i, (t, intensity) in enumerate(zip(self.transitions, self._racah_intensities)):
            if t[0][0] not in self.f_l or t[0][1] not in self.f_u:
                continue
            f_l, f_u = t[0]
            index_fl = np.argmax(self.f_l == f_l)
            ml_list = get_m(f_l)
            g_l = lande_f(self.i, self.j_l, f_l, self.gi, self.gj_l)
            g_u = lande_f(self.i, self.j_u, f_u, self.gi, self.gj_u)

            # print(f_l, f_u, g_l, g_u)

            self._index_0 = self._index

            if self.rho_as_par:
                for iq, _q in enumerate(self.q):
                    self._add_arg('q({})'.format(iq - 1), _q, True, False)

            for iq, _q in enumerate(self.q):
                if _q == 0.:
                    continue

                for im, m_l in enumerate(ml_list):

                    population = self.rho_ml[index_fl][im]

                    if (f_l, m_l) not in self.rho_indexes:
                        self._add_arg('rho({}, {})'.format(f_l, m_l), population, [0., 2 * f_l + 1], False)
                        self.rho_indexes.append((f_l, m_l))

                    m_u = quant(m_l + iq - 1)
                    if abs(m_u) > f_u:
                        continue

                    # print('{}: int[({}, {}), ({}, {})]'.format(iq - 1, f_l, m_l, f_u, m_u))
                    cg = clebsch_gordan(f_l, 1, f_u, m_l, iq - 1, m_u) ** 2
                    _intensity = float(3 * intensity * cg / (2 * f_u + 1) * 1e4)

                    if self.rho_as_par:
                        self.racah_intensities.append(population)
                        self.racah_indices.append(self._index_0 + im + 3)
                        self.transitions_m.append([(f_l, f_u), (m_l, m_u), (g_l, g_u), _intensity, t[1], t[2]])
                    else:
                        self.racah_indices.append(self._index)
                        self._add_arg('int[({}, {}), ({}, {})]'.format(f_l, m_l, f_u, m_u), _intensity, True, False)
                        self.racah_intensities.append(_intensity * _q * population)
                        self.transitions_m.append([(f_l, f_u), (m_l, m_u), (g_l, g_u), population, t[1], t[2]])

        self.scale_par_with_shift = ''
        self.index_scale_with_sift = -1
        self.index_shift_scale_par = -1
        if scale_par_with_shift is not None:
            self.scale_par_with_shift = scale_par_with_shift
            self.index_scale_with_sift = self.p.get(self.scale_par_with_shift, -1)
            self.index_shift_scale_par = self._index
            self._add_arg('shift_scale_par', 0., False, False)

    def evaluate(self, x: array_like, *args: array_iter, **kwargs: dict) -> ndarray:
        const_l = tuple(args[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(args[self.model.size + self.n_l + i] for i in range(self.n_u))

        if self.linear:
            return np.sum([args[i] * t[3] * self.model.evaluate(
                x - hfm_lin_shift(const_l, const_u, t[4], t[5], t[1][0], t[1][1], t[2][0], t[2][1],
                                  args[self._index_0 - 1]), *args, **kwargs)
                           for i, t in zip(self.racah_indices, self.transitions_m)], axis=0)
        else:
            hyper_const = [0. if len(const_l) < 1 else const_l[0],
                           0. if len(const_l) < 2 else const_l[1],
                           0. if len(const_l) < 3 else const_l[2]]
            shifts_l = hyper_zeeman_num(self.i, self.j_l, hyper_const,
                                        gi=self.gi, gj=self.gj_l, b_field=args[self._index_0 - 1],
                                        g_n_as_gyro=False, as_freq=True)
            hyper_const = [0. if len(const_u) < 1 else const_u[0],
                           0. if len(const_u) < 2 else const_u[1],
                           0. if len(const_u) < 3 else const_u[2]]
            shifts_u = hyper_zeeman_num(self.i, self.j_u, hyper_const,
                                        gi=self.gi, gj=self.gj_u, b_field=args[self._index_0 - 1],
                                        g_n_as_gyro=False, as_freq=True)


            shifts, args_list = [], []
            intensities = []
            for i, t in zip(self.racah_indices, self.transitions_m):
                il_m = shifts_l[1].index(t[1][0])
                il_f = shifts_l[2][il_m].index(t[0][0])

                iu_m = shifts_u[1].index(t[1][1])
                iu_f = shifts_u[2][iu_m].index(t[0][1])

                shift0 = hf_shift(const_l, const_u, t[4], t[5])
                shifts.append(shifts_u[0][iu_m][:, iu_f] - shifts_l[0][il_m][:, il_f])
                args_list.append([arg for arg in args])

                iq = int(t[1][1] - t[1][0] + 1)
                if self.rho_as_par:
                    intensities.append(args[i] * t[3] * args[self._index_0 + iq])
                else:
                    intensities.append(args[i] * t[3])

                if self.index_scale_with_sift != -1:
                    args_list[-1][self.index_scale_with_sift] *= \
                        (1 + args[self.index_shift_scale_par] * np.abs((shifts[-1] - shift0)))
                    # print(i, 1 + args[self.index_shift_scale_par] * np.abs((shifts[-1] - shift0)))

            # print(args[self._index_0:self._index_0+14])
            return np.sum([intensities[index] * self.model.evaluate(x - shifts[index], *args_list[index], **kwargs)
                           for index, (i, t) in enumerate(zip(self.racah_indices, self.transitions_m))], axis=0)

    def min(self) -> float:
        """
        :returns: A hint for an x-axis minimum for a complete display of the model.
        """
        const_l = tuple(self.vals[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(self.vals[self.model.size + self.n_l + i] for i in range(self.n_u))
        return self.model.min() + np.min([hfm_lin_shift(
            const_l, const_u, t[4], t[5], t[1][0], t[1][1], t[2][0], t[2][1], self.vals[self._index_0 - 1])
            for t in self.transitions_m])

    def max(self) -> float:
        """
        :returns: A hint for an x-axis maximum for a complete display of the model.
        """
        const_l = tuple(self.vals[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(self.vals[self.model.size + self.n_l + i] for i in range(self.n_u))
        return self.model.max() + np.max([hfm_lin_shift(
            const_l, const_u, t[4], t[5], t[1][0], t[1][1], t[2][0], t[2][1], self.vals[self._index_0 - 1])
            for t in self.transitions_m])

    def intervals(self) -> list[list[float]]:
        """
        :returns: A list of x-axis intervals for a complete display of the model.
        """
        const_l = tuple(self.vals[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(self.vals[self.model.size + self.n_l + i] for i in range(self.n_u))
        shifts = [hfm_lin_shift(const_l, const_u, t[4], t[5], t[1][0], t[1][1], t[2][0], t[2][1],
                                self.vals[self._index_0 - 1]) for t in self.transitions_m]
        return merge_intervals([[self.model.min() + shift, self.model.max() + shift] for shift in shifts]).tolist()

    def racah(self):
        """
        Set the intensity values to the Racah intensities.

        :returns:
        """
        for i, intensity in zip(self.racah_indices, self.racah_intensities):
            self.vals[i] = intensity
