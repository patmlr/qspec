# -*- coding: utf-8 -*-
"""
qspec._simulate_cpp
===================

Classes and methods for the 'simulate' module using the Python/C++ interface.
"""

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

from qspec.qtypes import *
from qspec._cpp import *
from qspec import tools
from qspec import get_f, get_m, g_j
import qspec.algebra as al


__all__ = ['Polarization', 'Laser', 'Environment', 'construct_electronic_state', 'construct_hyperfine_state',
           'gen_electronic_ls_state', 'gen_hyperfine_ls_state',
           'density_matrix_diagonal',
           'gen_electronic_state', 'gen_hyperfine_state', 'State', 'DecayMap', 'Atom', 'Interaction']


def sr_generate_y(denominator: np.ndarray, f_theta: np.ndarray, f_phi: np.ndarray,
                  counts: np.ndarray, shape: np.ndarray):
    """
    :param denominator: The denominator of the scattering rate.
    :param f_theta: The numerator with the 'theta-polarization'.
    :param f_phi: The numerator with the 'phi-polarization'.
    :param counts: The number of summands.
    :param shape: The shape of y.
    :returns: The scattering rate.
    """
    y = np.zeros((shape[0] * shape[1], ), dtype=float)  # Allocate memory.
    denominator_p = denominator.ctypes.data_as(c_complex_p)  # Get all pointers to the first elements of the arrays.
    f_theta_p = f_theta.ctypes.data_as(c_complex_p)
    f_phi_p = f_phi.ctypes.data_as(c_complex_p)
    counts_p = counts.ctypes.data_as(c_size_t_p)
    shape_p = shape.ctypes.data_as(c_size_t_p)
    y_p = y.ctypes.data_as(c_double_p)
    dll.sr_generate_y(denominator_p, f_theta_p, f_phi_p, counts_p, shape_p, y_p)  # Modify y "in-place" with C++.
    return y


def _process_q_axis(q_axis: array_like) -> ndarray:
    """
    Preprocess the quantization axis.

    :param q_axis: The quantization axis. Must be an integer in {0, 1, 2} or a 3d-vector.
    :returns: The quantization axis as a 3d-vector.
    """
    q_axis = np.asarray(q_axis, dtype=float)
    if not q_axis.shape:
        if q_axis % 1 != 0 or int(q_axis) not in {0, 1, 2}:
            raise ValueError('q_axis must be an element of {0, 1, 2} or a 3d-vector.')
        else:
            q_axis = tools.unit_vector(int(q_axis), 3)
    elif q_axis.shape != (3,):
        raise ValueError('q_axis must be an integer or a 3d-vector.')
    return q_axis


class CppClass:
    def __init__(self, instance: Union['CppClass', CppClassHandler] = None):
        self.instance = _cast_cpp_type(instance)


def _cast_cpp_type(instance: Union[CppClass, CppClassHandler] = None) -> CppClassHandler:
    if instance is None:
        return None
    if isinstance(instance, CppClass):
        return instance.instance
    return instance


class Polarization(CppClass):
    def __init__(self, vec: array_like = None, q_axis: array_like = 2, vec_as_q: bool = True,
                 instance: Union['Polarization', PolarizationHandler] = None):
        r"""
        Class representing a polarization state of light. The property `Polarization.x` holds the polarization
        vector in cartesian coordinates. The property `Polarization.q` holds the polarization vector
        in the helicity basis $(\vec{\sigma}^-, \vec{\pi}, \vec{\sigma}^+)$ for the given quantization axis `q_axis`.

        :param vec: The complex-valued polarization vector $\vec{\varepsilon}$ of an electromagnetic wave / photon.
         The user input is normalized to a vector with length 1.
         The default value corresponds to linear polarization in $z$-direction,
         such that `Polarization.x = [0, 0, 1]` and `Polarization.q = [0, 1, 0]`.
        :param q_axis: The quantization axis used to transform `Polarization.x` and `Polarization.q` into each other.
         Must be an integer in `{0, 1, 2}` or a 3d-vector. The default is `q_axis = 2` (z-axis).
        :param vec_as_q: Whether `vec` is given in the helicity basis (`True`) or in cartesian coordinates (`False`).
         The default is `True`.
        :param instance: An existing `Polarization` instance. If this is specified, the other parameters are omitted.
        """
        super().__init__(instance)

        if self.instance is None:
            self.instance = dll.polarization_construct()

            if vec is None:
                vec = tools.unit_vector(1, 3, dtype=complex)
            vec = np.asarray(vec, dtype=complex)
            if vec.shape != (3,):
                raise ValueError('vec must be a 3d-vector.')

            q_axis = _process_q_axis(q_axis)
            dll.polarization_init(self.instance, vec, q_axis, vec_as_q)

    def __del__(self):
        dll.polarization_destruct(self.instance)

    def def_q_axis(self, q_axis: array_like = 2, q_fixed: bool = False):
        r"""
        Define the quantization axis. This changes either `Polarization.x` or `Polarization.q`, depending on `q_fixed`.

        :param q_axis: The quantization axis. Must be an integer in `{0, 1, 2}` or a 3d-vector.
         The default is `q_axis = 2` (z-axis).
        :param q_fixed: Whether `q` (`True`) or `x` (`False`) should stay the same with the new quantization axis.
        """
        q_axis = _process_q_axis(q_axis)
        dll.polarization_def_q_axis(self.instance, q_axis, q_fixed)

    @property
    def q_axis(self) -> ndarray:
        r"""
        :returns: The quantization axis.
        """
        return dll.polarization_get_q_axis(self.instance)

    @property
    def x(self) -> ndarray:
        r"""
        :returns: The complex polarization in cartesian coordinates $(\vec{x}, \vec{y}, \vec{z})$.
        """
        return dll.polarization_get_x(self.instance)

    @property
    def q(self) -> ndarray:
        r"""
        :returns: The complex polarization in the helicity basis $(\vec{\sigma}^-, \vec{\pi}, \vec{\sigma}^+)$.
        """
        return dll.polarization_get_q(self.instance)


def _cast_laser_polarization(polarization: Polarization):
    if polarization is None:
        polarization = Polarization()
    elif not isinstance(polarization, Polarization):
        polarization = np.asarray(polarization, dtype=complex)
        if polarization.shape != (3,):
            raise ValueError('\'polarization\' must be a 3d-vector, but has shape {}.'
                             .format(polarization.shape))
        polarization = Polarization(vec=polarization, vec_as_q=False)
    return polarization


class Laser(CppClass):
    def __init__(
            self, freq: array_like, intensity: array_like = 1., polarization: Union[array_like, Polarization] = None,
            k: array_like = None, instance: Union['Laser', LaserHandler] = None):
        r"""
        Class representing a laser that has a frequency $\nu$, an intensity $I = \frac{1}{2}\varepsilon_0c\vec{E}^2$,
        a polarization $\vec{\varepsilon} = \vec{E} / |E|$ and a direction $\vec{k}$.

        :param freq: The frequency $\nu$ of the laser (MHz).
        :param intensity: The intensity $I$ of the laser
         $(\mu\mathrm{W} / \mathrm{mm}^2 = \mathrm{W} / \mathrm{m}^2)$.
        :param polarization: The polarization of the laser $\vec{\varepsilon}$ as a complex vector
         in cartesian coordinates or as a `Polarization` object.
        :param k: The direction of the laser $\vec{k}$. The user input is normalized to a vector with length 1.
        :param instance: An existing `Laser` instance. If this is specified, the other parameters are omitted.
        """
        super().__init__(instance)

        if self.instance is None:
            self.instance = dll.laser_construct()

            self._polarization = _cast_laser_polarization(polarization)

            if k is None:
                k = tools.unit_vector(0, 3)
            k = np.asarray(k, dtype=float)
            if k.shape != (3,):
                raise ValueError('\'k\' must be a 3d-vector, but has shape {}.'.format(k.shape))

            dll.laser_init(self.instance, c_double(freq), c_double(intensity), self._polarization.instance,
                           k.ctypes.data_as(c_double_p))
        else:
            self._polarization = Polarization(instance=dll.laser_get_polarization(self.instance))

    def __del__(self):
        dll.laser_destruct(self.instance)

    @property
    def freq(self):
        r"""
        :returns: The frequency of the laser $\nu$.
        """
        return dll.laser_get_freq(self.instance)

    @freq.setter
    def freq(self, value: scalar):
        r"""
        :param value: The new frequency of the laser $\nu$.
        """
        dll.laser_set_freq(self.instance, c_double(value))

    @property
    def intensity(self):
        r"""
        :returns: The intensity of the laser $I = \frac{1}{2}\varepsilon_0c\vec{E}^2$.
        """
        return dll.laser_get_intensity(self.instance)

    @intensity.setter
    def intensity(self, value: scalar):
        r"""
        :param value: The new intensity of the laser $I = \frac{1}{2}\varepsilon_0c\vec{E}^2$.
        """
        dll.laser_set_intensity(self.instance, c_double(value))

    @property
    def polarization(self):
        r"""
        :returns: The polarization of the laser $\varepsilon$ as a `Polarization` object.
        """
        return self._polarization

    @polarization.setter
    def polarization(self, value: Polarization):
        r"""
        :param value: The new polarization of the laser $\varepsilon$ as a complex vector in cartesian coordinates
         or as a `Polarization` object.
        """
        self._polarization = _cast_laser_polarization(value)
        dll.laser_set_polarization(self.instance, self._polarization.instance)

    @property
    def k(self):
        r"""
        :returns: The normalized direction of the laser $\vec{k}$. The default direction is $(1, 0, 0)$.
        """
        return dll.laser_get_k(self.instance)

    @k.setter
    def k(self, value: array_like):
        r"""
        :param value: The new direction of the laser $\vec{k}$. The user input is normalized to a vector with length 1.
        """
        _value = np.asarray(value, dtype=float)
        if _value.size != 3:
            raise ValueError('Interaction.k must be a 3d-vector, but has shape {}.'.format(_value.shape))
        dll.laser_set_k(self.instance, _value.flatten())

    def get_kpol(self, k: int, electric: bool, q_axis: array_like = 2):
        r"""
        :param k: The multipole order $k \geq 1$.
        :param electric: Whether the electric (`True`) or the magnetic (`False`) field component is returned.
        :param q_axis: The quantization axis. Must be an integer in `{0, 1, 2}` or a 3d-vector.
         The default is `q_axis = 2` (z-axis).
        :returns: The rank `k` electric or magnetic irreducible tensor polarization in the helicity basis
         for the given quantization axis.
        """
        q_axis = _process_q_axis(q_axis)
        vector_cd_p = np.ctypeslib.ndpointer(dtype=complex, shape=(2 * int(k) + 1, ))
        set_restype(dll.laser_get_kpol, vector_cd_p)
        return dll.laser_get_kpol(self.instance, c_bool(electric), c_size_t(k), q_axis.ctypes.data_as(c_double_p))


def _process_hyper_const(hyper_const: array_like) -> ndarray:
    """
    Preprocess the hyperfine-structure constants.

    :param hyper_const: The hyperfine-structure constants. Currently, constants up to the electric quadrupole order are
     supported (A, B). If 'hyper_const' is a scalar,
     it is assumed to be the constant A and the other orders are 0 (MHz).
    :returns: The hyperfine-structure constants as a 3d-vector.
    """
    if hyper_const is None or not hyper_const:
        hyper_const = [0., 0., 0.]
    elif not np.asarray(hyper_const, dtype=float).shape:
        hyper_const = [float(hyper_const), 0., 0.]
    hyper_const = list(hyper_const)
    while len(hyper_const) < 3:
        hyper_const.append(0.)
    return np.asarray(hyper_const, dtype=float)[:3]


# noinspection PyPep8Naming
class Environment(CppClass):
    def __init__(self, E: array_like = None, B: array_like = None,
                 instance: Union['Environment', EnvironmentHandler] = None):
        r"""
        Class representing an electromagnetic environment with static electric field $\vec{E}$
        and magnetic field $\vec{B}$. Currently implemented:

        <ul>
          <li>Nonlinear Zeeman effect (fully diagonalized, dipole transition strengths unaffected)</li>
        </ul>

        :param E: A static electric field $\vec{E}$. <span style="color: #F54927;">(NOT IMPLEMENTED)</span>
        :param B: A static magnetic field $\vec{B}$.
        :param instance: An existing `Environment` instance. If this is specified, the other parameters are omitted.
        """
        super().__init__(instance)

        if self.instance is None:
            self.instance = dll.environment_construct()
            self.E = E
            self.B = B

    def __del__(self):
        dll.environment_destruct(self.instance)

    @property
    def E(self):
        return dll.environment_get_E(self.instance) * dll.environment_get_e_E(self.instance)
    
    @E.setter
    def E(self, value: array_like):
        if value is None:
            dll.environment_set_E(self.instance, np.asarray([1, 0, 0], dtype=float))
            dll.environment_set_E_double(self.instance, c_double(0.))
        else:
            value = np.asarray(value, dtype=float)
            if not value.shape:
                dll.environment_set_E_double(self.instance, value)
            elif value.shape == (3, ):
                dll.environment_set_E(self.instance, value)
            else:
                raise ValueError('E must be a scalar, 3d-vector or None, but has shape {}'.format(value.shape))

    @property
    def B(self):
        return dll.environment_get_B(self.instance) * dll.environment_get_e_B(self.instance)
    
    @B.setter
    def B(self, value: array_like):
        if value is None:
            dll.environment_set_B(self.instance, np.array([0, 0, 1], dtype=float))
            dll.environment_set_B_double(self.instance, c_double(0.))
        else:
            value = np.asarray(value, dtype=float)
            if not value.shape:
                dll.environment_set_B_double(self.instance, value)
            elif value.shape == (3, ):
                dll.environment_set_B(self.instance, value)
            else:
                raise ValueError('B must be a scalar, 3d-vector or None, but has shape {}'.format(value.shape))


class State(CppClass):
    def __init__(self, freq_j: quant_like, parity: Union[str, bool],
                 j: quant_like, i: quant_like, f: quant_like, m: quant_like,
                 ls: quant_iter = None, jj: quant_iter = None, hyper_const: array_like = None,
                 gj: scalar = 0., gi: scalar = 0., label: str = None, instance: Union['State', StateHandler] = None):
        r"""
        Class representing an atomic quantum state $|\mathrm{[label]}\pi JIFm\rangle$.

        :param freq_j: The absolute frequency of the state without the hyperfine structure or the environment (MHz).
        :param parity: The parity $\pi$ of the state is used to check the selection rules.
         It can be either `'even'` (`'e'`, `False`) or `'odd'` (`'o'`, `True`).
        :param j: The electronic total angular momentum quantum number $J$.
        :param i: The nuclear spin quantum number $I$.
        :param f: The total angular momentum quantum number $F$.
        :param m: The $z$-projection quantum number $m$ of the total angular momentum `f`.
        :param ls: A list or a single pair of electronic angular momentum and spin quantum numbers $(l_i, s_i)$.
         If this is a list of LS-pairs, a list of $j_i$ quantum numbers can be specified for the parameter `jj`.
         IMPORTANT: Since $L$ and $S$ are not good quantum numbers, this parameter has no effect, currently.
        :param jj: A list of two electronic total angular momentum quantum numbers $(j_0, j_1)$
         used in the jj-coupling scheme if `ls` is a list.
         IMPORTANT: Since $j_i$ are not good quantum numbers, this parameter has no effect, currently.
        :param hyper_const: A list of the hyperfine-structure constants.
         Currently, constants up to the electric quadrupole order are supported ($A$, $B$).
         If `hyper_const` is a scalar, it is assumed to be the constant $A$ and the other orders are 0 (MHz).
        :param gj: The electronic g-factor $g_J$.
        :param gi: The nuclear g-factor $g_I$.
        :param label: The label of the state. The label is used to link states via a `DecayMap`.
        :param instance: An existing `State` instance. If this is specified, the other parameters are omitted.
        """
        super().__init__(instance)

        if self.instance is None:
            tools.check_half_integer(j, i, f, m)

            if ls is None:
                ls = []
            if not hasattr(ls, '__getitem__'):
                raise ValueError('The parameter \'ls\' must be a list of or a single (l_i, s_i) pair.')
            elif not ls:
                s = np.array([], dtype=float)
                l = np.array([], dtype=float)
            elif hasattr(ls[0], '__getitem__'):
                s = np.array([_ls[1] for _ls in ls], dtype=float, order='C')
                l = np.array([_ls[0] for _ls in ls], dtype=float, order='C')
            else:
                s = np.array([ls[1]], dtype=float)
                l = np.array([ls[0]], dtype=float)
            _s, _l = s.ctypes.data_as(c_double_p), l.ctypes.data_as(c_double_p)

            if jj is None:
                jj = np.full_like(s, -1., dtype=float)
            else:
                jj = np.array(jj, dtype=float, order='C').flatten()
            _jj = jj.ctypes.data_as(c_double_p)

            if isinstance(parity, str):
                if parity not in {'even', 'e', 'odd', 'o'}:
                    raise ValueError('Parameter \'parity\' must be either in {\'even\', \'e\', False}'
                                     ' or in {\'odd\', \'o\', True}.')
                parity = True if parity in {'odd', 'o'} else False
            else:
                parity = bool(parity)

            hyper_const = _process_hyper_const(hyper_const)
            if label is None:
                label = '{}({})'.format(int(np.around(freq_j, decimals=0)), j)

            self.instance = dll.state_construct()
            dll.state_init(self.instance, c_double(freq_j), _s, _l, c_double(j), c_double(i), c_double(f), c_double(m),
                           c_bool(parity), _jj, s.size, hyper_const, c_double(gj), c_double(gi),
                           c_char_p(bytes(label, 'utf-8')))

    def __del__(self):
        dll.state_destruct(self.instance)

    def __repr__(self):
        return '{}({})'.format(self.label, ('{}, ' * 4)[:-2]) \
            .format(*[tools.half_integer_to_str(qn, '/') for qn in [self.j, self.i, self.f, self.m]])

    def reset(self):
        r"""
        Reset the frequency `State.freq` shifted by the `Environment` to a vacuum environment.
        """
        dll.state_update(self.instance)

    def get_shift(self):
        r"""
        The difference between the shifted frequency `State.freq` of the hyperfine-structure state
         plus Environment and the frequency of the fine-structure state `State.freq_j`.
        """
        return dll.state_get_shift(self.instance)

    @property
    def freq_j(self):
        r"""
        :returns: The unperturbed frequency of the fine-structure state.
        """
        return dll.state_get_freq_j(self.instance)

    @freq_j.setter
    def freq_j(self, value: scalar):
        dll.state_set_freq_j(self.instance, c_double(value))

    @property
    def freq(self):
        r"""
        :returns: The shifted frequency of the hyperfine-structure state plus Environment.
        """
        return dll.state_get_freq(self.instance)

    @freq.setter
    def freq(self, value: scalar):
        dll.state_set_freq(self.instance, c_double(value))

    @property
    def j(self):
        r"""
        :returns: The electronic total angular momentum quantum number $J$.
        """
        return dll.state_get_j(self.instance)

    @property
    def i(self):
        r"""
        :returns: The nuclear spin quantum number $I$.
        """
        return dll.state_get_i(self.instance)

    @property
    def f(self):
        r"""
        :returns: The total angular momentum quantum number $F$.
        """
        return dll.state_get_f(self.instance)

    @property
    def m(self):
        r"""
        :returns: The projection quantum number $m$ of the total angular momentum $F$.
        """
        return dll.state_get_m(self.instance)

    @property
    def hyper_const(self):
        r"""
        :returns: The hyperfine-structure constants $(A, B, C)$ as a 3d-vector.
        """
        return dll.state_get_hyper_const(self.instance)

    @hyper_const.setter
    def hyper_const(self, value: array_like):
        r"""
        :param value: The new hyperfine-structure constants. Currently, constants up to the electric quadrupole order
         are supported (A, B). If `hyper_const` is a scalar, it is assumed to be the constant $A$
         and the other orders are 0 (MHz).
        :returns:
        """
        value = _process_hyper_const(value)
        dll.state_get_hyper_const(self.instance, value)

    @property
    def gj(self):
        r"""
        :returns: The electronic g-factor.
        """
        return dll.state_get_gj(self.instance)

    @gj.setter
    def gj(self, value: scalar):
        r"""
        :param value: The new electronic g-factor.
        :returns:
        """
        dll.state_set_gj(self.instance, c_double(value))

    @property
    def gi(self):
        r"""
        :returns: The nuclear g-factor.
        """
        return dll.state_get_gi(self.instance)

    @gi.setter
    def gi(self, value: scalar):
        r"""
        :param value: The new nuclear g-factor.
        :returns:
        """
        dll.state_set_gi(self.instance, c_double(value))

    @property
    def label(self):
        r"""
        :returns: The label of the state. The label is used to link states via `DecayMap`.
        """
        return dll.state_get_label(self.instance).decode('utf-8')

    @label.setter
    def label(self, value: str):
        r"""
        :param value: The label of the state. The label is used to link states via `DecayMap`.
        """
        dll.state_set_label(self.instance, c_char_p(bytes(value, 'utf-8')))


class DecayMap(CppClass):
    def __init__(self, labels: Iterable[tuple] = None, a: Iterable[Union[scalar_like, dict]] = None, k_max: int = 1,
                 instance: Union['DecayMap', DecayMapHandler] = None):
        r"""
        Class linking sets of atomic states via Einstein-A coefficients.
        The class supports all multipole orders for electric and magnetic transitions.

        :param labels: An iterable of pairs of labels connected via Einstein-A coefficients.
         The order of each pair is arbitrary.
        :param a: An Iterable of Einstein-A coefficients $A_{if}$, where the states $|i\rangle$ and $|f\rangle$
         have the labels specified in the list of `labels`. If `a[i]` is a single value,
         only the lowest allowed (not necessarily the dominant!) multipole transition will be used.
         Each `a[i]` can also be a `dict` with keys `'e'` or `'m'` to use either first allowed multipole order, or
          `f'e{k}'` or `f'm{k}'` to define specific rank-$k$ multipole transitions. (MHz).
        :param k_max: The maximum considered multipole order $k_\mathrm{max}$. The default value is 1 (dipole).
        :param instance: An existing `DecayMap` instance. If this is specified, the other parameters are omitted.
        """
        super().__init__(instance)

        if self.instance is None:
            self.instance = dll.decaymap_construct()
            dll.decaymap_set_k_em_max(self.instance, c_size_t(int(k_max)))
            if labels is None:
                labels = []
            self._labels = list(labels)

            single_leading_order = False
            if a is None:
                a = []
            for (s0, s1), _a in zip(self._labels, a):
                if isinstance(_a, dict):
                    ae = [_a.get('e', 0.)]
                    am = [_a.get('m', 0.)]
                    if ae[0] == 0.:
                        ae = [_a.get(f'e{k + 1}', 0.) for k in range(k_max)]
                        if sum(ae) == 0.:
                            ae = [0.]
                    if am[0] == 0.:
                        am = [_a.get(f'm{k + 1}', 0.) for k in range(k_max)]
                        if sum(am) == 0.:
                            am = [0.]
                else:
                    ae = [_a]
                    am = ae
                    single_leading_order = True
                ae = np.array(ae, dtype=float)
                am = np.array(am, dtype=float)

                dll.decaymap_add_decay(self.instance, c_char_p(bytes(s0, 'utf-8')), c_char_p(bytes(s1, 'utf-8')),
                                       ae.ctypes.data_as(c_double_p), c_size_t(int(ae.size)),
                                       am.ctypes.data_as(c_double_p), c_size_t(int(am.size)),
                                       c_bool(single_leading_order))
        else:
            self._labels = self._get_labels()

    def __del__(self):
        dll.decaymap_destruct(self.instance)

    def _get_labels(self):
        r"""
        :returns: The labels used in the C++ class.
        """
        return [(dll.decaymap_get_label(self.instance, 0, i).decode('utf-8'),
                 dll.decaymap_get_label(self.instance, 1, i).decode('utf-8')) for i in range(self.size)]

    @property
    def labels(self):
        r"""
        :returns: A list of pairs of labels connected via Einstein-A coefficients.
         The order of each pair is arbitrary.
        """
        return self._labels

    @property
    def size(self):
        r"""
        :returns: The number of linked sets of atomic states.
        """
        return dll.decaymap_get_size(self.instance)

    @property
    def k_max(self):
        r"""
        :returns: The maximum considered multipole order $k_\matrhrm{max}$. The default value is 1 (dipole).
        """
        return dll.decaymap_get_k_em_max(self.instance)

    # @property
    # def a(self):
    #     """
    #     :returns: The list of Einstein-A coefficients.
    #     """
    #     vector_d_p = np.ctypeslib.ndpointer(dtype=float, shape=(self.size, ))
    #     set_restype(dll.decaymap_get_a, vector_d_p)
    #     return dll.decaymap_get_a(self.instance).tolist()
    #
    # def get_a(self, label_0: str, label_1: str):
    #     """
    #     :returns: The leading order Einstein-A coefficient of the tuple of states `(label_0, label_1)`.
    #     """
    #     return dll.decaymap_get_a_i(
    #         self.instance, c_char_p(bytes(label_0, 'utf-8')), c_char_p(bytes(label_1, 'utf-8')))

    def get_ae(self, label_0: str, label_1: str, k: int):
        r"""
        :param label_0: The label of the first state.
        :param label_1: The label of the second state.
        :param k: The multipole order $k \geq 1$.
        :returns: The Einstein-A coefficient of the tuple of states `(label_0, label_1)`
         of electric multipole order `k`.
        """
        return dll.decaymap_get_ae_ik(
            self.instance, c_char_p(bytes(label_0, 'utf-8')), c_char_p(bytes(label_1, 'utf-8')), c_size_t(int(k)))

    def get_am(self, label_0: str, label_1: str, k: int):
        r"""
        :param label_0: The label of the first state.
        :param label_1: The label of the second state.
        :param k: The multipole order $k \geq 1$.
        :returns: The Einstein-A coefficient of the tuple of states `(label_0, label_1)`
         of magnetic multipole order `k`.
        """
        return dll.decaymap_get_am_ik(
            self.instance, c_char_p(bytes(label_0, 'utf-8')), c_char_p(bytes(label_1, 'utf-8')), c_size_t(int(k)))

    def get_gamma(self, label_0: str, label_1: str, parity_equal: bool):
        r"""
        :param label_0: The label of the first state.
        :param label_1: The label of the second state.
        :param parity_equal: The parity of the transition between two states can be equal (True) or change (False).
        :returns: The FWHM of the transition between the states with labels `(label_0, label_1)`.
        """
        return dll.decaymap_get_gamma(self.instance, c_char_p(bytes(label_0, 'utf-8')),
                                      c_char_p(bytes(label_1, 'utf-8')), c_bool(bool(parity_equal)))


def _gen_label_map(atom):
    """
    :param atom: The atom.
    :returns: A dictionary with state labels as keys
     and an array of the indices of the states with the labels as values.
    """
    if isinstance(atom, int):
        return {'States 0 - {}'.format(atom): np.arange(atom, dtype=int)}
    all_labels = [s.label for s in atom]
    labels = []
    for s in all_labels:
        if s not in labels:
            labels.append(s)
    label_map = {s: np.array([i for i, _s in enumerate(all_labels) if _s == s]) for s in labels}
    return label_map


class Atom(CppClass):
    def __init__(self, states: Iterable[State] = None, decay_map: DecayMap = None, mass: scalar = 0,
                 instance: Union['Atom', AtomHandler] = None):
        r"""
        Class representing an Atom and its inner structure.

        :param states: The states $|\mathrm{[label]}\pi JIFm\rangle$ of the atom.
        :param decay_map: The `DecayMap` connecting the atomic states.
        :param mass: The mass $m$ of the atom (u).
        :param instance: An existing `Atom` instance. If this is specified, the other parameters are omitted.
        """
        super().__init__(instance)

        if self.instance is None:
            self.instance = dll.atom_construct()
            if states is None:
                states = []
            if decay_map is None:
                decay_map = DecayMap()
            self.states = list(states)
            self.decay_map = decay_map
            self.mass = mass
            self.label_map = None
            self.update()
        else:
            self._states = None

    def __del__(self):
        dll.atom_destruct(self.instance)

    def __iter__(self) -> Generator[State, None, None]:
        for state in self.states:
            yield state

    def __getitem__(self, key: int) -> State:
        return self.states[key]

    def update(self, env: Environment = None):
        r"""
        Update the atom.
        :param env: The electromagnetic `Environment` of the atom.
        """
        if env is not None:
            dll.atom_set_env(self.instance, env.instance)
        dll.atom_update(self.instance)
        self.label_map = _gen_label_map(self)

    @property
    def states(self):
        r"""
        :returns: A list of the atom states $|\mathrm{[label]}\pi JIFm\rangle$.
        """
        return self._states

    @states.setter
    def states(self, value: Iterable[State]):
        dll.atom_clear_states(self.instance)
        self._states = list(value)
        for s in self._states:
            dll.atom_add_state(self.instance, s.instance)

    @property
    def decay_map(self):
        r"""
        :returns: The `DecayMap` connecting the atomic states.
        """
        return self._decay_map

    @decay_map.setter
    def decay_map(self, value: DecayMap):
        self._decay_map = value
        if self._decay_map is None:
            self._decay_map = DecayMap()
        dll.atom_set_decay_map(self.instance, value.instance)

    @property
    def mass(self):
        r"""
        :returns: The mass $m$ of the atom (u).
        """
        return dll.atom_get_mass(self.instance)

    @mass.setter
    def mass(self, value: scalar):
        dll.atom_set_mass(self.instance, c_double(value))

    @property
    def size(self):
        r"""
        :returns: The number of states of the atom.
        """
        return dll.atom_get_size(self.instance)

    @property
    def gs(self) -> ndarray:
        r"""
        :returns: The indexes of the states with same label as the first state `Atom.states[0]` in the Atom.
        """
        vector_i_p = np.ctypeslib.ndpointer(dtype=c_size_t, shape=(dll.atom_get_gs_size(self.instance), ))
        set_restype(dll.atom_get_gs, vector_i_p)
        return dll.atom_get_gs(self.instance)

    def get_multipole_types(self, label_0, label_1):
        r"""
        :param label_0: The label of the first state.
        :param label_1: The label of the second state.
        :returns: (multipole_orders) A set of multipole orders contributing to the transition
         between the specified labels in the format `f'e{k}'` and `f'm{k}'`.
        """
        indexes = [[i, j] for i, s0 in enumerate(self.states) for j, s1 in enumerate(self.states)
                   if s0.label == label_0 and s1.label == label_1 and i < j]
        mtypes = set()
        ek_array = self.ek
        mk_array = self.mk
        for ij in indexes:
            for k, (ek, mk) in enumerate(zip(ek_array, mk_array)):
                _ek = ek[ij[0], ij[1]]
                _mk = mk[ij[0], ij[1]]
                if _ek + _mk:
                    mtypes.add(f'{"m" if _mk else "e"}{_ek + _mk}')
        return mtypes

    @property
    def d_em(self):
        r"""
        :returns: Multipole transition strengths ordered by the rank $k$, starting at `k = 1` (dipole).
        """
        matrix_d_p = np.ctypeslib.ndpointer(dtype=float, shape=(self.size, self.size))
        set_restype(dll.atom_get_d_em, matrix_d_p)
        return np.array([dll.atom_get_d_em(self.instance, c_size_t(k + 1)) for k in range(self.decay_map.k_max)])

    @property
    def ek(self):
        r"""
        :returns: A map of the electric multipole orders of the transitions between the states.
        """
        matrix_i_p = np.ctypeslib.ndpointer(dtype=np.int32, shape=(self.size, self.size))
        set_restype(dll.atom_get_ek, matrix_i_p)
        return np.array([dll.atom_get_ek(self.instance, c_size_t(k + 1)) for k in range(self.decay_map.k_max)])

    @property
    def mk(self):
        r"""
        :returns: A map of the magnetic multipole orders of the transitions between the states.
        """
        matrix_i_p = np.ctypeslib.ndpointer(dtype=np.int32, shape=(self.size, self.size))
        set_restype(dll.atom_get_mk, matrix_i_p)
        return np.array([dll.atom_get_mk(self.instance, c_size_t(k + 1)) for k in range(self.decay_map.k_max)])

    @property
    def emk(self):
        r"""
        :returns: A map of the multipole orders of the transitions between the states.
        """
        matrix_i_p = np.ctypeslib.ndpointer(dtype=np.int32, shape=(self.size, self.size))
        set_restype(dll.atom_get_emk, matrix_i_p)
        return np.array([dll.atom_get_emk(self.instance, c_size_t(k + 1)) for k in range(self.decay_map.k_max)])

    @property
    def l0(self):
        a = dll.atom_get_L0(self.instance)
        return np.ctypeslib.as_array(a, (self.size, self.size)).T

    @property
    def l1(self):
        a = dll.atom_get_L1(self.instance)
        return np.ctypeslib.as_array(a, (self.size, self.size)).T

    def get_y0(self, ground_state_labels: Union[Iterable[str], str] = None) -> np.ndarray:
        r"""
        :param ground_state_labels: An Iterable of labels belonging to ground states.
        :returns: (y0) The initial population of the atom.
        """
        if ground_state_labels is None:
            ground_state_labels = [self.states[0].label]
        indices = np.array([i for i, state in enumerate(self) if state.label in ground_state_labels])
        y0 = np.zeros(self.size)
        y0[indices] = 1 / indices.size
        return y0
    
    def get_y0_mc(self, n_samples: int = None, ground_state_labels: Union[Iterable[str], str] = None) -> np.ndarray:
        r"""
        :param n_samples: The number of samples to create.
        :param ground_state_labels: An Iterable of labels belonging to ground states.
        :returns: (y0_mc) The initial population of the atom for the Monte-Carlo master equation solver.
        """
        if ground_state_labels is None:
            ground_state_labels = [self.states[0].label]
        indices = np.array([i for i, state in enumerate(self) if state.label in ground_state_labels])
        y0 = np.zeros((n_samples, self.size), dtype=complex)
        if n_samples % indices.size:
            raise ValueError(f'The number of samples ({n_samples}) must be a multiple of'
                             f' the number of ground states ({indices.size})')
        batch = int(n_samples / indices.size)
        for i, index in enumerate(indices):
            y0[i * batch:(i + 1) * batch, index] \
                = np.exp(np.random.random(size=batch) * 2 * np.pi * 1j)
        return y0

    def get_state_indexes(self, labels: Union[Iterable[str], str] = None,
                          f: Union[Iterable[scalar], scalar] = None,
                          m: Union[Iterable[scalar], scalar] = None) -> np.ndarray:
        """
        :param labels: The labels of the states whose indexes are to be returned.
        :param f: The $F$ quantum numbers whose indexes are to be returned.
        :param m: The $m$ quantum numbers whose indexes are to be returned.
        :returns: The indexes corresponding to the specified labels and F quantum numbers.
        """
        if labels is None:
            labels = set(s.label for s in self.states)

        if f is None:
            f = set(s.f for s in self.states)
        try:
            f = set(f)
        except TypeError:
            f = {f}

        if m is None:
            m = set(s.m for s in self.states)
        try:
            m = set(m)
        except TypeError:
            m = {m}

        return np.array([i for i, s in enumerate(self.states) if s.label in labels
                         and s.f in f and s.m in m], dtype=int)

    def scattering_rate(self, rho: array_like, as_density_matrix: bool = True, k: array_like = None,
                        theta: array_like = None, phi: array_like = None,
                        k_vec: array_like = None, x_vec: array_like = None,
                        i: array_like = None, f: array_like = None, axis: int = 1) -> ndarray:
        r"""
        The photon scattering rate

        $$\begin{aligned}
        \Gamma_\mathrm{sc}\left(\rho, \hat{k}(\theta, \phi), \vec{\varepsilon}\right) &= \sum\limits_{f\in\mathcal{F}}
        \,\sum\limits_{i\in\mathcal{I}}\sum\limits_{Xk_{fi}}\sum\limits_{j\in\mathcal{I}}\sum\limits_{Xk_{fj}}
        \rho_{ji}\sqrt{A_{if}^{Xk_{fi}}A_{jf}^{Xk_{fj}}}\\[1ex]
        &\quad\times\left\lbrace\sum\limits_\lambda (-1)^{k_{fi} + \lambda}\,a_{fi,k_{fi}}^\lambda
        \left[(-\mathrm{i})^{k_{fi} - X_{fi}}\,\vec{\varepsilon}
        \cdot\vec{Y}_{k_{fi}\lambda}^{(X_{fi})}(\hat{k}(\theta, \phi))\right]\right\rbrace\\[1ex]
        &\quad\times\left\lbrace\sum\limits_\lambda (-1)^{k_{fj} + \lambda}\,a_{jf,k_{fj}}^\lambda
        \left[(-\mathrm{i})^{k_{fj} - X_{fj}}\,\vec{\varepsilon}
        \cdot\vec{Y}_{k_{fj}\lambda}^{(X_{fj})}(\hat{k}(\theta, \phi))\right]^\ast\right\rbrace\\[3ex]
        a_{fi,k_{fi}}^\lambda &= (-1)^{F_f + I + k_{fi} + J_i}\sqrt{(2F_f + 1)(2J_i + 1)}
        \langle F_fm_fk_{fi}\lambda|F_im_i\rangle\begin{Bmatrix}J_i & J_f & k_{fi} \\F_f & F_i & I\end{Bmatrix}\\[3ex]
        X_{\!fi} &= \begin{cases}+1, & \text{if electric } (\mathrm{E}k_{fi}) \\
        \ \,0, & \text{if magnetic } (\mathrm{M}k_{fi})\end{cases},
        \end{aligned}$$

        where $A_{if}^{Xk_{fi}}$ is the Einstein coefficient for the electric (magnetic)
        decay $|i\rangle\rightarrow|f\rangle$ and the rank-$k_{fi}$ multipole order,
        $\vec{Y}_{k_{fi}\lambda}^{(X_{fi})}(\hat{k})$ is the vector spherical harmonic
        (see p. 215, Eq. (35) in [<a href=https://doi.org/10.1142/0270>1</a>]), $\rho$ is the density matrix,
        $\hat{k}$ is the direction of emission and $\vec{\varepsilon}$
        is the complex polarization vector of the emitted photons.
        The calculation includes interference terms between all multipole ranks `1 <= k <= Atom.decay_map.k_max`
        if emission directions are chosen through the (`theta`, `phi`) or `k_vec` parameters.
        Parity mixing is currently not considered, such that all rank-$k$ electric (magnetic) transition are pure.
        The emitted polarization can be chosen through the `x_vec` parameter. If `x_vec` is `None`, the above equation
        will be summed over two orthogonal polarization vectors.
        The emitted multipole orders can be limited through the `k` parameter.
        The transitions contributing to the scattering rate can be limited through the index lists `i` and `f`
        of the initial and final states, before and after spontaneous decay, respectively.</br></br>

        If no emission direction is chosen (`theta = phi = k_vec = None`), the above equation simplifies
        to the scattering rate into the complete $4\pi$ solid angle (without polarization selection)

        $$\begin{aligned}
        \Gamma_\mathrm{sc}(\rho) &= \sum\limits_{f\in\mathcal{F}}
        \sum\limits_{i\in\mathcal{I}}\sum\limits_{(Xk)_{fi}}
        \rho_{ii}A_{if}^{Xk_{fi}}\left(a_{fi,k_{fi}}^{m_i - m_f}\right)^2.
        \end{aligned}$$

        :param rho: The density matrix $\rho$ of the `Atom`. Must have the same size as the `Atom`
         along the specified `axis`, and `axis + 1` if `as_density_matrix == True`.
        :param as_density_matrix: Whether 'rho' is a state vector or a density matrix.
        :param k: The rank(s) $k$ of the emitted multipole radiation. If `None`,
         all orders `1 <= k <= Atom.decay_map.k_max` are considered.
        :param theta: The elevation angle of detection relative to the $z$-axis.
        :param phi: The azimuthal angle of detection in the $xy$-plane.
        :param k_vec: An iterable of directional vectors $\hat{k}$ emitted by the atom.
         `k_vec` must have shape `(3, )` or `(m, 3)` .
        :param x_vec: An iterable of complex polarization vectors $\vec{\varepsilon}$ emitted by the atom.
         `x_vec` must have shape `(3, )` or `(m, 3)` or be a `str` indicating a special polarization:
          <ul>
            <li>$e_\theta$: `{'z', 'theta', 't'}`</li>
            <li>$e_\phi$: `{'x', 'y', 'xy', 'phi', 'p'}`</li>
            <li>$\sigma^-$: `{'-', 's-', 'sigma-', 'l'}`</li>
            <li>$\sigma^+$: `{'+', 's+', 'sigma+', 'r'}`</li>
          </ul>
          In these cases, the polarizations are created automatically based on the emission directions.
        :param i: The initially excited state indexes to consider for spontaneous decay.
         If `None`, all states are considered.
        :param f: The final decayed state indexes to consider for spontaneous decay.
         If `None`, all states are considered.
        :param axis: The axis along which the population is aligned in `rho`. The default is `axis = 1`,
         expecting `rho` as an array with shape `(n, Atom.size, Atom.size, ... )`.
        :returns: (Gamma_sc) The scattering rate $\Gamma_\mathrm{sc}$ as an array with shape `(m, n, ...)`.
        """

        rho = np.asarray(rho, dtype=complex)

        if as_density_matrix:
            axes = [j for j in range(len(rho.shape)) if j not in {axis, axis + 1}]
            add_axes = [axis, axis + 1]
        else:
            axes = [j for j in range(len(rho.shape)) if j != axis]
            add_axes = [axis]

        results_shape = tuple(rho.shape[j] for j in axes)

        axes += add_axes
        rho = np.transpose(rho, axes=axes).copy()

        if k is None:
            k = np.array(list(range(1, self.decay_map.k_max + 1)), dtype=int)
        else:
            k = np.array(k, dtype=int).flatten()

        k_size = k.size

        if i is None:
            i = np.arange(self.size, dtype=int)
        else:
            i = np.asarray(i, dtype=int).flatten()
        if f is None:
            f = np.arange(self.size, dtype=int)
        else:
            f = np.asarray(f, dtype=int).flatten()

        as_density_matrix = c_bool(bool(as_density_matrix))
        k = k.ctypes.data_as(c_size_t_p)
        rho = rho.ctypes.data_as(c_complex_p)
        rho_size = c_size_t(int(np.prod(results_shape)))
        i_size = c_size_t(i.size)
        i = i.ctypes.data_as(c_size_t_p)
        f_size = c_size_t(f.size)
        f = f.ctypes.data_as(c_size_t_p)

        if theta is None and phi is None:

            if k_vec is None and x_vec is None:
                results = np.zeros(results_shape, dtype=float, order='C')
                results_c = results.ctypes.data_as(c_double_p)

                error = dll.atom_scattering_rate_4pi(self.instance, results_c, k, k_size,
                                                     rho, rho_size, as_density_matrix,
                                                     i, i_size, f, f_size)

            elif x_vec is None:
                k_vec = _cast_3d_vec(k_vec, dtype=float)

                results_shape = (k_vec.shape[0], ) + results_shape
                results = np.zeros(results_shape, dtype=float, order='C')
                results_c = results.ctypes.data_as(c_double_p)

                k_vec_size = c_size_t(k_vec.shape[0])
                k_vec = k_vec.ctypes.data_as(c_double_p)

                error = dll.atom_scattering_rate_k(self.instance, results_c, k, k_size,
                                                   rho, rho_size, as_density_matrix,
                                                   k_vec, k_vec_size,
                                                   i, i_size, f, f_size)
            elif k_vec is None:
                raise ValueError('Either (\'theta\', \'phi\') or \'k_vec\' must be specified if \'x_vec\' is given.')

            else:
                k_vec = _cast_3d_vec(k_vec, dtype=float)

                results_shape = (k_vec.shape[0], ) + results_shape
                results = np.zeros(results_shape, dtype=float, order='C')
                results_c = results.ctypes.data_as(c_double_p)

                k_vec_size = c_size_t(k_vec.shape[0])
                k_vec = k_vec.ctypes.data_as(c_double_p)

                if isinstance(x_vec, str):
                    x_vec = c_size_t(_cast_x_vec_str(x_vec))
                    error = dll.atom_scattering_rate_qk_xb(self.instance, results_c, k, k_size,
                                                           rho, rho_size, as_density_matrix,
                                                           k_vec, x_vec, k_vec_size,
                                                           i, i_size, f, f_size)

                else:
                    x_vec = _cast_3d_vec(x_vec, dtype=complex)
                    x_vec = x_vec.ctypes.data_as(c_complex_p)

                    error = dll.atom_scattering_rate_qk(self.instance, results_c, k, k_size,
                                                        rho, rho_size, as_density_matrix,
                                                        k_vec, x_vec, k_vec_size,
                                                        i, i_size, f, f_size)

        elif theta is None or phi is None:
            raise ValueError('\'theta\' and \'phi\' must either both be specified or both be None.')

        else:
            if k_vec is not None:
                raise ValueError('Both (\'theta\', \'phi\') and \'k_vec\' were specified.'
                                 ' This is redundant. Use only one of both.')

            theta = np.asarray(theta, dtype=float).flatten()
            phi = np.asarray(phi, dtype=float).flatten()

            if theta.size != phi.size:
                raise ValueError(f'\'theta\' and \'phi\' must have the same size,'
                                 f' but have sizes {theta.size} and {phi.size}.')

            results_shape = (theta.size, ) + results_shape
            results = np.zeros(results_shape, dtype=float, order='C')
            results_c = results.ctypes.data_as(c_double_p)

            k_vec_size = c_size_t(theta.size)
            theta = theta.ctypes.data_as(c_double_p)
            phi = phi.ctypes.data_as(c_double_p)

            if x_vec is None:
                error = dll.atom_scattering_rate_k_tp(self.instance, results_c, k, k_size,
                                                      rho, rho_size, as_density_matrix,
                                                      theta, phi, k_vec_size,
                                                      i, i_size, f, f_size)
            else:
                if isinstance(x_vec, str):
                    x_vec = c_size_t(_cast_x_vec_str(x_vec))
                    error = dll.atom_scattering_rate_qk_tp_xb(self.instance, results_c, k, k_size,
                                                              rho, rho_size, as_density_matrix,
                                                              theta, phi, x_vec, k_vec_size,
                                                              i, i_size, f, f_size)
                else:
                    x_vec = _cast_3d_vec(x_vec, dtype=complex)
                    x_vec = x_vec.ctypes.data_as(c_complex_p)
                    error = dll.atom_scattering_rate_qk_tp(self.instance, results_c, k, k_size,
                                                           rho, rho_size, as_density_matrix,
                                                           theta, phi, x_vec, k_vec_size,
                                                           i, i_size, f, f_size)

        if error == -1:
            raise ValueError('Integer parameter \'1 <= k <= DecayMap.k_em_max\' is out of range.')

        return results

    def _scattering_rate(self, rho: array_like, theta: array_like = None, phi: array_like = None,
                         as_density_matrix: bool = True, i: array_like = None, j: array_like = None, axis: int = 1):
        """
        Scattering rate of the atom into the direction

            e_r = (sin(theta), cos(theta) * sin(phi), cos(theta) * cos(phi))

        where the z-axis is the quantization axis, which is either (0, 0, 1) or the B-field axis.

        :param rho: The density matrix of the atom. Must have the same size as the atom
         along the specified 'axis' and 'axis' + 1.
        :param theta: The elevation angle of detection relative to the quantization axis.
        :param phi: The azimuthal angle of detection relative to the quantization axis.
        :param as_density_matrix: Whether 'rho' is a state vector or a density matrix.
        :param i: The initially excited state indexes to consider for spontaneous decay.
         If None, all states are considered.
        :param j: The final decayed state indexes to consider for spontaneous decay. If None, all states are considered.
        :param axis: The axis along which the population is aligned in 'rho'.
        :returns: The scattering rate of the atom given the population 'rho' (MHz or Events / s).
        :raises ValueError: 'rho' must have the same size as the atom along the specified 'axis'.
        """
        rho = np.asarray(rho)

        if i is None:
            i = np.arange(self.size, dtype=int)
        else:
            i = np.array(i).flatten()
        if j is None:
            j = np.arange(self.size, dtype=int)
        else:
            j = np.array(j).flatten()

        if axis < 0:
            axis += len(rho.shape)

        l0 = np.array([[1. if _i in i and _j in j else 0. for _i in range(self.size)] for _j in range(self.size)])
        l0 *= self.l0

        if theta is None and phi is None:
            if as_density_matrix:
                rho = np.diagonal(rho.real, axis1=axis, axis2=axis + 1)
                if len(rho.shape) > axis + 1:
                    axes = list(range(len(rho.shape)))
                    axes[axis + 1:] = axes[axis:-1]
                    axes[axis] = len(rho.shape) - 1
                    rho = np.transpose(rho, axes=axes)
            axes = [ax for ax in range(axis)]
            if axes:
                l0 = np.expand_dims(l0, axis=axes)
            axes = [axis + ax + 2 for ax in range(len(rho.shape) - axis - 1)]
            if axes:
                l0 = np.expand_dims(l0, axis=axes)

            sr = tools.transform(l0, rho, axis=axis)
            return np.sum(sr, axis=axis)

        elif theta is None or phi is None:
            raise ValueError('\'theta\' and \'phi\' must either both be specified or both be None.')

        if not as_density_matrix:
            rho = tools.vector_to_diag_matrix(rho, axis=axis)

        a_cart = [[al.a_dipole_cart(self.states[_j].i, self.states[_j].j, self.states[_j].f, self.states[_j].m,
                                    self.states[_i].j, self.states[_i].f, self.states[_i].m)
                   * np.sqrt(2 * self.states[_j].i + 1) * np.sqrt(2 * self.states[_j].j + 1)
                   * np.sqrt(self.decay_map.get_ae(self.states[_j].label, self.states[_i].label, 1))
                   if l0[_j, _i] else np.zeros(3, dtype=complex)
                   if _i in i and _j in j else 0. for _i in range(self.size)] for _j in range(self.size)]
        e_theta = tools.e_theta(theta, phi)
        e_phi = tools.e_phi(theta, phi)

        c_theta = np.array([[np.sum(e_theta * _a_cart) for _a_cart in a_cart_list]
                            for a_cart_list in a_cart])
        c_phi = np.array([[np.sum(e_phi * _a_cart) for _a_cart in a_cart_list]
                          for a_cart_list in a_cart])

        ct_theta = np.array([[np.sum(e_theta * np.conj(_a_cart))
                              for _a_cart in a_cart_list] for a_cart_list in a_cart])
        ct_phi = np.array([[np.sum(e_phi * np.conj(_a_cart))
                            for _a_cart in a_cart_list] for a_cart_list in a_cart])

        axes = [ax for ax in range(axis)]
        if axes:
            c_theta = np.expand_dims(c_theta, axis=axes)
            c_phi = np.expand_dims(c_phi, axis=axes)
            ct_theta = np.expand_dims(ct_theta, axis=axes)
            ct_phi = np.expand_dims(ct_phi, axis=axes)
        axes = [axis + ax + 2 for ax in range(len(rho.shape) - axis - 2)]
        if axes:
            c_theta = np.expand_dims(c_theta, axis=axes)
            c_phi = np.expand_dims(c_phi, axis=axes)
            ct_theta = np.expand_dims(ct_theta, axis=axes)
            ct_phi = np.expand_dims(ct_phi, axis=axes)

        sr = (np.sum([np.expand_dims(tools.get_subarray(c_theta, k, axis), axis=axis + 1)
                      * np.expand_dims(tools.get_subarray(rho, k, axis + 1), axis=axis)
                      for k in range(self.size)], axis=0)
              * np.sum([np.expand_dims(tools.get_subarray(ct_theta, k, axis + 1), axis=axis)
                        * np.expand_dims(tools.get_subarray(rho, k, axis), axis=axis + 1)
                        for k in range(self.size)], axis=0))
        
        sr += (np.sum([np.expand_dims(tools.get_subarray(c_phi, k, axis), axis=axis + 1)
                       * np.expand_dims(tools.get_subarray(rho, k, axis + 1), axis=axis)
                       for k in range(self.size)], axis=0)
               * np.sum([np.expand_dims(tools.get_subarray(ct_phi, k, axis + 1), axis=axis)
                         * np.expand_dims(tools.get_subarray(rho, k, axis), axis=axis + 1)
                         for k in range(self.size)], axis=0))
        return 3 / (8 * np.pi) * np.sum(np.sum(sr, axis=axis), axis=axis).real

    def plot(self, indices: array_like = None, draw_bounds: bool = False, show: bool = True):
        """
        Plot a term scheme of the atom.

        :param indices: The indices of the states to be drawn. If `None`, all states are drawn.
        :param draw_bounds: Whether to draw the upper vertical bounds of the states.
        :param show: Whether to show the plot.
        :returns: The $x$ and $y$ positions of the states as well as the distance constant $d$ in plot.
        """
        if indices is None:
            indices = np.argsort([state.freq for state in self.states])
        d = 2
        y_i = 0
        y_dict = {}
        m_max = max(s.m for s in self.states)
        ret_x = {}
        ret_y = {}
        for i in indices:
            s = self.states[i]
            key = (s.label, s.j, s.i, s.f)
            if key not in y_dict.keys():
                if key[:-1] not in [key_1[:-1] for key_1 in y_dict.keys()]:
                    y_i += 3
                y_dict[key] = y_i * d
                y_i += 1
                if draw_bounds:
                    plt.hlines([y_dict[key] + d * 0.05, y_dict[key] + d * 0.95][1:],
                               xmin=-m_max - 0.5, xmax=m_max + 0.5, ls='--', colors='grey')
            x = np.array([s.m - 0.45, s.m + 0.45])  # + x_off[key[:-1]]
            y = np.array([y_dict[key], y_dict[key]])
            ret_x[i] = x
            ret_y[i] = y
            plt.plot(x, y, 'k-')

        x_ticks = np.linspace(-m_max, m_max, int(2 * m_max + 1), dtype=float)
        plt.xticks(x_ticks)
        y_ticks = np.array([[_y[1], (_y[0][0], _y[0][-1])] for _y in y_dict.items()], dtype=object)
        order = np.argsort(y_ticks, axis=0)[:, 0]
        y_ticks = y_ticks[order]
        plt.yticks(y_ticks[:, 0].astype(float), [str(y_l) for y_l in y_ticks[:, 1]])
        plt.xlabel(r'$m$')
        plt.ylabel(r'$(\mathrm{label}, F)$')
        if show:
            plt.tight_layout()
            plt.show()
        return ret_x, ret_y, d


def _cast_t(t) -> (ndarray, int):
    t = np.asarray(t, dtype=float).flatten()
    t.sort()
    t_size = t.size
    if t[0] != 0:
        if t[0] < 0:
            raise ValueError('All times \'t\' must be positive.')
        t = np.ascontiguousarray(np.concatenate([np.zeros(1, dtype=float), t], axis=0), dtype=float)
    return t, t.size, t_size != t.size


def _cast_delta(delta: array_like, m: Optional[int], size: int) -> ndarray:
    """
    :param delta: An array of frequency shifts for the laser(s). 'delta' must be a scalar or a 1d- or 2d-array
     with shapes (., ) or (., #lasers), respectively.
    :param m: The index of the shifted laser. If delta is a 2d-array, 'm' ist omitted.
    :param size: The number of available lasers.
    :returns: An array of vectors with size 'size' containing frequency shifts for the lasers.
    """
    if delta is None:
        return np.zeros((1, size))
    delta = np.array(delta, dtype=float, order='C')
    if len(delta.shape) != 2 and not -size <= m < size:
        raise IndexError('Laser index \'m\' is out of bounds. Must be {} <= m < {} or None but is {}.'
                         .format(-size, size, m))
    error = False
    if len(delta.shape) > 2:
        error = True
    elif len(delta.shape) == 0:
        if m is None:
            delta = np.full((1, size), delta, dtype=float)
        else:
            delta = np.expand_dims(tools.unit_vector(m, size, dtype=float) * delta, axis=0)
    elif len(delta.shape) == 1:
        if m is None:
            delta = delta[:, None] + np.expand_dims(np.zeros(size), axis=0)
        else:
            delta = tools.unit_vector(np.full(delta.size, m, dtype=int), size, dtype=float) \
                    * np.expand_dims(delta, axis=1)
    elif delta.shape[1] != size:
        error = True
    if error:
        raise ValueError('\'delta\' must be a scalar or a 1d- or 2d-array with shapes '
                         '(., ) or (., #lasers), respectively.')
    return delta


def _cast_y0(y0: Optional[array_like], i_solver: int, atom: Atom):
    """
    :param y0: The initial states of an ensemble of n atoms. Depending on the solver, this must have shape
     i_solver = 0 and 1: (#states, ) or (n, #states).
     i_solver = 2: (#states, ), (n, #states) or (n, #states, #states).
    :param i_solver: The index of the solver.
    :param atom: The atom.
    :returns: The correctly shaped 'y0' for the chosen solver and its C type.
    """
    size = atom.size
    gs = atom.gs

    if i_solver == 0:  # Rate equations.
        if y0 is None:
            y0 = np.zeros(size, dtype=float)
            y0[gs] = 1 / gs.size
            return y0, c_double_p

        y0 = np.array(y0, dtype=float, order='C')

        if not y0.shape or y0.shape[-1] != size:
            raise ValueError('\'y0\' must have size {} in the last axis but has shape {}.'.format(size, y0.shape))

        if len(y0.shape) < 2:  # Add the missing sample axis.
            y0 = np.expand_dims(y0, axis=0)
        elif len(y0.shape) > 2:  # Flatten all but the last axis.
            shape = (sum(y0.shape[:-1]), y0.shape[-1])
            y0 = y0.reshape(shape, order='C')
        y0 /= np.sum(y0, axis=-1)[:, None]
        dtype, pointer = float, c_double_p

    elif i_solver == 1:  # Schroedinger equation / MC master.
        if y0 is None:
            y0 = np.zeros(size, dtype=complex)
            y0[gs[0]] = 1
            return y0, c_complex_p

        y0 = np.array(y0, dtype=complex, order='C')

        if not y0.shape or y0.shape[-1] != size:
            raise ValueError('\'y0\' must have size {} in the last axis but has shape {}.'.format(size, y0.shape))

        if len(y0.shape) < 2:  # Add the missing sample axis.
            y0 = np.expand_dims(y0, axis=0)
        elif len(y0.shape) > 2:  # Flatten all but the last axis.
            shape = (sum(y0.shape[:-1]), y0.shape[-1])
            y0 = y0.reshape(shape, order='C')
        y0 /= tools.absolute_complex(y0, axis=-1)[:, None]  # Normalize
        dtype, pointer = complex, c_complex_p

    elif i_solver == 2:  # Master equation.
        if y0 is None:
            y0 = np.zeros(size, dtype=complex)
            y0[gs] = 1 / gs.size
            return np.diag(y0), c_complex_p

        y0 = np.array(y0, dtype=complex, order='C')

        if not y0.shape or (len(y0.shape) <= 2 and y0.shape[-1] != size) \
                or (len(y0.shape) > 2 and y0.shape[-2:] != (size, size)):
            raise ValueError('\'y0\' must have size {} in the last axis if len(y0.shape) <= 2,'
                             ' or shape {} in the last two axes if len(y0.shape) > 2.'
                             .format(size, (size, size), y0.shape))

        if len(y0.shape) == 2:  # Normalize and create n diagonal matrices.
            y0 /= np.sum(y0, axis=1)[:, None]
            y0 = np.array([np.diag(_y0) for _y0 in y0])
        elif len(y0.shape) > 2:  # Flatten all but the last two axes and normalize.
            shape = (sum(y0.shape[:-2]), y0.shape[-2], y0.shape[-1])
            y0 = y0.reshape(shape, order='C')
            y0 /= np.sum(np.diagonal(y0, axis1=-2, axis2=-1), axis=-1)[:, None, None]
        else:  # Normalize and add the missing sample axis.
            y0 = np.expand_dims(np.diag(y0 / np.sum(y0)), axis=0)
        dtype, pointer = complex, c_complex_p

    else:
        raise ValueError('Solver ' + str(i_solver) + ' not available \'i_solver\' must be in {0, 1, 2}.')

    return np.ascontiguousarray(y0, dtype=dtype), pointer


def _cast_v(v: Optional[array_like]):
    """
    :param v: Atom velocities. Must be a scalar or have shape (n, ) or (n, 3). In the first two cases,
     the velocity vector(s) is assumed to be aligned with the x-axis.
    :returns: The correctly shaped velocities with shape (n, 3).
    :raises ValueError: If 'v' has the wrong shape.
    """
    if v is None:
        return np.array([[0, 0, 0]], dtype=float)
    v = np.array(v, dtype=float, order='C')
    if len(v.shape) == 0:
        return np.array([[v, 0, 0]], dtype=float)
    elif len(v.shape) == 1:
        ret = np.zeros((v.size, 3), dtype=float)
        ret[:, 0] = v
        return ret
    elif len(v.shape) == 2:
        if v.shape[1] == 3:
            return v
    raise ValueError('\'v\' must be a scalar or have shape (n, ) or (n, 3) but has shape {}.'.format(v.shape))


def _cast_3d_vec(x: Optional[array_like], dtype: type = float):
    """
    :param x: An array of or a single 3d vector(s). `x` must have shape (3, ) or (n, 3).
    :param dtype: The type of the array elements of `x`.
    :returns: The correctly shaped polarization vectors with shape (n, 3).
    :raises ValueError: If 'x' has the wrong shape.
    """
    while True:
        if x is None:
            break

        x = np.array(x, dtype=dtype, order='C')

        if len(x.shape) == 0:
            break

        elif len(x.shape) == 1:
            if x.size != 3:
                break
            return x[None, :]

        elif len(x.shape) == 2:
            if x.shape[1] == 3:
                return x

        break

    raise ValueError('\'x\' must have shape (3, ) or (n, 3).')


def _cast_x_vec_str(x_vec: str) -> int:
    if x_vec.lower() in {'z', 'theta', 't'}:
        return 0
    elif x_vec.lower() in {'x', 'y', 'xy', 'phi', 'p'}:
        return 1
    elif x_vec.lower() in {'-', 's-', 'sigma-', 'l'}:
        return 2
    elif x_vec.lower() in {'+', 's+', 'sigma+', 'r'}:
        return 3
    else:
        raise ValueError(
            '\'x_vec\' must be an array of complex vectors with the same size as'
            ' (\'theta\', \'phi\') or \'k_vec\', or a str'
            ' in {\'z\', \'theta\', \'t\'} for vectors along the longitudes,'
            ' in {\'x\', \'y\', \'xy\', \'phi\', \'p\'} for vectors along the latitudes,'
            ' in {\'-\', \'s-\', \'sigma-\', \'l\'} for sigma- polarized light or'
            ' in {\'+\', \'s+\', \'sigma+\', \'r\'} for sigma+ polarized light.')


class Interaction(CppClass):
    def __init__(self, atom: Atom = None, lasers: Iterable[Laser] = None, environment: Environment = None,
                 delta_max: scalar = 1e3, controlled: bool = True,
                 instance: Union['Interaction', InteractionHandler] = None):
        r"""
        Class representing an Interaction between an `Atom` and a list of `lasers` in an `Environment`.
        All frequencies are in $\mathrm{MHz}$, all times are in $\mu\mathrm{s}$.

        :param atom: The `Atom` interacting with the `lasers`.
        :param lasers: The lasers interacting with the `Atom`.
        :param environment: The electromagnetic `Environment` of the interaction.
        :param delta_max: The maximum difference between a laser and a transition frequency
         for that transition to be considered laser-driven. The default value is `1000.0` (MHz).
        :param controlled: Whether the ODE solver uses an error controlled stepper or a fixed step size.
         Setting this to True is particularly useful for dynamics where a changing resolution is required.
         However, this comes at the cost of computing time.
         The default step size for the uncontrolled stepper is `dt = 1e-3` (1 ns).
        :param instance: An existing `Interaction` instance. If this is specified, the other parameters are omitted.
        """
        super().__init__(instance)

        if self.instance is None:
            self.instance = dll.interaction_construct()
            self._environment = self._get_environemnt()
            if atom is None:
                atom = Atom()
            if lasers is None:
                lasers = []
            self.atom = atom
            self.lasers = list(lasers)
            self.environment = environment
            self.delta_max = delta_max
            self.controlled = controlled
            self.update()
        else:
            self._environment = self._get_environemnt()
            self._atom = self._get_atom()
            self._lasers = self._get_lasers()
            self.update()

    def __del__(self):
        dll.interaction_destruct(self.instance)

    def _get_environemnt(self):
        r"""
        :return: The environment used in the C++ class.
        """
        return Environment(instance=dll.interaction_get_environment(self.instance))

    def _get_atom(self):
        r"""
        :return: The atom used in the C++ class.
        """
        return Atom(instance=dll.interaction_get_atom(self.instance))

    def _get_lasers(self):
        r"""
        :returns: The lasers used in the C++ class.
        """
        return [Laser(0, instance=dll.interaction_get_laser(self.instance, m))
                for m in range(dll.interaction_get_lasers_size(self.instance))]

    def update(self):
        r"""
        Update the Interaction.
        """
        error = dll.interaction_update(self.instance)
        if error == -1:
            raise ValueError('An electro-magnetic wave cannot oscillate along its k-vector.')

    def resonance_info(self):
        r"""
        Prints the detunings of the base frequencies of the lasers in the given atomic system.
        In particular useful for systems with a hyperfine structure. Here $\Delta = \nu_0 - \nu_\mathrm{L}$.
        """
        dll.interaction_resonance_info(self.instance)

    @property
    def environment(self):
        r"""
        :returns: The `Environment` of the interaction.
        """
        return self._environment

    @environment.setter
    def environment(self, value: Environment):
        if value is None:
            value = Environment()
        self._environment = value
        dll.interaction_set_environment(self.instance, self._environment.instance)

    @property
    def atom(self):
        r"""
        :returns: The `Atom` of the interaction.
        """
        return self._atom

    @atom.setter
    def atom(self, value: Atom):
        self._atom = value
        dll.interaction_set_atom(self.instance, value.instance)

    @property
    def lasers(self):
        r"""
        :returns: The lasers of the interaction.
        """
        return self._lasers

    @lasers.setter
    def lasers(self, value: Iterable[Laser]):
        if value is None:
            value = []
        self._lasers = list(value)
        dll.interaction_clear_lasers(self.instance)
        for laser in self.lasers:
            dll.interaction_add_laser(self.instance, laser.instance)

    @property
    def delta_max(self):
        r"""
        :returns: The maximum difference between a laser and a transition frequency
         for that transition to be considered laser-driven. The default value is `1000.0` (MHz).
        """
        return dll.interaction_get_delta_max(self.instance)

    @delta_max.setter
    def delta_max(self, value: scalar):
        dll.interaction_set_delta_max(self.instance, c_double(value))

    @property
    def controlled(self):
        r"""
        :returns: Whether the ODE solver uses an error controlled stepper or a fixed step size.
         Setting this to True is particularly useful for dynamics where a changing resolution is required.
         However, this comes at the cost of computing time.
         The default step size for the uncontrolled stepper is `dt = 1e-3` (1 ns).
        """
        return dll.interaction_get_controlled(self.instance)

    @controlled.setter
    def controlled(self, value: bool):
        dll.interaction_set_controlled(self.instance, c_bool(value))

    @property
    def dense(self):
        r"""
        :returns: Whether the ODE solver uses an error controlled dense output stepper or a fixed step size.
         If True, this overrides the controlled flag.
         Setting this to True is particularly useful for dynamics where a changing resolution is required.
         However, this comes at the cost of computing time.
         The default step size for the uncontrolled stepper is `dt = 1e-3` (1 ns).
        """
        return dll.interaction_get_dense(self.instance)

    @dense.setter
    def dense(self, value: bool):
        dll.interaction_set_dense(self.instance, c_bool(value))

    @property
    def dt(self):
        r"""
        :returns: The (initial) step size of (controlled) solvers.
        """
        return dll.interaction_get_dt(self.instance)

    @dt.setter
    def dt(self, value: scalar):
        dll.interaction_set_dt(self.instance, c_double(value))

    @property
    def dt_max(self):
        r"""
        :returns: The maximum step size of controlled solvers.
        """
        return dll.interaction_get_dt_max(self.instance)

    @dt_max.setter
    def dt_max(self, value: scalar):
        dll.interaction_set_dt_max(self.instance, c_double(value))

    @property
    def atol(self):
        r"""
        :returns: The absolute error tolerance of controlled solver.
        """
        return dll.interaction_get_atol(self.instance)

    @atol.setter
    def atol(self, value: scalar):
        dll.interaction_set_atol(self.instance, c_double(value))

    @property
    def rtol(self):
        r"""
        :returns: The relative error tolerance of controlled solver.
        """
        return dll.interaction_get_rtol(self.instance)

    @rtol.setter
    def rtol(self, value: scalar):
        dll.interaction_set_rtol(self.instance, c_double(value))

    @property
    def loop(self):
        r"""
        :returns: Whether there are loops formed by the lasers in the atom.
        """
        return dll.interaction_get_loop(self.instance)

    @property
    def time_dependent(self):
        r"""
        :returns: Whether the system hamiltonian is allowed to be time-dependent.
        """
        return dll.interaction_get_time_dependent(self.instance)

    @time_dependent.setter
    def time_dependent(self, value: bool):
        dll.interaction_set_time_dependent(self.instance, c_bool(value))

    @property
    def summap(self):
        r"""
        :returns: An array with shape `(atom.size, atom.size)`, indicating the laser-connected states.
        """
        matrix_i_p = np.ctypeslib.ndpointer(dtype=np.int32, shape=(self.atom.size, self.atom.size))
        set_restype(dll.interaction_get_summap, matrix_i_p)
        return dll.interaction_get_summap(self.instance)

    @property
    def atommap(self):
        r"""
        :returns: A projection matrix $A$ mapping the state frequencies onto the diagonal of the Hamiltonian.
         It holds $H_{ii} \leftarrow \sum\limits_j A_{ij} (\omega_0)_j$.
        """
        matrix_d_p = np.ctypeslib.ndpointer(dtype=float, shape=(self.atom.size, self.atom.size))
        set_restype(dll.interaction_get_atommap, matrix_d_p)
        return dll.interaction_get_atommap(self.instance).T

    @property
    def deltamap(self):
        r"""
        :returns: A projection matrix $B$ mapping the laser frequencies onto the diagonal of the Hamiltonian.
         It holds $H_{ii} \leftarrow s\sum\limits_j B_{im} \omega_m.
        """
        matrix_d_p = np.ctypeslib.ndpointer(dtype=float, shape=(len(self.lasers), self.atom.size))
        set_restype(dll.interaction_get_deltamap, matrix_d_p)
        return dll.interaction_get_deltamap(self.instance).T

    @property
    def history_size(self):
        r"""
        :returns: The length of the history of states visited during the generation of the diagonal maps.
        """
        return dll.interaction_get_n_history(self.instance)

    @property
    def history(self) -> ndarray:
        r"""
        :returns: The history of states visited during the generation of the diagonal maps.
        """
        vector_i_p = np.ctypeslib.ndpointer(dtype=c_size_t, shape=(self.history_size, ))
        set_restype(dll.interaction_get_history, vector_i_p)
        return dll.interaction_get_history(self.instance)

    def delta(self) -> ndarray:
        r"""
        The diagonal of the Hamiltonian without Doppler or additional laser frequency shifts

        $$
        \operatorname{diag}(H_\text{diagonal}) = A\vec{\omega}_0 + B\vec{\omega},
        $$

        where $A$ (`Interaction.atommap`) is a matrix with shape `(atom.size, atom.size)`,
        mapping the atomic frequencies $\vec{\omega}_0$ onto the diagonal of the Hamiltonian, and
        $B$ (`Interaction.deltamap`) is a matrix with shape `(atom.size, #lasers)`,
        mapping the laser frequencies $\vec{\omega}$ onto the diagonal of the Hamiltonian.

        :returns: (Delta) The diagonal of the Hamiltonian without Doppler or additional laser-frequency shifts.
        """
        vector_d_p = np.ctypeslib.ndpointer(dtype=float, shape=(self.atom.size, ))
        set_restype(dll.interaction_get_delta, vector_d_p)
        return dll.interaction_get_delta(self.instance)

    def rabi(self, m: int = None) -> ndarray:
        r"""
        The Rabi-frequency matrices $\Omega_m$ generated by laser `m` with direction $\hat{k}$
        and polarization $\vec{\varepsilon}$

        $$\begin{aligned}
        (\Omega_m)_{ij} &= (\Omega_m)_{ji}^\ast = \frac{E_m}{\hbar}\sum\limits_{(Xk)_{ij}}\sum\limits_\lambda
        (-1)^{k_{ij} + \lambda}\ d_{ij,k_{ij}}^\lambda
        \left[(-\mathrm{i})^{k_{ij} - X_{ij}}\,\vec{\varepsilon}_m
        \cdot\vec{Y}_{k_{ij}\lambda}^{(X_{ij})}(\hat{k}_m)\right]\\[2ex]
        d_{ij,k_{ij}}^\lambda &= d_{ji,k_{ij}}^\lambda = a_{ij,k_{ij}}^\lambda
        \sqrt{A_{ji}^{Xk_{fj}}\,8\pi^2\varepsilon_0\hbar\left(\!\frac{c}{\omega_{ij}}\!\right)^{\!3}}
        \quad\text{for all }\omega_i < \omega_j\\[2ex]
        a_{ij,k_{ij}}^\lambda &= (-1)^{F_i + I + k_{ij} + J_j}\sqrt{(2F_i + 1)(2J_j + 1)}
        \langle F_im_ik_{ij}\lambda|F_jm_j\rangle\begin{Bmatrix}J_j & J_i & k_{ij} \\F_i & F_j & I\end{Bmatrix}\\[2ex]
        X_{\!fi} &= \begin{cases}+1, & \text{if electric } (\mathrm{E}k_{fi}) \\
        \ \,0, & \text{if magnetic } (\mathrm{M}k_{fi})\end{cases}\\[2ex]
        E_m &= \sqrt{\frac{2I_m}{\varepsilon_0c}},
        \end{aligned}$$

        where $A_{if}^{Xk_{fi}}$ is the Einstein coefficient for the electric (magnetic)
        decay $|i\rangle\rightarrow|f\rangle$ and the rank-$k_{fi}$ multipole order,
        $\vec{Y}_{k_{fi}\lambda}^{(X_{fi})}(\hat{k})$ is the vector spherical harmonic
        (see p. 215, Eq. (35) in [<a href=https://doi.org/10.1142/0270>1</a>]),
        $\hat{k}_m$ is the direction, $\vec{\varepsilon}_m$ the polarization,
        and $I_m$ the optical intensity of laser `m`.

        :param m: The laser index `m`. If `None`, an array of the Rabi frequencies of all lasers is returned.
        :returns: (Omega) The Rabi-frequency matrix $\Omega_m$ generated by laser `m` or an array for all lasers
         with shape `(nl, atom.size, atom.size)`, where `nl` is the number of lasers of the `Interaction`
         ($2\pi\,\mathrm{MHz}$).
        """
        matrix_cd_p = np.ctypeslib.ndpointer(dtype=np.complex128, shape=(self.atom.size, self.atom.size))
        set_restype(dll.interaction_get_rabi, matrix_cd_p)
        if m is None:
            return np.array([dll.interaction_get_rabi(self.instance, c_size_t(_m))
                             for _m in range(len(self.lasers))], dtype=complex)
        return dll.interaction_get_rabi(self.instance, c_size_t(m))

    def hamiltonian(self, t: array_like, delta: array_like = None, m: Optional[int] = 0, v: array_like = None):
        r"""
        The interaction Hamiltonian of the coherent light-matter interaction in frequency units

        $$\begin{align}
        H &= H_\text{diagonal} + H_\text{off-diagonal}\\[1ex]
          &= \operatorname{diag}(A\vec{\omega}_0 + B\vec{\omega}^\prime) + \frac{1}{2}\sum\limits_m\Omega_m,\\[2ex]
        \omega^\prime_m &= 2\pi(\nu_m + \Delta_m) \gamma(\vec{v})(1 - \hat{k}_m\cdot\frac{\vec{v}}{c})
        \end{align}$$

        where $A$ (`Interaction.atommap`) is a matrix with shape `(atom.size, atom.size)`,
        mapping the atomic frequencies $\vec{\omega}_0$ onto the diagonal of the Hamiltonian,
        $B$ (`Interaction.deltamap`) is a matrix with shape `(atom.size, #lasers)`,
        mapping the laser frequencies in the rest-frame of the atom $\vec{\omega}^\prime$
        onto the diagonal of the Hamiltonian, $\vec{v}$ is the velocity vector of the atom,
        $\hat{k}_m$ is the direction of laser `m`,
        $\Delta_m$ is the detuning of lasers `m`,
        $\gamma(\vec{v})$ is the time-dilation factor
        <a href="{{ '/doc/functions/physics/gamma_3d.html' | relative_url }}">
        `qspec.physics.gamma_3d`</a>,
        and $\Omega_m$ is the complex Rabi-frequency matrix of laser `m`, see
        <a href="{{ '/doc/functions/simulate/Interaction/rabi.html' | relative_url }}">
        `qspec.simulate.Interaction.rabi`</a>.</br></br>

        If the Hamiltonian is time-dependent, because two or more lasers drive the same transition
        or form loops within the atom, the off-diagonal Hamiltonian becomes

        $$
        (H_\text{off-diagonal})_{ij} = \frac{1}{2}\sum\limits_m(\Omega_m)_{ij}
        \exp\left[\operatorname{sign}(j - i)\,\mathrm{i}t\left(B\vec{\omega}^\prime\,-
        (T_m)_{ij}\,\omega^\prime_m\right)\right]\quad\text{for all }i\neq j,
        $$

        where $T_m$ is a matrix for laser `m` that maps the laser frequency onto
        the transitions $|i\rangle\rightarrow |j\rangle$, whose entries take values $0,\pm 1$,
        depending on the energetic order of the two involved states and if the transition is driven by laser `m`.

        :param t: The times $t$ when to compute the solution. Any array is cast to the shape `(nt, )`,
         where `nt` is the size of the array `t` (&mu;s).
        :param delta: An array of laser frequency shifts $\vec{\Delta}$. `delta` must be a scalar, a 1d- or 2d-array
         with shapes `(n, )` or `(n, nl)`, respectively, where `nl` is the number of lasers of the `Interaction` (MHz).
        :param m: The index of the shifted laser. If `delta` is a 2d-array, `m` ist omitted.
        :param v: Atom velocities $\vec{v}$. Must be a scalar or have shape `(n, )` or `(n, 3)`. In the first two cases,
         the velocity vector(s) are assumed to be aligned with the $x$-axis (m/s).
        :returns: (H) The (time-dependent) Hamiltonian(s) for `n` samples and `nt` times in the shape
         `(n, atom.size, atom.size, nt)` ($2\pi\,\mathrm{MHz}$).
        """
        t, t_size, ex = _cast_t(t)

        if isinstance(delta, np.ndarray) and isinstance(v, np.ndarray):
            if delta.shape == v.shape:
                if delta.flags.f_contiguous:
                    delta = np.ascontiguousarray(delta)
                if v.flags.f_contiguous:
                    v = np.ascontiguousarray(v)
            sample_size = delta.shape[0]
        else:
            delta = _cast_delta(delta, m, len(self.lasers))
            v = _cast_v(v)

            sample_size = max([delta.shape[0], v.shape[0], 1])

            delta = np.array(np.broadcast_to(delta, (sample_size, len(self.lasers))), dtype=float, order='C')
            v = np.array(np.broadcast_to(v, (sample_size, 3)), dtype=float, order='C')

        results = np.zeros((sample_size, self.atom.size, self.atom.size, t_size), dtype=complex)
        dll.interaction_get_hamiltonian(self.instance, t.ctypes.data_as(c_double_p), delta.ctypes.data_as(c_double_p),
                                        v.ctypes.data_as(c_double_p), results.ctypes.data_as(c_complex_p),
                                        c_size_t(t_size), c_size_t(sample_size))
        if ex:
            results = results[:, :, :, 1:]
        return results

    def rates(self, t: array_like, delta: array_like = None, m: Optional[int] = 0, v: array_like = None,
              y0: array_like = None, analytic: bool = False) \
            -> ndarray:
        r"""
        Solver for the rate equations

        $$\begin{aligned}
        \frac{\partial\rho_{ii}}{\partial t}
        &= \sum\limits_j \left[ \left(\sum\limits_m R_{ij}^m\right)(\rho_{jj} - \rho_{ii})
        + \Gamma_{\!ij}\,\rho_{jj} - \Gamma_{\!ji}\,\rho_{ii}\right]\\[2ex]
        R_{ij}^m &= \frac{|\Omega_{ij}^m|^2\,\tilde{\Gamma}_{\!ij}}{
        (\omega^\prime_m - \omega_{ij})^2 + \frac{1}{4}\tilde{\Gamma}_{\!ij}^2}\\[2ex]
        \omega^\prime_m &= 2\pi(\nu_m + \Delta_m) \gamma(\vec{v})(1 - \hat{k}_m\cdot\frac{\vec{v}}{c})\\[2ex]
        \tilde{\Gamma}_{\!ij} &= \sum\limits_u\sum\limits_{Xk_{ui}}A_{iu}^{Xk_{ui}}
        + \sum\limits_v\sum\limits_{Xk_{vj}}A_{jv}^{Xk_{vj}}\\[2ex]
        \Gamma_{\!ij} &= \sum\limits_{Xk_{ij}} (a_{ij,k_{ij}}^{m_j - m_i})^2\,A_{ji}^{Xk_{ij}}\\[2ex]
        a_{ij,k_{ij}}^\lambda &= (-1)^{F_i + I + k_{ij} + J_j}\sqrt{(2F_i + 1)(2J_j + 1)}
        \langle F_im_ik_{ij}\lambda|F_jm_j\rangle\begin{Bmatrix}J_j & J_i & k_{ij} \\F_i & F_j & I\end{Bmatrix}\\[2ex]
        X_{\!ij} &= \begin{cases}+1, & \text{if electric } (\mathrm{E}k_{ij}) \\
        \ \,0, & \text{if magnetic } (\mathrm{M}k_{ij})\end{cases},
        \end{aligned}$$

        where $A_{ji}^{Xk_{ij}}$ is the Einstein coefficient for the electric (magnetic)
        decay $|i\rangle\rightarrow|f\rangle$ and the rank-$k_{fi}$ multipole order,
        $\vec{v}$ is the velocity vector of the atom,
        $\hat{k}_m$ is the direction of laser `m`,
        $\Delta_m$ is the detuning of lasers `m`,
        $\gamma(\vec{v})$ is the time-dilation factor, see
        <a href="{{ '/doc/functions/physics/gamma_3d.html' | relative_url }}">
        `qspec.physics.gamma_3d`</a>
        and $\Omega_{ij}^m$ is the Rabi frequency induced by laser `m`, see
        <a href="{{ '/doc/functions/simulate/Interaction/rabi.html' | relative_url }}">
        `qspec.simulate.Interaction.rabi`</a>. Solutions for `n` samples can be calculated in parallel for `nt` times.

        :param t: The times $t$ when to compute the solution. Any array is cast to the shape `(nt, )`,
         where `nt` is the size of the array `t` (&mu;s).
        :param delta: An array of laser frequency shifts $\vec{\Delta}$. `delta` must be a scalar, a 1d- or 2d-array
         with shapes `(n, )` or `(n, nl)`, respectively, where `nl` is the number of lasers of the `Interaction` (MHz).
        :param m: The index of the shifted laser. If `delta` is a 2d-array, `m` ist omitted.
        :param v: Atom velocities $\vec{v}$. Must be a scalar or have shape `(n, )` or `(n, 3)`. In the first two cases,
         the velocity vector(s) are assumed to be aligned with the $x$-axis (m/s).
        :param y0: The initial state of the `Atom`.
         This must be `None` or have shape `(Atom.size, )` or `(n, Atom.size)`.
         If `None`, all states with the same label as the first `State` in `atom.states` are populated equally.
        :param analytic: Calculate the rate equations analytically through a matrix exponential (`True`)
         or numerically (`False`, default).
        :returns: (diag_rho_t) The integrated rate equations as a real-valued array of shape `(n, atom.size, nt)`.
        """
        t, t_size, ex = _cast_t(t)

        if isinstance(delta, np.ndarray) and isinstance(v, np.ndarray) and isinstance(y0, np.ndarray):
            if delta.shape == v.shape == y0.shape:
                if delta.flags.f_contiguous:
                    delta = np.ascontiguousarray(delta)
                if v.flags.f_contiguous:
                    v = np.ascontiguousarray(v)
                if y0.flags.f_contiguous:
                    y0 = np.ascontiguousarray(y0)
            sample_size = delta.shape[0]
        else:
            delta = _cast_delta(delta, m, len(self.lasers))
            v = _cast_v(v)
            y0, ctype = _cast_y0(y0, 0, self.atom)

            sample_size = max([delta.shape[0], v.shape[0], 1])

            delta = np.array(np.broadcast_to(delta, (sample_size, len(self.lasers))), dtype=float, order='C')
            v = np.array(np.broadcast_to(v, (sample_size, 3)), dtype=float, order='C')
            y0 = np.array(np.broadcast_to(y0, (sample_size, self.atom.size)), dtype=float, order='C')

        results = np.zeros((sample_size, self.atom.size, t_size), dtype=float)
        dll.interaction_rates(self.instance, t.ctypes.data_as(c_double_p), delta.ctypes.data_as(c_double_p),
                              v.ctypes.data_as(c_double_p), y0.ctypes.data_as(c_double_p),
                              results.ctypes.data_as(c_double_p), c_size_t(t_size), c_size_t(sample_size),
                              c_bool(analytic))
        if ex:
            results = results[:, :, 1:]
        return results

    def schroedinger(self, t: array_like, delta: array_like = None, m: Optional[int] = 0, v: array_like = None,
                     y0: array_like = None) \
            -> ndarray:
        r"""
        Solver for the Schr&ouml;dinger equation

        $$\begin{aligned}
        \frac{\partial\vec{\psi}}{\partial t} &= -\mathrm{i}H\vec{\psi},
        \end{aligned}$$

        where the Hamiltonian $H$ (2&pi;&thinsp;MHz) is time-independent whenever possible, see
        <a href="{{ '/doc/functions/simulate/Interaction/hamiltonian.html' | relative_url }}">
        `qspec.simulate.Interaction.hamiltonian`</a>.
        Solutions for `n` samples can be calculated in parallel for `nt` times.

        :param t: The times $t$ when to compute the solution. Any array is cast to the shape `(nt, )`,
         where `nt` is the size of the array `t` (&mu;s).
        :param delta: An array of laser frequency shifts $\vec{\Delta}$. `delta` must be a scalar, a 1d- or 2d-array
         with shapes `(n, )` or `(n, nl)`, respectively, where `nl` is the number of lasers of the `Interaction` (MHz).
        :param m: The index of the shifted laser. If `delta` is a 2d-array, `m` ist omitted.
        :param v: Atom velocities $\vec{v}$. Must be a scalar or have shape `(n, )` or `(n, 3)`. In the first two cases,
         the velocity vector(s) are assumed to be aligned with the $x$-axis (m/s).
        :param y0: The initial state of the `Atom`.
         This must be `None` or have shape `(Atom.size, )` or `(n, Atom.size)`.
         If `None`, all states with the same label as the first `State` in `atom.states` are populated equally.
        :returns: (psi_t) The integrated Schr&ouml;dinger equation as a complex-valued array of shape `(n, Atom.size, nt)`.
        """
        t, t_size, ex = _cast_t(t)

        if isinstance(delta, np.ndarray) and isinstance(v, np.ndarray) and isinstance(y0, np.ndarray):
            if delta.shape == v.shape == y0.shape:
                if delta.flags.f_contiguous:
                    delta = np.ascontiguousarray(delta)
                if v.flags.f_contiguous:
                    v = np.ascontiguousarray(v)
                if y0.flags.f_contiguous:
                    y0 = np.ascontiguousarray(y0)
            sample_size = delta.shape[0]
        else:
            delta = _cast_delta(delta, m, len(self.lasers))
            v = _cast_v(v)
            y0, ctype = _cast_y0(y0, 1, self.atom)

            sample_size = max([delta.shape[0], v.shape[0], y0.shape[0], 1])

            delta = np.array(np.broadcast_to(delta, (sample_size, len(self.lasers))), dtype=float, order='C')
            v = np.array(np.broadcast_to(v, (sample_size, 3)), dtype=float, order='C')
            y0 = np.array(np.broadcast_to(y0, (sample_size, self.atom.size)), dtype=complex, order='C')

        results = np.zeros((sample_size, self.atom.size, t_size), dtype=complex)
        dll.interaction_schroedinger(self.instance, t.ctypes.data_as(c_double_p), delta.ctypes.data_as(c_double_p),
                                     v.ctypes.data_as(c_double_p), y0.ctypes.data_as(c_complex_p),
                                     results.ctypes.data_as(c_complex_p), c_size_t(t_size), c_size_t(sample_size))
        if ex:
            results = results[:, :, 1:]
        return results

    def master(self, t: array_like, delta: array_like = None, m: Optional[int] = 0, v: array_like = None,
               y0: array_like = None) \
            -> ndarray:
        r"""
        Solver for the master equation

        $$\begin{aligned}
        \frac{\partial\rho}{\partial t} &= -i[H, \rho]
        + \sum\limits_{i,j} \Gamma_{\!ij}\mathcal{D}[\sigma_{ij}]\rho\\[2ex]
        \mathcal{D}[\sigma]\rho &\coloneqq \sigma\rho\sigma^\dagger
        - \frac{1}{2}(\sigma^\dagger\sigma\rho + \rho\sigma^\dagger\sigma),\quad\sigma_{ij} = |i\rangle\langle j|\\[2ex]
        \Gamma_{\!ij} &= \sum\limits_{Xk_{ij}} (a_{ij,k_{ij}}^{m_j - m_i})^2\,A_{ji}^{Xk_{ij}}\\[2ex]
        a_{ij,k_{ij}}^\lambda &= (-1)^{F_i + I + k_{ij} + J_j}\sqrt{(2F_i + 1)(2J_j + 1)}
        \langle F_im_ik_{ij}\lambda|F_jm_j\rangle\begin{Bmatrix}J_j & J_i & k_{ij} \\F_i & F_j & I\end{Bmatrix}\\[2ex]
        X_{\!ij} &= \begin{cases}+1, & \text{if electric } (\mathrm{E}k_{ij}) \\
        \ \,0, & \text{if magnetic } (\mathrm{M}k_{ij})\end{cases},
        \end{aligned}$$

        where the Hamiltonian $H$ (2&pi;&thinsp;MHz) is time-independent whenever possible, see
        <a href="{{ '/doc/functions/simulate/Interaction/hamiltonian.html' | relative_url }}">
        `qspec.simulate.Interaction.hamiltonian`</a>.
        Solutions for `n` samples can be calculated in parallel for `nt` times.

        :param t: The times $t$ when to compute the solution. Any array is cast to the shape `(nt, )`,
         where `nt` is the size of the array `t` (&mu;s).
        :param delta: An array of laser frequency shifts $\vec{\Delta}$. `delta` must be a scalar, a 1d- or 2d-array
         with shapes `(n, )` or `(n, nl)`, respectively, where `nl` is the number of lasers of the `Interaction` (MHz).
        :param m: The index of the shifted laser. If `delta` is a 2d-array, `m` ist omitted.
        :param v: Atom velocities $\vec{v}$. Must be a scalar or have shape `(n, )` or `(n, 3)`. In the first two cases,
         the velocity vector(s) are assumed to be aligned with the $x$-axis (m/s).
        :param y0: The initial state / density matrix of the `Atom`.
         This must be `None` or have shape `(Atom.size, )`, `(n or 1, Atom.size)` or `(n or 1, Atom.size, Atom.size)`.
         If `None`, all states with the same label as the first `State` in `atom.states` are populated equally.
        :returns: (rho_t) The integrated master equation as a complex-valued array of shape `(n, Atom.size, Atom.size, nt)`.
        """
        t, t_size, ex = _cast_t(t)

        if isinstance(delta, np.ndarray) and isinstance(v, np.ndarray) and isinstance(y0, np.ndarray):
            if delta.shape == v.shape == y0.shape:
                if delta.flags.f_contiguous:
                    delta = np.ascontiguousarray(delta)
                if v.flags.f_contiguous:
                    v = np.ascontiguousarray(v)
                if y0.flags.f_contiguous:
                    y0 = np.ascontiguousarray(y0)
            sample_size = delta.shape[0]
        else:
            delta = _cast_delta(delta, m, len(self.lasers))
            v = _cast_v(v)
            y0, ctype = _cast_y0(y0, 2, self.atom)

            sample_size = max([delta.shape[0], v.shape[0], 1])

            delta = np.array(np.broadcast_to(delta, (sample_size, len(self.lasers))), dtype=float, order='C')
            v = np.array(np.broadcast_to(v, (sample_size, 3)), dtype=float, order='C')
            y0 = np.array(np.broadcast_to(y0, (sample_size, self.atom.size, self.atom.size)), dtype=complex, order='C')

        results = np.zeros((sample_size, self.atom.size, self.atom.size, t_size), dtype=complex)
        dll.interaction_master(self.instance, t.ctypes.data_as(c_double_p), delta.ctypes.data_as(c_double_p),
                               v.ctypes.data_as(c_double_p), y0.ctypes.data_as(c_complex_p),
                               results.ctypes.data_as(c_complex_p), c_size_t(t_size), c_size_t(sample_size))
        if ex:
            results = results[:, :, :, 1:]
        return results

    def mc_master(self, t: array_like, delta: array_like = None, m: Optional[int] = 0, v: array_like = None,
                  y0: array_like = None, dynamics: bool = False, ntraj: int = 500, as_density_matrix: bool = True) \
            -> (ndarray, ndarray):
        r"""
        Solver for the Monte-Carlo master equation, which is the Schr&ouml;dinger equation
        with a non-hermitian Hamiltonian

        $$\begin{aligned}
        \frac{\partial\vec{\psi}}{\partial t} &= -\mathrm{i}(H + H_\text{leaky})\vec{\psi},\qquad
        \rho_{ij} = \lim\limits_{n\rightarrow\infty}\frac{1}{n}\sum\limits_{s=1}^n
        \langle i|\psi_s\rangle\langle\psi_s|j\rangle\\[2ex]
        (H_\text{leaky})_{jj} &= -\frac{\mathrm{i}}{2}\sum\limits_i\Gamma_{ij},\qquad
        (H_\text{leaky})_{ij}\big|_{i\neq j} = 0\\[2ex]
        \Gamma_{\!ij} &= \sum\limits_{Xk_{ij}} (a_{ij,k_{ij}}^{m_j - m_i})^2\,A_{ji}^{Xk_{ij}}\\[2ex]
        a_{ij,k_{ij}}^\lambda &= (-1)^{F_i + I + k_{ij} + J_j}\sqrt{(2F_i + 1)(2J_j + 1)}
        \langle F_im_ik_{ij}\lambda|F_jm_j\rangle\begin{Bmatrix}J_j & J_i & k_{ij} \\F_i & F_j & I\end{Bmatrix}\\[2ex]
        X_{\!ij} &= \begin{cases}+1, & \text{if electric } (\mathrm{E}k_{ij}) \\
        \ \,0, & \text{if magnetic } (\mathrm{M}k_{ij})\end{cases},
        \end{aligned}$$

        where the Hamiltonian $H$ (2&pi;&thinsp;MHz) is time-independent whenever possible, see
        <a href="{{ '/doc/functions/simulate/Interaction/hamiltonian.html' | relative_url }}">
        `qspec.simulate.Interaction.hamiltonian`</a>, $H_\text{leaky}$ is an imaginary diagonal operator,
        and $\rho$ is the density matrix in the limit of an infinite number of samples.
        Solutions for `n` samples can be calculated in parallel for `nt` times.
        The complexity of the Monte-Carlo approach only scales linearly with `Atom.size`,
        as compared with that of the exact master equation, scaling with `Atom.size ** 2`.

        :param t: The times $t$ when to compute the solution. Any array is cast to the shape `(nt, )`,
         where `nt` is the size of the array `t` (&mu;s).
        :param delta: An array of laser frequency shifts $\vec{\Delta}$. `delta` must be a scalar, a 1d- or 2d-array
         with shapes `(n, )` or `(n, nl)`, respectively, where `nl` is the number of lasers of the `Interaction` (MHz).
        :param m: The index of the shifted laser. If `delta` is a 2d-array, `m` ist omitted.
        :param v: Atom velocities $\vec{v}$. Must be a scalar or have shape `(n, )` or `(n, 3)`. In the first two cases,
         the velocity vector(s) are assumed to be aligned with the $x$-axis (m/s).
        :param y0: The initial state of the `Atom`.
         This must be `None` or have shape `(Atom.size, )` or `(n, Atom.size)`.
         If `None`, all states with the same label as the first `State` in `atom.states` are populated equally.
        :param dynamics: Whether to compute the momentum dynamics of the photon-atom interactions.
        :param ntraj: The number of samples `n` to compute if no samples were given with `delta`, `v`, or `y0`.
        :param as_density_matrix: Whether the result is returned as density matrices or as state vectors.
        :returns: (rho_t) The integrated Monte-Carlo master equation as a complex-valued array of shape
         `(n, Atom.size, Atom.size, nt)` or `(n, Atom.size, nt)`.
        """
        if self.controlled:
            raise ValueError('Controlled or Dense steppers are not supported for \'mc_master\'.'
                             ' Decrease the step size if necessary.')
        if dynamics and self.atom.mass <= 0:
            raise ValueError('To simulate mechanical dynamics, the mass of the atom must be specified.')

        t, t_size, ex = _cast_t(t)

        if isinstance(delta, np.ndarray) and isinstance(v, np.ndarray) and isinstance(y0, np.ndarray):
            if delta.shape == v.shape == y0.shape:
                if delta.flags.f_contiguous:
                    delta = np.ascontiguousarray(delta)
                if v.flags.f_contiguous:
                    v = np.ascontiguousarray(v)
                if y0.flags.f_contiguous:
                    y0 = np.ascontiguousarray(y0)
            sample_size = delta.shape[0]
        else:
            delta = _cast_delta(delta, m, len(self.lasers))
            v = _cast_v(v)
            y0, ctype = _cast_y0(y0, 1, self.atom)

            sample_size = max([delta.shape[0], v.shape[0], y0.shape[0], 1])
            if sample_size == 1:
                sample_size = ntraj

            delta = np.array(np.broadcast_to(delta, (sample_size, len(self.lasers))), dtype=float, order='C')
            v = np.array(np.broadcast_to(v, (sample_size, 3)), dtype=float, order='C')
            y0 = np.array(np.broadcast_to(y0, (sample_size, self.atom.size)), dtype=complex, order='C')

        results = np.zeros((sample_size, self.atom.size, t_size), dtype=complex)
        dll.interaction_mc_master(self.instance, t.ctypes.data_as(c_double_p), delta.ctypes.data_as(c_double_p),
                                  v.ctypes.data_as(c_double_p), y0.ctypes.data_as(c_complex_p), c_bool(dynamics),
                                  results.ctypes.data_as(c_complex_p), c_size_t(t_size), c_size_t(sample_size))

        if ex:
            results = results[:, :, 1:]

        if as_density_matrix:
            results = results[:, :, None, :] * results[:, None, :, :].conj()

        return results, v

    def scattering_rate(self, rho: array_like, as_density_matrix: bool = True, k: array_like = None,
                        theta: array_like = None, phi: array_like = None,
                        k_vec: array_like = None, x_vec: array_like = None,
                        i: array_like = None, f: array_like = None, axis: int = 1) -> ndarray:
        r"""
        The photon scattering rate

        $$\begin{aligned}
        \Gamma_\mathrm{sc}\left(\rho, \hat{k}(\theta, \phi), \vec{\varepsilon}\right) &= \sum\limits_{f\in\mathcal{F}}
        \,\sum\limits_{i\in\mathcal{I}}\sum\limits_{Xk_{fi}}\sum\limits_{j\in\mathcal{I}}\sum\limits_{Xk_{fj}}
        \rho_{ji}\sqrt{A_{if}^{Xk_{fi}}A_{jf}^{Xk_{fj}}}\\[1ex]
        &\quad\times\left\lbrace\sum\limits_\lambda (-1)^{k_{fi} + \lambda}\,a_{fi,k_{fi}}^\lambda
        \left[(-\mathrm{i})^{k_{fi} - X_{fi}}\,\vec{\varepsilon}
        \cdot\vec{Y}_{k_{fi}\lambda}^{(X_{fi})}(\hat{k}(\theta, \phi))\right]\right\rbrace\\[1ex]
        &\quad\times\left\lbrace\sum\limits_\lambda (-1)^{k_{fj} + \lambda}\,a_{jf,k_{fj}}^\lambda
        \left[(-\mathrm{i})^{k_{fj} - X_{fj}}\,\vec{\varepsilon}
        \cdot\vec{Y}_{k_{fj}\lambda}^{(X_{fj})}(\hat{k}(\theta, \phi))\right]^\ast\right\rbrace\\[3ex]
        a_{fi,k_{fi}}^\lambda &= (-1)^{F_f + I + k_{fi} + J_i}\sqrt{(2F_f + 1)(2J_i + 1)}
        \langle F_fm_fk_{fi}\lambda|F_im_i\rangle\begin{Bmatrix}J_i & J_f & k_{fi} \\F_f & F_i & I\end{Bmatrix}\\[3ex]
        X_{\!fi} &= \begin{cases}+1, & \text{if electric } (\mathrm{E}k_{fi}) \\
        \ \,0, & \text{if magnetic } (\mathrm{M}k_{fi})\end{cases},
        \end{aligned}$$

        where $A_{if}^{Xk_{fi}}$ is the Einstein coefficient for the electric (magnetic)
        decay $|i\rangle\rightarrow|f\rangle$ and the rank-$k_{fi}$ multipole order,
        $\vec{Y}_{k_{fi}\lambda}^{(X_{fi})}(\hat{k})$ is the vector spherical harmonic
        (see p. 215, Eq. (35) in [<a href=https://doi.org/10.1142/0270>1</a>]), $\rho$ is the density matrix,
        $\hat{k}$ is the direction of emission and $\vec{\varepsilon}$
        is the complex polarization vector of the emitted photons.
        The calculation includes interference terms between all multipole ranks `1 <= k <= Atom.decay_map.k_max`
        if emission directions are chosen through the (`theta`, `phi`) or `k_vec` parameters.
        Parity mixing is currently not considered, such that all rank-$k$ electric (magnetic) transition are pure.
        The emitted polarization can be chosen through the `x_vec` parameter. If `x_vec` is `None`, the above equation
        will be summed over two orthogonal polarization vectors.
        The emitted multipole orders can be limited through the `k` parameter.
        The transitions contributing to the scattering rate can be limited through the index lists `i` and `f`
        of the initial and final states, before and after spontaneous decay, respectively.</br></br>

        If no emission direction is chosen (`theta = phi = k_vec = None`), the above equation simplifies
        to the scattering rate into the complete $4\pi$ solid angle (without polarization selection)

        $$\begin{aligned}
        \Gamma_\mathrm{sc}(\rho) &= \sum\limits_{f\in\mathcal{F}}
        \sum\limits_{i\in\mathcal{I}}\sum\limits_{(Xk)_{fi}}
        \rho_{ii}A_{if}^{Xk_{fi}}\left(a_{fi,k_{fi}}^{m_i - m_f}\right)^2.
        \end{aligned}$$

        :param rho: The density matrix $\rho$ of the `Atom`. Must have the same size as the `Atom`
         along the specified `axis`, and `axis + 1` if `as_density_matrix == True`.
        :param as_density_matrix: Whether 'rho' is a state vector or a density matrix.
        :param k: The rank(s) $k$ of the emitted multipole radiation. If `None`,
         all orders `1 <= k <= Atom.decay_map.k_max` are considered.
        :param theta: The elevation angle of detection relative to the $z$-axis.
        :param phi: The azimuthal angle of detection in the $xy$-plane.
        :param k_vec: An iterable of directional vectors $\hat{k}$ emitted by the atom.
         `k_vec` must have shape `(3, )` or `(m, 3)` .
        :param x_vec: An iterable of complex polarization vectors $\vec{\varepsilon}$ emitted by the atom.
         `x_vec` must have shape `(3, )` or `(m, 3)` or be a `str` indicating a special polarization:
          <ul>
            <li>$e_\theta$: `{'z', 'theta', 't'}`</li>
            <li>$e_\phi$: `{'x', 'y', 'xy', 'phi', 'p'}`</li>
            <li>$\sigma^-$: `{'-', 's-', 'sigma-', 'l'}`</li>
            <li>$\sigma^+$: `{'+', 's+', 'sigma+', 'r'}`</li>
          </ul>
          In these cases, the polarizations are created automatically based on the emission directions.
        :param i: The initially excited state indexes to consider for spontaneous decay.
         If `None`, all states are considered.
        :param f: The final decayed state indexes to consider for spontaneous decay.
         If `None`, all states are considered.
        :param axis: The axis along which the population is aligned in `rho`. The default is `axis = 1`,
         expecting `rho` as an array with shape `(n, Atom.size, Atom.size, ... )`.
        :returns: (Gamma_sc) The scattering rate $\Gamma_\mathrm{sc}$ as an array with shape `(m, n, ...)`.
        """
        return self.atom.scattering_rate(rho, as_density_matrix=as_density_matrix, k=k, theta=theta, phi=phi,
                                         k_vec=k_vec, x_vec=x_vec, i=i, f=f, axis=axis)


def density_matrix_diagonal(rho: array_like, axis: int = 1) -> ndarray:
    r"""
    The diagonal of the density matrix $\operatorname{diag}(\rho)$.

    :param rho: The density matrix $\rho$. Must have the same size in `axis` and `axis + 1`.
    :param axis: The axis along which the population is aligned in `rho`. The default is `axis = 1`,
     expecting `rho` as an array with shape `(n, Atom.size, Atom.size, ... )`.
    :returns: the diagonal of the density matrix as an array with shape `(n, Atom.size, ...)`.
    """
    return np.transpose(np.diagonal(rho, axis1=axis, axis2=axis + 1).real, axes=[0, 2, 1])


def _define_colors(n: int, label_map: dict, colormap: str = None):
    """
    :param n: The size of the system.
    :param label_map: A dictionary with state labels as keys
     and an array of the indices of the states with the labels as values.
    :param colormap: A matplotlib colormap.
    :returns: A list of colors with size n.
    """
    cmap = cm.get_cmap(colormap)
    labels = [(k, v) for k, v in label_map.items()]
    labels = sorted(labels, key=lambda kv: min(kv[1]))
    colors = ['', ] * n
    for i, (label, indices) in enumerate(labels):
        for index in indices:
            if colormap is None:
                colors[index] = tools.COLORS.PYPLOT[i % 10]
            else:
                colors[index] = cmap(i / (len(labels) - 1))
    return colors


def construct_electronic_state(freq_0: quant_like, s: quant_like, l: quant_like, j: quant_like, i: quant_like = 0,
                               hyper_const: Iterable[array_like] = None, g: array_like = 0, label: str = None) \
        -> list[State]:
    r"""
    DEPRECATED: For LS coupling, use `gen_electronic_ls_state` instead.
    For general electronic states use `gen_electronic_state` instead.

    Creates all substates of a fine-structure state $|\mathrm{[label]}\pi SLJI\rangle$ using a common label.

    :param freq_0: The energetic position of the state without the hyperfine structure or the magnetic field (MHz).
    :param s: The electron spin quantum number $S$.
    :param l: The electronic angular momentum quantum number $L$.
    :param j: The electronic total angular momentum quantum number $J$.
    :param i: The nuclear spin quantum number $I$.
    :param hyper_const: A list of the hyperfine-structure constants.
     Currently, constants up to the electric quadrupole order are supported ($A$, $B$).
     If `hyper_const` is a scalar, it is assumed to be the constant $A$ and the other orders are 0 (MHz).
    :param g: The nuclear g-factor.
    :param label: The label of the state. The label is used to link states via a `DecayMap`.
    :returns: (states) A list of the created states $|\mathrm{[label]}\pi JIFm\rangle$.
    """
    tools.printw('DEPRECATED: For LS coupling, use `gen_electronic_ls_state` instead.'
                 ' For general electronic states use `gen_electronic_state` instead.')
    return gen_electronic_ls_state(freq_0, s, l, j, i=i, hyper_const=hyper_const, gi=g, label=label)


def gen_electronic_ls_state(freq_j: quant_like, s: quant_like, l: quant_like, j: quant_like, i: quant_like = 0,
                               hyper_const: Iterable[array_like] = None, gi: array_like = 0, label: str = None) \
        -> list[State]:
    r"""
    Creates all substates of a fine-structure state $|\mathrm{[label]}\pi SLJI\rangle$ using a common label.

    :param freq_j: The energetic position of the state without the hyperfine structure or the magnetic field (MHz).
    :param s: The electron spin quantum number $S$.
    :param l: The electronic angular momentum quantum number $L$.
    :param j: The electronic total angular momentum quantum number $J$.
    :param i: The nuclear spin quantum number $I$.
    :param hyper_const: A list of the hyperfine-structure constants.
     Currently, constants up to the electric quadrupole order are supported ($A$, $B$).
     If `hyper_const` is a scalar, it is assumed to be the constant $A$ and the other orders are 0 (MHz).
    :param gi: The nuclear g-factor $g_I$.
    :param label: The label of the state. The label is used to link states via a `DecayMap`.
    :returns: (states) A list of the created states $|\mathrm{[label]}\pi JIFm\rangle$.
    """
    f = get_f(i, j)
    m = [get_m(_f) for _f in f]
    fm = [(_f, _m) for _f, m_f in zip(f, m) for _m in m_f]
    gj = g_j(j, (l, s), None, None)
    parity = bool(l % 2)
    return [State(freq_j, parity, j, i, _f, _m,
                  ls=(l, s), hyper_const=hyper_const, gj=gj, gi=gi, label=label) for (_f, _m) in fm]


def construct_hyperfine_state(freq_0: quant_like, s: quant_like, l: quant_like, j: quant_like, i: quant_like,
                              f: quant_like, hyper_const: Iterable[scalar] = None, g: scalar = 0, label: str = None) \
        -> list[State]:
    r"""
    DEPRECATED: For LS coupling, use `gen_hyperfine_ls_state` instead.
    For general hyperfine states use `gen_hyperfine_state` instead.

    Creates all magnetic substates of a hyperfine-structure state $|\mathrm{[label]}\pi SLJIF\rangle$
    using a common `label`.

    :param freq_0: The energetic position of the state without the hyperfine structure or the magnetic field (MHz).
    :param s: The electron spin quantum number $S$.
    :param l: The electronic angular momentum quantum number $L$.
    :param j: The electronic total angular momentum quantum number $J$.
    :param i: The nuclear spin quantum number $I$.
    :param f: The total angular momentum quantum number $F$.
    :param hyper_const: A list of the hyperfine-structure constants.
     Currently, constants up to the electric quadrupole order are supported ($A$, $B$).
     If `hyper_const` is a scalar, it is assumed to be the constant $A$ and the other orders are 0 (MHz).
    :param g: The nuclear g-factor.
    :param label: The label of the state. The label is used to link states via a `DecayMap`.
    :returns: (states) A list of the created states $|\mathrm{[label]}\pi JIFm\rangle$.
    """
    tools.printw('DEPRECATED: For LS coupling, use `gen_electronic_ls_state` instead.'
                 ' For general electronic states use `gen_electronic_state` instead.')
    return gen_hyperfine_ls_state(freq_0, s, l, j, i, f, hyper_const=hyper_const, gi=g, label=label)


def gen_hyperfine_ls_state(freq_j: quant_like, s: quant_like, l: quant_like, j: quant_like, i: quant_like,
                           f: quant_like, hyper_const: Iterable[scalar] = None, gi: scalar = 0, label: str = None) \
        -> list[State]:
    r"""
    Creates all magnetic substates of a hyperfine-structure state $|\mathrm{[label]}\pi SLJIF\rangle$
    using a common `label`.

    :param freq_j: The energetic position of the state without the hyperfine structure or the magnetic field (MHz).
    :param s: The electron spin quantum number $S$.
    :param l: The electronic angular momentum quantum number $L$.
    :param j: The electronic total angular momentum quantum number $J$.
    :param i: The nuclear spin quantum number $I$.
    :param f: The total angular momentum quantum number $F$.
    :param hyper_const: A list of the hyperfine-structure constants.
     Currently, constants up to the electric quadrupole order are supported ($A$, $B$).
     If `hyper_const` is a scalar, it is assumed to be the constant $A$ and the other orders are 0 (MHz).
    :param gi: The nuclear g-factor $g_I$.
    :param label: The label of the state. The label is used to link states via a `DecayMap`.
    :returns: (states) A list of the created states $|\mathrm{[label]}\pi JIFm\rangle$.
    """
    gj = g_j(j, (l, s), None, None)
    parity = bool(l % 2)
    return [State(freq_j, parity, j, i, f, _m,
                  ls=(l, s), hyper_const=hyper_const, gj=gj, gi=gi, label=label) for _m in get_m(f)]


def gen_electronic_state(
        freq_j: quant_like = 0., parity: Union[bool, str] = None, j: quant_like = 0, i: quant_like = 0,
        ls: quant_iter = None, jj: quant_iter = None,
        hyper_const: Iterable[array_like] = None, gj: array_like = None, gi: array_like = 0,
        label: str = None) -> list[State]:
    r"""
    Creates all substates of a fine-structure state $|\mathrm{[label]}\pi JI\rangle$ using a common label.

    :param freq_j: The energetic position of the state without the hyperfine structure or the magnetic field (MHz).
    :param parity: The parity $\pi$ of the state is used to check the selection rules.
     If None, it is inferred from `ls` if possible.
     It can be either `'even'` (`'e'`, `False`) or `'odd'` (`'o'`, `True`).
    :param j: The electronic total angular momentum quantum number $J$.
    :param i: The nuclear spin quantum number $I$.
    :param ls: A list or a single pair of electronic angular momentum and spin quantum numbers $(l_i, s_i)$
     used to check the selection rules and to calculate the electronic g-factor in the LS-coupling scheme.
     If this is a list of LS-pairs, the parameter `jj` requires a list of $j_i$ quantum numbers.
    :param jj: A list of two electronic total angular momentum quantum numbers $(j_0, j_1)$
     used to calculate the electronic g-factor in the jj-coupling scheme.
     Either a list of two $(l_i, s_i)$ pairs needs to be specified for the parameter `ls`
     or a list of g-factors $g_{j_i}$ for the parameter `gj`.
    :param hyper_const: A list of the hyperfine-structure constants.
     Currently, constants up to the electric quadrupole order are supported ($A$, $B$). If 'hyper_const' is a scalar,
     it is assumed to be the constant $A$ and the other orders are 0 (MHz).
    :param gj: A list of two $g_{j_i}$ or a single electronic g-factor $g_J$. If `gj` is a list, `jj` is required
     and `ls` is overwritten. If `gj` is a scalar, both `ls` and `jj` are overwritten.
    :param gi: The nuclear g-factor $g_I$.
    :param label: The label of the states. The labels are used to link states via a `DecayMap`.
    :returns: (states) A list of the created states $|\mathrm{[label]}\pi JIFm\rangle$.
    """
    if parity is None and not hasattr(ls[0], '__getitem__'):
        parity = bool(ls[0] % 2)
    elif parity is None:
        raise ValueError('Could not infer the state \'parity\' from \'ls\'.'
                         'Please use only one (L, S) pair or specify the parity.')

    f = get_f(i, j)
    m = [get_m(_f) for _f in f]
    fm = [(_f, _m) for _f, m_f in zip(f, m) for _m in m_f]
    gj = g_j(j, ls, jj, gj)
    return [State(freq_j, parity, j, i, _f, _m, ls=ls, jj=jj,
                  hyper_const=hyper_const, gj=gj, gi=gi, label=label) for (_f, _m) in fm]


def gen_hyperfine_state(
        freq_j: quant_like = 0., parity: Union[bool, str] = None,
        j: quant_like = 0, i: quant_like = 0, f: quant_like = 0,
        ls: quant_iter = None, jj: quant_iter = None, hyper_const: Iterable[array_like] = None,
        gj: array_like = None, gi: array_like = 0, label: str = None) -> list[State]:
    r"""
    Creates all magnetic substates of a hyperfine-structure state $|\mathrm{[label]}\pi JIF\rangle$
    using a common `label`.

    :param freq_j: The energetic position of the state without the hyperfine structure or the magnetic field (MHz).
    :param parity: The parity $\pi$ of the state is used to check the selection rules.
     If None, it is inferred from `ls` if possible.
     It can be either `'even'` (`'e'`, `False`) or `'odd'` (`'o'`, `True`).
    :param j: The electronic total angular momentum quantum number $J$.
    :param i: The nuclear spin quantum number $I$.
    :param f: The total angular momentum quantum number $F$.
    :param ls: A list or a single pair of electronic angular momentum and spin quantum numbers $(l_i, s_i)$
     used to check the selection rules and to calculate the electronic g-factor in the LS-coupling scheme.
     If this is a list of LS-pairs, the parameter `jj` requires a list of $j_i$ quantum numbers.
    :param jj: A list of two electronic total angular momentum quantum numbers $(j_0, j_1)$
     used to calculate the electronic g-factor in the jj-coupling scheme.
     Either a list of two $(l_i, s_i)$ pairs needs to be specified for the parameter `ls`
     or a list of g-factors $g_{j_i}$ for the parameter `gj`.
    :param hyper_const: A list of the hyperfine-structure constants.
     Currently, constants up to the electric quadrupole order are supported ($A$, $B$). If 'hyper_const' is a scalar,
     it is assumed to be the constant $A$ and the other orders are 0 (MHz).
    :param gj: A list of two $g_{j_i}$ or a single electronic g-factor $g_J$. If `gj` is a list, `jj` is required
     and `ls` is overwritten. If `gj` is a scalar, both `ls` and `jj` are overwritten.
    :param gi: The nuclear g-factor $g_I$.
    :param label: The label of the states. The labels are used to link states via a `DecayMap`.
    :returns: (states) A list of the created states $|\mathrm{[label]}\pi JIFm\rangle$.
    """
    if parity is None and not hasattr(ls[0], '__getitem__'):
        parity = bool(ls[0] % 2)
    else:
        raise ValueError('Could not infer the state \'parity\' from \'ls\'.'
                         'Please use only one (L, S) pair or specify the parity.')

    gj = g_j(j, ls, jj, gj)
    return [State(freq_j, parity, j, i, f, _m, ls=ls, jj=jj,
                  hyper_const=hyper_const, gj=gj, gi=gi, label=label) for _m in get_m(f)]
