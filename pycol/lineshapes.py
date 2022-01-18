# -*- coding: utf-8 -*-
"""
PyCLS.lineshapes

Created on 29.04.2020

@author: Patrick Mueller

Module for spectral lineshape functions.
"""

import inspect
# from sqlite3 import OperationalError
import numpy as np
import scipy.stats as st
import scipy.special as sp
import scipy.integrate as si

from .types import *
from . import tools
from . import physics as ph
from . import algebra as al
from . import stats as stt
# from .databases import load_dipole_coef

SIGMA_TO_FWHM = np.sqrt(8. * np.log(2.))


def sigma_to_fwhm(sigma: array_like):
    """
    :param sigma: The standard deviation of a Gauss distribution.
    :returns: The full width at half maximum of a Gauss distribution with standard deviation 'sigma'.
    """
    return sigma * SIGMA_TO_FWHM


def fwhm_to_sigma(fwhm: array_like):
    """
    :param fwhm: The full width at half maximum of a Gauss distribution.
    :returns: The standard deviation of a Gauss distribution with full width at half maximum 'fwhm'.
    """
    return fwhm * SIGMA_TO_FWHM


def voigt_fwhm(gamma, sigma):
    """
    :param gamma: The full width at half maximum of the lorentzian part of the Voigt profile.
    :param sigma: The standard deviation of the gaussian part of the Voigt profile.
    :returns: The total full width at half maximum of a Voigt profile.
    """
    return 0.5346 * gamma + np.sqrt(0.2166 * gamma ** 2 + sigma_to_fwhm(sigma) ** 2)


def lorentz(x, gamma):
    """
    :param x: The x quantile.
    :param gamma: The full width at half maximum of the Lorentz profile.
    :returns: The value of a Voigt profile at the value 'x'.
    """
    return st.cauchy.pdf(x, scale=0.5 * gamma)


def lorentz_qi(x, x_0, x_1, gamma_0, gamma_1):
    """
    :param x: The x quantile.
    :param x_0: The x-axis (energy, frequency, ...) position of the first transition.
    :param x_1: The x-axis (energy, frequency, ...) position of the second transition.
    :param gamma_0: The full width at half maximum of the first level.
    :param gamma_1: The full width at half maximum of the second level.
    :returns: The value of the interference terms of a Lorentz profile at the value 'x'.
    """
    cross = np.sqrt(gamma_0 * gamma_1) / (2. * np.pi) / ((x - x_0 + 1j * gamma_0 / 2.) * (x - x_1 - 1j * gamma_1 / 2.))
    return cross.real


def gauss(x, sigma):
    """
    :param x: The x quantile.
    :param sigma: The standard deviation of the Gauss distribution.
    :returns: The value of the normalized gauss distribution with standard deviation 'sigma' at the value 'x'.
    """
    return st.norm.pdf(x, scale=sigma)


def voigt(x, gamma, sigma):
    """
    :param x: The x quantile.
    :param gamma: The full width at half maximum of the Lorentzian part.
    :param sigma: The standard deviation of the underlying Gauss distribution.
    :returns: The value of a Voigt profile at the value 'x'.
    """
    z = (x + 1j * gamma / 2.) / (np.sqrt(2.) * sigma)
    return np.real(sp.wofz(z)) / (sigma * np.sqrt(2. * np.pi))


def pseudo_voigt(x, gamma, a):
    """
    :param x: The x quantile.
    :param gamma: The full width at half maximum of the Lorentzian part.
    :param a: The linear coefficient between Lorentz and Gauss profile,
     where a=1 is a pure Lorentzian and a=0 is a pure Gaussian.
    :returns: The value of a Pseudo-Voigt profile at the value 'x'.
    """
    if not abs(a - 0.5) <= 0.5:
        raise ValueError('Parameter a must be in the interval [0, 1] but is {}'.format(a))
    sigma = gamma / SIGMA_TO_FWHM
    return a * lorentz(x, gamma) + (1. - a) * gauss(x, sigma)


def lorentz_convolved(dist: Callable, k_max: int = 15):
    """
    :param dist: A one-dimensional probability density function. Its integral over all real numbers must equate to one.
    :param k_max: Defines the maximum number of samples to calculate the convolution integral.
     The number of samples n is n <= 2 ** k_max + 1. The number of samples which are used for the calculation
     are determined dynamically based on the width of 'dist' and the width of the Lorentz profile 'gamma'.
    :returns: The lineshape resulting from the convolution of the Lorentz profile
     with a user-specified probability density function 'dist'. The parameters of 'dist' are appended
     to the 'gamma' parameter of the Lorentz profile.
    """

    def _spectrum(x, *args, **kwargs):
        ab = stt.relevant_interval(dist, *args[1:], **kwargs, show=True)
        a_i = x - ab[1]
        b_i = x - ab[0]
        k = np.min([k_max, tools.floor_log2(np.max(20 * (b_i - a_i) / args[0])) + 1])
        if k < 8:
            k = 8
        _x = np.expand_dims(np.asarray(x), axis=0)
        _x_i = np.linspace(a_i, b_i, 2 ** k + 1)
        y = lorentz(_x_i, args[0]) * dist(_x - _x_i, *args[1:], **kwargs)
        return si.romb(y, dx=(b_i - a_i) / (2 ** k), axis=0, show=False)

    spectrum = None
    arg_spec = inspect.getfullargspec(dist)
    p = [p_k + '_' if p_k == 'gamma' else p_k for p_k in arg_spec.args[1:]]
    defaults = arg_spec.defaults
    args_str = '' if len(p) == 0 else ', ' + ', '.join(p)
    def_kwargs_str = ''
    exec_kwargs_str = ''
    if defaults is not None:
        args_str = '' if len(p) - len(defaults) == 0 else ', ' + ', '.join(p[:(len(p) - len(defaults))])
        exec_kwargs_str = ', **{' + ', '.join(['\'{0}\': {0}'.format(p_k)
                                               for p_k in p[(len(p) - len(defaults)):]]) + '}'
        for p_k, d in zip(p[(len(p) - len(defaults)):], defaults):
            def_kwargs_str += ', {0}={1}'.format(p_k, d)
    shape_str = 'def spectrum(x, gamma{0}{1}):\n' \
                '    return _spectrum(x, gamma{0}{2})'.format(args_str, def_kwargs_str, exec_kwargs_str)
    name_space = {'_spectrum': _spectrum, 'spectrum': spectrum}
    exec(shape_str, name_space)
    print('Convolved Lorentz spectrum definition: {}'.format(shape_str[4:shape_str.find(':')]))
    return name_space['spectrum']


def define_shape(shape: Union[Callable, str] = 'lorentz', shape_args: Iterable[str] = None):
    """
    :param shape: A Callable or a str representing a Callable which is used as the lineshape for each individual peak.
    :param shape_args: The parameter names of 'shape'.
     If it is None, the parameters are tried to be inherited from shape.
    :returns: The specified shape and parameter names.
    """
    if isinstance(shape, str):
        if shape.lower() == 'lorentz':
            _shape = lorentz
        elif shape.lower() == 'gauss':
            _shape = gauss
        elif shape.lower() == 'voigt':
            _shape = voigt
        else:
            raise ValueError('Specified shape is not supported. Use one of {} or specify a Callable.')
    else:
        if not isinstance(shape, Callable):
            raise TypeError('Specified shape is neither a Callable nor a str.')
        _shape = shape
    if isinstance(_shape, Shape):
        p = _shape.arg_list
    else:
        if shape_args is None:
            arg_spec = inspect.getfullargspec(_shape)
            p = arg_spec.args[1:]
        else:
            p = shape_args
    return _shape, p


def _fix_list(m: list, size: int):
    """
    :param m: A list of int values
    :param size: The size of the returned list.
    :returns: The list 'm' cut at the specified 'size' or increased with zeros to match the 'size'.
    """
    n = size - len(m)
    if n < 0:
        m = m[:n]
    elif n > 0:
        m += [0, ] * n
    return m


class Shape:
    """
    Class representing a lineshape without any model-specific parameters such as offsets or scales.
    Instances of this class are Callables. A list of their parameters can be viewed by printing '<Shape>.arg_list'.
    The initialization parameters can only be set at creation time of the shape.
    """
    def __init__(self, func: Union[Callable, str] = 'lorentz', func_args: Iterable[str] = None):
        self.func, self.func_args = define_shape(func, func_args)
        self.definition = {}
        self.arg_list = []
        self.arg_dict = {}
        self.n = 0
        Shape.define(self)

    def __call__(self, x, *args, **kwargs):
        return self.func(x, *args)

    def define(self, **kwargs):
        self.definition = dict(**kwargs)
        self.func_args = ['s_{}'.format(arg) if arg[:2] != 's_' else arg for arg in self.func_args]
        self.arg_list = self.func_args.copy()
        self.arg_dict = {arg: self.arg_list.index(arg) for arg in self.arg_list}
        self.n = len(self.arg_list)


class HyperfineShape(Shape):
    """
    Class representing a hyperfine structure lineshape without any model-specific parameters such as offsets or scales.
    Instances of this class are Callables. A list of their parameters can be viewed by printing
    '<HyperfineShape>.arg_list'. The initialization parameters can only be set at creation time of the shape.
    """
    def __init__(self, func: Union[Callable, str] = 'lorentz', func_args: Iterable[str] = None,
                 i: scalar = 0, j_l: scalar = 0, j_u: scalar = 1):
        """
        :param func: The shapes underlying function.
        :param func_args: The argument names of 'func'. If None, the names are tried to be
         inherited the from the signature of the function.
        :param i: The nuclear spin quantum number I. Default is 0.
        :param j_l: The electronic total angular momentum quantum number J of the lower state. Default is 0.
        :param j_u: The electronic total angular momentum quantum number J of the upper state. Default is 1.
        """
        super().__init__(func, func_args)

        tools.check_half_integer(i, j_l, j_u)
        self.i, self.j_l, self.j_u = i, j_l, j_u
        self.fl_list = [f1 + abs(i - j_l) for f1 in range(int(i + j_l - abs(i - j_l) + 1))]
        self.fu_list = [f2 + abs(i - j_u) for f2 in range(int(i + j_u - abs(i - j_u) + 1))]
        self.f = [[fl, fu] for fl in self.fl_list for fu in self.fu_list if abs(fu - fl) <= 1]
        self.f_qi = [[fl, fu1, fu2] for fl in self.fl_list for fu1 in self.fu_list for fu2 in self.fu_list
                     if abs(fu1 - fl) <= 1 and abs(fu2 - fl) <= 1 and fu1 < fu2]
        self.f_str = [[tools.half_integer_to_str(f_i[0]), tools.half_integer_to_str(f_i[1])] for f_i in self.f]
        self.f_qi_str = [[tools.half_integer_to_str(f_i[0]), tools.half_integer_to_str(f_i[1]),
                          tools.half_integer_to_str(f_i[2])] for f_i in self.f_qi]

        self.hyper_scale_a = []
        self.hyper_scale_b = []
        self.hyper_scale_a_qi = []
        self.hyper_scale_b_qi = []
        self._generate_hyper_scales()

        self.a, self.b, self.c = None, None, None
        self._load_dipoles()

        self.definition = {}
        self.arg_list = []
        self.arg_dict = {}
        self.n = 0
        self.define()
    
    def _load_dipoles(self):
        """
        Loads the dipole coefficients from the database or calculates them.
        It is recommended to fill the database first.
        :returns: None.
        """
        # try:
        #     a = [load_dipole_coef('A', self.i, self.j_l, self.j_u, f_i[0], f_i[1], f_i[1]) for f_i in self.f]
        #     b = [load_dipole_coef('B', self.i, self.j_l, self.j_u, f_i[0], f_i[1], f_i[1]) for f_i in self.f]
        #     c = [load_dipole_coef('C', self.i, self.j_l, self.j_u, f_i[0], f_i[1], f_i[2]) for f_i in self.f_qi]
        # except (TypeError, OperationalError):
        a = [float(al.a(self.i, self.j_l, f_i[0], self.j_u, f_i[1], as_sympy=False)) for f_i in self.f]
        b = [float(al.b(self.i, self.j_l, f_i[0], self.j_u, f_i[1], as_sympy=False)) for f_i in self.f]
        c = [float(al.c(self.i, self.j_l, f_i[0], self.j_u, f_i[1], f_i[2], as_sympy=False)) for f_i in self.f_qi]
        self.a, self.b, self.c = a, b, c

    def _generate_hyper_scales(self):
        """
        Precomputes the unscaled hyperfine structure splittings for faster computation at runtime.
        :returns: None.
        """
        self.hyper_scale_a = [[ph.hyperfine(self.i, self.j_l, f_i[0], 1),
                               ph.hyperfine(self.i, self.j_u, f_i[1], 1)] for f_i in self.f]
        self.hyper_scale_b = [[ph.hyperfine(self.i, self.j_l, f_i[0], 0, b=1),
                               ph.hyperfine(self.i, self.j_u, f_i[1], 0, b=1)] for f_i in self.f]
        self.hyper_scale_a_qi = [[ph.hyperfine(self.i, self.j_l, f_i[0], 1),
                                  ph.hyperfine(self.i, self.j_u, f_i[1], 1),
                                  ph.hyperfine(self.i, self.j_u, f_i[2], 1)] for f_i in self.f_qi]
        self.hyper_scale_b_qi = [[ph.hyperfine(self.i, self.j_l, f_i[0], 0, b=1),
                                  ph.hyperfine(self.i, self.j_u, f_i[1], 0, b=1),
                                  ph.hyperfine(self.i, self.j_u, f_i[2], 0, b=1)] for f_i in self.f_qi]

    def define(self, f_f: Iterable[scalar] = None, qi: bool = False, fixed_ratios: bool = True, **kwargs):
        """
        Define whether to use all peaks, quantum interference and fixed intensity ratios.

        :param f_f: Hyperfine transitions which are not used. TODO: Not implemented yet.
        :param qi: Whether to use quantum interference. This adds a single parameter if fixed_ratios is True
         and cross-intensities if fixed_ratios is False. Default is False.
        :param fixed_ratios: Whether the relative intensity ratios should be fixed to the Racah coefficients.
        :param kwargs: Additional definitions. Do not have any effect currently.
        :returns: None. Defines the parameter space of the shape.
        """
        self.definition = dict(f_f=f_f, qi=qi, fixed_ratios=fixed_ratios, **kwargs)

        if qi:
            print('Quantum interference is enabled -> The Hyperfine-class uses a Lorentzian func.\n'
                  'Use the Convolved-class to use other funcs with quantum interference.')
            self.func, self.func_args = define_shape('lorentz')
        self.func_args = ['s_{}'.format(arg) if arg[:2] != 's_' else arg for arg in self.func_args]
        self.arg_list = self.func_args.copy()
        if self.i > 0:
            if self.j_l > 0:
                self.arg_list.append('A_l')
            if self.j_u > 0:
                self.arg_list.append('A_u')
            if self.i > 0.5:
                if self.j_l > 0.5:
                    self.arg_list.append('B_l')
                if self.j_u > 0.5:
                    self.arg_list.append('B_u')
        if fixed_ratios:
            if qi:
                self.arg_list.append('qi')
        else:
            for f_i in self.f_str:
                self.arg_list.append('Int__{}__{}'.format(*f_i))
            if qi:
                for f_i in self.f_qi_str:
                    self.arg_list.append('QInt__{}__{}__{}'.format(*f_i))
        self.arg_dict = {arg: self.arg_list.index(arg) for arg in self.arg_list}
        self.n = len(self.arg_list)

    def _calc_int(self, i, *args):
        """
        :param i: The ith transitions index.
        :param args: The arguments of the shape.
        :returns: The intensity of the ith transition.
        """
        if self.definition['fixed_ratios']:
            return self.a[i] + self.b[i] * args[self.arg_dict['qi']] if self.definition['qi'] else self.a[i]
        return args[self.arg_dict['Int__{}__{}'.format(*self.f_str[i])]]

    def _calc_int_qi(self, i, *args):
        """
        :param i: The ith transitions index.
        :param args: The arguments of the shape.
        :returns: The quantum interference intensity of the ith cross-transition.
        """
        if self.definition['fixed_ratios']:
            return self.c[i] * args[self.arg_dict['qi']]
        return args[self.arg_dict['QInt__{}__{}__{}'.format(*self.f_qi_str[i])]]

    def _calc_x0(self, i: int, *args):
        """
        :param i: The ith transitions index.
        :param args: The arguments of the shape.
        :returns: The x-shift of the ith transition.
        """
        x0 = 0.
        if self.i > 0:
            if self.j_l > 0:
                x0 -= args[self.arg_dict['A_l']] * self.hyper_scale_a[i][0]
            if self.j_u > 0:
                x0 += args[self.arg_dict['A_u']] * self.hyper_scale_a[i][1]
            if self.i > 0.5:
                if self.j_l > 0.5:
                    x0 -= args[self.arg_dict['B_l']] * self.hyper_scale_b[i][0]
                if self.j_u > 0.5:
                    x0 += args[self.arg_dict['B_u']] * self.hyper_scale_b[i][1]
        return x0

    def _calc_x0_qi(self, i: int, *args):
        """
        :param i: The ith transitions index.
        :param args: The arguments of the shape.
        :returns: The x-shifts of the ith cross-transition.
        """
        x0_0 = 0.
        x0_1 = 0.
        if self.i > 0:
            if self.j_l > 0:
                x0_0 -= args[self.arg_dict['A_l']] * self.hyper_scale_a_qi[i][0]
                x0_1 -= args[self.arg_dict['A_l']] * self.hyper_scale_a_qi[i][0]
            if self.j_u > 0:
                x0_0 += args[self.arg_dict['A_u']] * self.hyper_scale_a_qi[i][1]
                x0_1 += args[self.arg_dict['A_u']] * self.hyper_scale_a_qi[i][2]
            if self.i > 0.5:
                if self.j_l > 0.5:
                    x0_0 -= args[self.arg_dict['B_l']] * self.hyper_scale_b_qi[i][0]
                    x0_1 -= args[self.arg_dict['B_l']] * self.hyper_scale_b_qi[i][0]
                if self.j_u > 0.5:
                    x0_0 += args[self.arg_dict['B_u']] * self.hyper_scale_b_qi[i][1]
                    x0_1 += args[self.arg_dict['B_u']] * self.hyper_scale_b_qi[i][2]
        return x0_0, x0_1

    def __call__(self, x, *args, **kwargs):
        """
        :param x: The x quantiles.
        :param args: The parameters.
        :param kwargs: Additional keyword arguments. Currently not used.
        :returns: The shape at the specified x values.
        """
        x = np.asarray(x)
        func_args = [args[self.arg_dict[arg]] for arg in self.func_args]
        y = np.sum([self._calc_int(i, *args) * self.func(x - self._calc_x0(i, *args), *func_args)
                    for i in range(len(self.f))], axis=0)
        if self.definition['qi']:
            y += np.sum([2 * self._calc_int_qi(i, *args) * lorentz_qi(x, *self._calc_x0_qi(i, *args), args[0], args[0])
                         for i in range(len(self.f_qi))], axis=0)
        return y


class Model:
    """
    Class representing a lineshape model. Instances of this class are Callables.
    A list of their parameters can be viewed by printing '<Model>.arg_list'.
    """
    def __init__(self, shape: Union[Shape, Callable, str] = 'lorentz', offset_order: int = 0, n_shape: int = 1,
                 x_cuts: Iterable = None, arg_map: dict = None, **kwargs):
        """
        :param shape: A Shape, Callable or a str representing a Callable which is used as the lineshape for each line.
        :param offset_order: The polynomial order of the offset. For example, an order of 2 yields the model definition
         model(..., y0, y1, y2) which has a parabola y0 + y1 * x + y2 * x ** 2 as an offset.
        :param n_shape: The multiplicity of the shape (i.e. the number of peaks for single-peak shapes).
        :param x_cuts: An Iterable of x-values where tu cut the x-axis to use multiple offset parameters.
         The offset parameters are appended to the arguments of the model,
         such that for example model(..., norm_n, y0_0, y0_1) if the x-axis is cut once and offset_order=0.
         If the model is computed at the exact value of a cut, the offset corresponding to the larger x-values is used.
        :param arg_map: A dictionary of the parameters as keys and lists of int values as values.
         Each list entry represents one of the 'n_shape' shape calls of the specific parameter.
         Note for the offset parameters, there are len(x_cuts) + 1 list entries.
         All shape calls which have the same number for a parameter use a common argument for that specific parameter.
         For example, if 'n_shape' = 3 and 'arg_map' = {'scale': [0, 2, 2]},
         only two arguments for the parameter 'scale', named 'scale_0' and 'scale_2' are created. If 'arg_map' is None,
         individual arguments are created for the two general parameters 'x0' and 'scale' and common arguments for the
         remaining parameters of the specified 'shape'. For example, if 'shape' is 'lorentz' and n = 3,
         arg_map = {'x0': [0, 1, 2], 'scale': [0, 1, 2], 'gamma': [0, 0, 0]} and the model is defined as
         model(gamma, x0_0, x0_1, x0_2, scale_0, scale_1, scale_2, y0).
        :param kwargs: The keyword arguments are passed to the 'Shape' definition if shape is not a 'Shape' instance.
        """
        self.shape = shape
        if not isinstance(self.shape, Shape):
            self.shape = Shape(func=shape, **kwargs)

        self.n_shape, self.arg_map, self.x_cuts, self.offset_order = 1, {}, [], 0

        self.arg_list_offset = []
        self.arg_list_signal = []
        self.arg_list = []
        self.arg_dict_offset = {}
        self.arg_dict_signal = {}
        self.arg_dict = {}

        self.n_signal = 0
        self.n_offset = 0
        self.n = 0

        self.definition = {}
        self.define(n_shape=n_shape, arg_map=arg_map, x_cuts=x_cuts, offset_order=offset_order)

    def define(self, offset_order: int = 0, n_shape: int = 1, x_cuts: Iterable = None, arg_map: dict = None, **kwargs):
        """
        :param offset_order: The polynomial order of the offset. For example, an order of 2 yields the model definition
         model(..., y0, y1, y2) which has a parabola y0 + y1 * x + y2 * x ** 2 as an offset.
        :param n_shape: The multiplicity of the shape (i.e. the number of peaks for single-peak shapes).
        :param x_cuts: An Iterable of x-values where tu cut the x-axis to use multiple offset parameters.
         The offset parameters are appended to the arguments of the model,
         such that for example model(..., norm_n, y0_0, y0_1) if the x-axis is cut once and offset_order=0.
         If the model is computed at the exact value of a cut, the offset corresponding to the larger x-values is used.
        :param arg_map: A dictionary of the parameters as keys and lists of int values as values.
         Each list entry represents one of the 'n_shape' shape calls of the specific parameter.
         Note for the offset parameters, there are len(x_cuts) + 1 list entries.
         All shape calls which have the same number for a parameter use a common argument for that specific parameter.
         For example, if 'n_shape' = 3 and 'arg_map' = {'scale': [0, 2, 2]},
         only two arguments for the parameter 'scale', named 'scale_0' and 'scale_2' are created. If 'arg_map' is None,
         individual arguments are created for the two general parameters 'x0' and 'scale' and common arguments for the
         remaining parameters of the specified 'shape'. For example, if 'shape' is 'lorentz' and n = 3,
         arg_map = {'x0': [0, 1, 2], 'scale': [0, 1, 2], 'gamma': [0, 0, 0]} and the model is defined as
         model(gamma, x0_0, x0_1, x0_2, scale_0, scale_1, scale_2, y0).
        :param kwargs: Additional definitions. Do not have any effect currently.
        :returns: None. Defines the parameter space of the model.
        """
        self.n_shape = n_shape
        if arg_map is None:
            arg_map = {}
        self.x_cuts = [] if x_cuts is None else list(x_cuts)
        self.offset_order = offset_order
        self.definition = dict(n_shape=n_shape, arg_map=arg_map, x_cuts=x_cuts, offset_order=offset_order, **kwargs)
        self.definition = tools.merge_dicts(self.definition, self.shape.definition)

        self.arg_list_signal = self.shape.arg_list.copy() + ['x0', 'scale']
        self.arg_list_offset = ['y{}'.format(n_o) for n_o in range(self.offset_order + 1)]
        self.arg_list = self.arg_list_signal + self.arg_list_offset

        _arg_map = {arg: [n for n in range(self.n_shape)] if arg in ['x0', 'scale'] else [0, ] * self.n_shape
                    for arg in self.arg_list_signal}  # Default signal arg_map.
        # Merge default signal arg_map with custom arg_map. Complete entries of custom arg_map.
        self.arg_map = {arg: m if arg not in arg_map.keys() else _fix_list(arg_map[arg], self.n_shape)
                        for arg, m in _arg_map.items()}
        # Define signal parameters with the given arg_map.
        self.arg_list_signal = ['{}__{}'.format(arg, n) if len(set(self.arg_map[arg])) > 1 else arg
                                for arg in self.arg_list_signal for n in sorted(list(set(self.arg_map[arg])))]
        # Repeat for offset parameters.
        _arg_map = {arg: [n for n in range(len(self.x_cuts) + 1)] for arg in self.arg_list_offset}
        self.arg_map = tools.merge_dicts(self.arg_map, {arg: m if arg not in arg_map.keys()
                                         else _fix_list(arg_map[arg], len(self.x_cuts) + 1)
                                                        for arg, m in _arg_map.items()})
        self.arg_list_offset = ['{}__{}'.format(arg, n) if len(set(self.arg_map[arg])) > 1 else arg
                                for arg in self.arg_list_offset for n in sorted(list(set(self.arg_map[arg])))]

        self.arg_list = self.arg_list_signal + self.arg_list_offset
        self.n_signal = len(self.arg_list_signal)
        self.n_offset = len(self.arg_list_offset)
        self.n = self.n_signal + self.n_offset
        self.arg_dict_signal = {arg: self.arg_list_signal.index(arg) for arg in self.arg_list_signal}
        self.arg_dict_offset = {arg: self.arg_list_offset.index(arg) for arg in self.arg_list_offset}
        self.arg_dict = {arg: self.arg_list.index(arg) for arg in self.arg_list}

    def define_shape(self, f_f: Iterable[scalar] = None, qi: bool = False, fixed_ratios: bool = True, **kwargs):
        """
        :param f_f:
        :param qi:
        :param fixed_ratios:
        :param kwargs:
        :returns: None. Redefines the 'shape' of the model.
        """
        self.shape.define(f_f=f_f, qi=qi, fixed_ratios=fixed_ratios, **kwargs)
        self.define(**self.definition)

    def get_default_arg_map(self):
        """
        :returns: The default arg_map.
        """
        signal = {arg: [n for n in range(self.n_shape)] if arg in ['x0', 'scale'] else [0, ] * self.n_shape
                  for arg in self.arg_list_signal}
        offset = {arg: [n for n in range(len(self.x_cuts) + 1)] for arg in self.arg_list_offset}
        return tools.merge_dicts(signal, offset)

    def get_arg_map_keys(self):
        """
        :returns: The keys of arg_map.
        """
        return self.arg_map.keys()

    def get_arg_label(self, arg: str, n: int):
        """
        :param arg: The argument whose name is returned. Must be in 'arg_map'.
        :param n: The number of the specified argument whose name is returned.
        :returns: The name of the nth parameter corresponding to the specified argument.
        """
        return '{}__{}'.format(arg, self.arg_map[arg][n]) if len(set(self.arg_map[arg])) > 1 else arg

    def get_arg_index(self, arg: str, n: int, sub_dict: str = None):
        """
        :param arg: The argument whose index is returned. Must be in 'arg_map'.
        :param n: The number of the specified argument whose index is returned.
        :param sub_dict: The specific argument list to get the index from. Can be None, 'signal' or 'offset'.
        :returns: The index of the nth parameter corresponding to the specified argument.
        """
        label = self.get_arg_label(arg, n)
        if sub_dict is None:
            return self.arg_dict[label]
        elif sub_dict == 'signal':
            return self.arg_dict_signal[label]
        elif sub_dict == 'offset':
            return self.arg_dict_offset[label]

    def offset(self, x, *args):
        """
        :param x: The x quantiles.
        :param args: The offset parameters.
        :returns: The offset at the specified x values.
        """
        if len(args) != self.n_offset:
            raise ValueError('The function \'offset\' expects {} arguments but got {}. The arguments are {}.'
                             .format(self.n_offset, len(args), self.arg_list_offset))
        poly = [np.polynomial.polynomial.Polynomial([args[self.get_arg_index('y{}'.format(n_o), n, 'offset')]
                                                     for n_o in range(self.definition['offset_order'] + 1)])
                for n in range(len(self.x_cuts) + 1)]

        order = np.argsort(x)
        inverse_order = np.array([int(np.where(order == i)[0])
                                  for i in range(order.size)])  # Invert the order for later.
        x_sorted = x[order]
        cut_i = [np.nonzero(x_sorted >= x_cut)[0][0] for x_cut in self.x_cuts]
        x_sorted = np.split(x_sorted, cut_i)
        y_sorted = [p(x_i) for x_i, p in zip(x_sorted, poly)]
        return np.concatenate([y_i for y_i in y_sorted], axis=0)[inverse_order]

    def signal(self, x, *args):
        """
        :param x: The x quantiles.
        :param args: The signal parameters.
        :returns: The signal at the specified x values.
        """
        if len(args) != self.n_signal:
            raise ValueError('The function \'signal\' expects {} arguments but got {}. The arguments are {}.'
                             .format(self.n_signal, len(args), self.arg_list_signal))
        return np.sum([args[self.get_arg_index('scale', n, 'signal')]
                       * self.shape(x - args[self.get_arg_index('x0', n, 'signal')],
                                    *[args[self.get_arg_index(arg, n, 'signal')] for arg in self.shape.arg_list])
                       for n in range(self.n_shape)], axis=0)

    def __call__(self, x, *args, **kwargs):
        """
        :param x: The x quantiles.
        :param args: The parameters.
        :param kwargs: Additional keyword arguments. Currently not used.
        :returns: The model at the specified x values.
        """
        return self.signal(x, *args[:self.n_signal]) + self.offset(x, *args[self.n_signal:])


class SumModel(Model):
    """
    Class representing a sum of lineshape models. Instances of this class are Callables.
    A list of their parameters can be viewed by printing '<SumModel>.arg_list'.
    """
    def __init__(self, *models: Model, offset_order: int = 0, n_shape: int = 1,
                 x_cuts: Iterable = None, arg_map: dict = None, **kwargs):
        """
        :param models: An arbitrary number of models which ought o be summed.
        :param offset_order: The polynomial order of the offset. For example, an order of 2 yields the model definition
         model(..., y0, y1, y2) which has a parabola y0 + y1 * x + y2 * x ** 2 as an offset.
        :param n_shape: The multiplicity of the shape (i.e. the number of peaks for single-peak shapes).
        :param x_cuts: An Iterable of x-values where tu cut the x-axis to use multiple offset parameters.
         The offset parameters are appended to the arguments of the model,
         such that for example model(..., norm_n, y0_0, y0_1) if the x-axis is cut once and offset_order=0.
         If the model is computed at the exact value of a cut, the offset corresponding to the larger x-values is used.
        :param arg_map: A dictionary of the parameters as keys and lists of int values as values.
         Each list entry represents one of the 'n_shape' shape calls of the specific parameter.
         Note for the offset parameters, there are len(x_cuts) + 1 list entries.
         All shape calls which have the same number for a parameter use a common argument for that specific parameter.
         For example, if 'n_shape' = 3 and 'arg_map' = {'scale': [0, 2, 2]},
         only two arguments for the parameter 'scale', named 'scale_0' and 'scale_2' are created. If 'arg_map' is None,
         individual arguments are created for the two general parameters 'x0' and 'scale' and common arguments for the
         remaining parameters of the specified 'shape'. For example, if 'shape' is 'lorentz' and n = 3,
         arg_map = {'x0': [0, 1, 2], 'scale': [0, 1, 2], 'gamma': [0, 0, 0]} and the model is defined as
         model(gamma, x0_0, x0_1, x0_2, scale_0, scale_1, scale_2, y0).
        :param kwargs: The keyword arguments are passed to the 'Shape' definition if shape is not a 'Shape' instance.
        """
        self.models = list(models)
        func_args = ['m{}__{}'.format(i, arg) for i, model in enumerate(self.models) for arg in model.arg_list_signal]

        def func(x, *args):
            return np.sum([model.signal(x, *args[:len(model.arg_list_signal)]) for model in self.models], axis=0)

        super().__init__(Shape(func=func, func_args=func_args), offset_order=offset_order, n_shape=n_shape,
                         x_cuts=x_cuts, arg_map=arg_map, **kwargs)
