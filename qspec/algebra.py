# -*- coding: utf-8 -*-
"""
qspec.algebra
=============

Module including methods for calculating dipole coefficients.
"""

import numpy as np
from scipy.constants import physical_constants
from sympy import nsimplify, sqrt, sin, cos, pi
from sympy.vector import CoordSys3D, Vector
import sympy.physics.wigner as spw

from qspec.qtypes import *

__all__ = ['quant', 'cast_sympy', 'clebsch_gordan', 'wigner_3j', 'wigner_6j', 'a', 'b', 'ab', 'c', 'abc', 'f_0', 'g_0',
           'c_dipole', 'a_dipole', 'a_dipole_cart', 'reduced_f_root', 'reduced_f', 'a_tilda', 'a_m_tilda',
           'mu_j_m1', 'mu_jj_m1']


def cast_sympy(*args: sympy_like, as_sympy: bool = True, dtype: type = float) -> sympy_like:
    r"""
    Cast the arguments `args` to a <a href="https://www.sympy.org/en/index.html">
    `sympy`</a> type (symbol) or the specified `dtype`.

    :param args: The arguments.
    :param as_sympy: Return the result as a symbol (`True`) or as a `float` (`False`).
    :param dtype: The type to use if `as_sympy == False`.
    :returns: (cast_args) The cast arguments.
    """
    ret = tuple(nsimplify(arg) if as_sympy else dtype(arg) for arg in args)
    if len(ret) == 1:
        ret = ret[0]
    return ret


def clebsch_gordan(j1: sympy_quant, j2: sympy_quant, j3: sympy_quant,
                   m1: sympy_quant, m2: sympy_quant, m3: sympy_quant, as_sympy: bool = False) \
        -> Union[sympy_core, float]:
    r"""
    Calculates the Clebsch-Gordan coefficient
    $$C_{J_1 m_1, J_2 m_2}^{J_3 m_3} \coloneqq \langle J_1, m_1, J_2, m_2\, |\, J_3, m_3\rangle.$$

    :param j1: The angular momentum quantum number $J_1$.
    :param j2: The angular momentum quantum number $J_2$.
    :param j3: The angular momentum quantum number $J_3$.
    :param m1: The magnetic quantum number $m_1$.
    :param m2: The magnetic quantum number $m_2$.
    :param m3: The magnetic quantum number $m_3$.
    :param as_sympy: Return the result as a symbol (`True`) or as a `float` (`False`).
    :returns: (cg) The Clebsch-Gordan coefficient $C_{J_1 m_1, J_2 m_2}^{J_3 m_3}$.
    """
    ret = spw.clebsch_gordan(j1, j2, j3, m1, m2, m3)
    return cast_sympy(ret, as_sympy=as_sympy)


def wigner_3j(j1: sympy_quant, j2: sympy_quant, j3: sympy_quant,
              m1: sympy_quant, m2: sympy_quant, m3: sympy_quant, as_sympy: bool = False) -> Union[sympy_core, float]:
    r"""
    Calculate the Wigner-3j symbol

    $$W_{m_1m_2m_3}^{J_1J_2J_3}\coloneqq
    \begin{pmatrix}
    J_1 & J_2 & J_3 \\
    m_1 & m_2 & m_3
    \end{pmatrix}.$$

    :param j1: The angular momentum quantum number $J_1$.
    :param j2: The angular momentum quantum number $J_2$.
    :param j3: The angular momentum quantum number $J_3$.
    :param m1: The magnetic quantum number $m_1$.
    :param m2: The magnetic quantum number $m_2$.
    :param m3: The magnetic quantum number $m_3$.
    :param as_sympy: Return the result as a symbol (`True`) or as a `float` (`False`).
    :returns: (w3) The Wigner-3j symbol $W_{m_1m_2m_3}^{J_1J_2J_3}$.

    """
    ret = spw.wigner_3j(j1, j2, j3, m1, m2, m3)
    return cast_sympy(ret, as_sympy=as_sympy)


def wigner_6j(j1: sympy_quant, j2: sympy_quant, j3: sympy_quant,
              j4: sympy_quant, j5: sympy_quant, j6: sympy_quant, as_sympy: bool = False) -> Union[sympy_core, float]:
    r"""
    Calculate the Wigner-6j symbol

    $$W_{J_4J_5J_6}^{J_1J_2J_3}\coloneqq
    \begin{Bmatrix}
    J_1 & J_2 & J_3 \\
    J_4 & J_5 & J_6
    \end{Bmatrix}.$$

    :param j1: $J_1$
    :param j2: $J_2$
    :param j3: $J_3$
    :param j4: $J_4$
    :param j5: $J_5$
    :param j6: $J_6$
    :param as_sympy: Return the result as a symbol (`True`) or as a `float` (`False`).
    :returns: (w6) The Wigner-6j symbol $W_{J_4J_5J_6}^{J_1J_2J_3}$.
    """
    ret = spw.wigner_6j(j1, j2, j3, j4, j5, j6)
    return cast_sympy(ret, as_sympy=as_sympy)


""" Linear polarization absorption dipole coefficients a, b and c.
    Refer to [Brown et al., Phys. Rev. A 87, 032504 (2013)]. """


def a(i: sympy_quant, j_l: sympy_quant, f_l: sympy_quant, j_u: sympy_quant, f_u: sympy_quant, as_sympy: bool = False) \
        -> Union[sympy_core, float]:
    r"""
    The coefficient $A_F^{F^\prime}$, parameterizing the isotropic part of
    the perturbative differential scattering rate, as described in
    [<a href="https://doi.org/10.1103/PhysRevA.87.032504">R. C. Brown et al., Phys. Rev. A 87, 032504 (2013)</a>].

    :param i: The nuclear spin quantum number $I$.
    :param j_l: The electronic total angular momentum quantum number $J_l$ of the lower state.
    :param f_l: The total angular momentum quantum number $F_l$ of the lower state.
    :param j_u: The electronic total angular momentum quantum number $J_u$ of the upper state.
    :param f_u: The total angular momentum quantum number $F_u$ of the upper state.
    :param as_sympy: Return the result as a symbol (`True`) or as a `float` (`False`).
    :returns: (A) The $A_F^{F^\prime}$ coefficient.
    """
    ret = spw.wigner_6j(j_u, j_l, 1, f_l, f_u, i) ** 2
    i, j_l, f_l, j_u, f_u, ret = cast_sympy(i, j_l, f_l, j_u, f_u, ret, as_sympy=as_sympy)
    return ret * (2 * f_l + 1) * (2 * f_u + 1) * (2 * j_u + 1) / (3 * (2 * j_l + 1) * (2 * i + 1))
    # i, j_l, f_l, j_u, f_u = cast_sympy(as_sympy, i, j_l, f_l, j_u, f_u)
    # return f_0(i, j_l, f_l, j_u, f_u, 0, as_sympy=as_sympy) - b(i, j_l, f_l, j_u, f_u, as_sympy=as_sympy)  # (slower)


def b(i: sympy_quant, j_l: sympy_quant, f_l: sympy_quant, j_u: sympy_quant, f_u: sympy_quant, as_sympy: bool = False) \
        -> Union[sympy_core, float]:
    r"""
    The coefficient $B_F^{F^\prime}$, parameterizing the classical angle-dependent part of
    the perturbative differential scattering rate, as described in
    [<a href="https://doi.org/10.1103/PhysRevA.87.032504">R. C. Brown et al., Phys. Rev. A 87, 032504 (2013)</a>].

    :param i: The nuclear spin quantum number $I$.
    :param j_l: The electronic total angular momentum quantum number $J_l$ of the lower state.
    :param f_l: The total angular momentum quantum number $F_l$ of the lower state.
    :param j_u: The electronic total angular momentum quantum number $J_u$ of the upper state.
    :param f_u: The total angular momentum quantum number $F_u$ of the upper state.
    :param as_sympy: Return the result as a symbol (`True`) or as a `float` (`False`).
    :returns: (B) The $B_F^{F^\prime}$ coefficient.
    """
    i, j_l, f_l, j_u, f_u, two, p = cast_sympy(i, j_l, f_l, j_u, f_u, 2, pi, as_sympy=as_sympy)
    return (two / 3) * (f_0(i, j_l, f_l, j_u, f_u, 0, as_sympy) - f_0(i, j_l, f_l, j_u, f_u, p / 2, as_sympy))


def ab(i: sympy_quant, j_l: sympy_quant, f_l: sympy_quant, j_u: sympy_quant, f_u: sympy_quant, as_sympy: bool = False) \
        -> (Union[sympy_core, float], Union[sympy_core, float]):
    r"""
    The coefficients $A_F^{F^\prime}$ and $B_F^{F^\prime}$, parameterizing the isotropic and the classical
    angle-dependent part of the perturbative differential scattering rate, respectively, as described in
    [<a href="https://doi.org/10.1103/PhysRevA.87.032504">R. C. Brown et al., Phys. Rev. A 87, 032504 (2013)</a>].

    :param i: The nuclear spin quantum number $I$.
    :param j_l: The electronic total angular momentum quantum number $J_l$ of the lower state.
    :param f_l: The total angular momentum quantum number $F_l$ of the lower state.
    :param j_u: The electronic total angular momentum quantum number $J_u$ of the upper state.
    :param f_u: The total angular momentum quantum number $F_u$ of the upper state.
    :param as_sympy: Return the result as a symbol (`True`) or as a `float` (`False`).
    :returns: (A, B) The $A_F^{F^\prime}$ and $B_F^{F^\prime}$ coefficients.
    """
    return a(i, j_l, f_l, j_u, f_u, as_sympy), b(i, j_l, f_l, j_u, f_u, as_sympy)


def c(i: sympy_quant, j_l: sympy_quant, f_l: sympy_quant, j_u: sympy_quant, f1_u: sympy_quant, f2_u: sympy_quant,
      as_sympy: bool = False) -> Union[sympy_core, float]:
    r"""
    The coefficient $C_F^{F^{\prime}F^{\prime\prime}}$, parameterizing the quantum interference part of
    the perturbative differential scattering rate, as described in
    [<a href="https://doi.org/10.1103/PhysRevA.87.032504">R. C. Brown et al., Phys. Rev. A 87, 032504 (2013)</a>].

    :param i: The nuclear spin quantum number $I$.
    :param j_l: The electronic total angular momentum quantum number $J_l$ of the lower state.
    :param f_l: The total angular momentum quantum number $F_l$ of the lower state.
    :param j_u: The electronic total angular momentum quantum number $J_u$ of the upper state.
    :param f1_u: The first total angular momentum quantum number $F_u$ of the upper state.
    :param f2_u: The second total angular momentum quantum number $F_u$ of the upper state.
    :param as_sympy: Return the result as a symbol (`True`) or as a `float` (`False`).
    :returns: (C) The $C_F^{F^{\prime}F^{\prime\prime}}$ coefficient.
    """
    i, j_l, f_l, j_u, f1_u, f2_u = cast_sympy(i, j_l, f_l, j_u, f1_u, f2_u, as_sympy=as_sympy)
    return g_0(i, j_l, f_l, j_u, f1_u, f2_u, 0, as_sympy)


def abc(i: sympy_quant, j_l: sympy_quant, f_l: sympy_quant, j_u: sympy_quant, f1_u: sympy_quant, f2_u: sympy_quant,
        as_sympy: bool = False) -> (Union[sympy_core, float], Union[sympy_core, float], Union[sympy_core, float]):
    r"""
    The coefficients $A_F^{F^\prime}$, $B_F^{F^\prime}$ and $C_F^{F^{\prime}F^{\prime\prime}}$,
    parameterizing the isotropic, classical angle-dependent and quantum interference part of
    the perturbative differential scattering rate, respectively, as described in
    [<a href="https://doi.org/10.1103/PhysRevA.87.032504">R. C. Brown et al., Phys. Rev. A 87, 032504 (2013)</a>].

    :param i: The nuclear spin quantum number $I$.
    :param j_l: The electronic total angular momentum quantum number $J_l$ of the lower state.
    :param f_l: The total angular momentum quantum number $F_l$ of the lower state.
    :param j_u: The electronic total angular momentum quantum number $J_u$ of the upper state.
    :param f1_u: The first total angular momentum quantum number $F_u$ of the upper state.
    :param f2_u: The second total angular momentum quantum number $F_u$ of the upper state.
    :param as_sympy: Return the result as a symbol (`True`) or as a `float` (`False`).
    :returns: (A, B, C) The $A_F^{F^\prime}$, $B_F^{F^\prime}$ and $C_F^{F^{\prime}F^{\prime\prime}}$ coefficients.
    """
    return a(i, j_l, f_l, j_u, f1_u, as_sympy), b(i, j_l, f_l, j_u, f1_u, as_sympy), \
        c(i, j_l, f_l, j_u, f1_u, f2_u, as_sympy)


def f_0(i: sympy_quant, j_l: sympy_quant, f_l: sympy_quant, j_u: sympy_quant, f_u: sympy_quant, theta_l: sympy_like,
        as_sympy: bool = False) -> Union[sympy_core, float]:
    r"""
    The $f$ function, parameterizing the classical part of the perturbative differential scattering rate,
    as described in
    [<a href="https://doi.org/10.1103/PhysRevA.87.032504">R. C. Brown et al., Phys. Rev. A 87, 032504 (2013)</a>].

    :param i: The nuclear spin quantum number $I$.
    :param j_l: The electronic total angular momentum quantum number $J_l$ of the lower state.
    :param f_l: The total angular momentum quantum number $F_l$ of the lower state.
    :param j_u: The electronic total angular momentum quantum number $J_u$ of the upper state.
    :param f_u: The total angular momentum quantum number $F_u$ of the upper state.
    :param theta_l: The angle $\theta_\mathrm{L}$ between the electric field of the linearly polarized incoming photon
     and the direction of detection.
    :param as_sympy: Return the result as a symbol (`True`) or as a `float` (`False`).
    :returns: (f) The value of the $f$ function.
    """
    i, j_l, f_l, j_u, f_u, theta_l, c_sum = cast_sympy(i, j_l, f_l, j_u, f_u, theta_l, 0, as_sympy=as_sympy)
    if abs(f_u - f_l) > 1:
        return c_sum
    m_i_list = [cast_sympy(m - f_l, as_sympy=as_sympy) for m in range(int(2 * f_l + 1))]
    f_f_list = [cast_sympy(f + abs(i - j_l), as_sympy=as_sympy) for f in range(int(i + j_l - abs(i - j_l) + 1))]
    for m_i in m_i_list:
        for f_f in f_f_list:
            if abs(f_u - f_f) > 1:
                continue
            m_f_list = [cast_sympy(m - f_f, as_sympy=as_sympy) for m in range(int(2 * f_f + 1))]
            for m_f in m_f_list:
                c_sum_x = c_dipole(i, j_l, f_l, m_i, j_u, f_u, j_l, f_f, m_f, theta_l, 'x', as_sympy)
                c_sum_y = c_dipole(i, j_l, f_l, m_i, j_u, f_u, j_l, f_f, m_f, theta_l, 'y', as_sympy)
                c_sum_x *= c_sum_x.conjugate()
                c_sum_y *= c_sum_y.conjugate()
                c_sum += c_sum_x + c_sum_y
    if not as_sympy:
        c_sum = c_sum.real
    return 3 * c_sum / (2 * sum([2 * f + 1 for f in f_f_list]))


def g_0(i: sympy_quant, j_l: sympy_quant, f_l: sympy_quant, j_u: sympy_quant,
        f1_u: sympy_quant, f2_u: sympy_quant, theta_l: sympy_like, as_sympy: bool = False) -> Union[sympy_core, float]:
    r"""
    The $g$ function, parameterizing the quantum interference part of the perturbative differential scattering rate,
    as described in
    [<a href="https://doi.org/10.1103/PhysRevA.87.032504">R. C. Brown et al., Phys. Rev. A 87, 032504 (2013)</a>].

    :param i: The nuclear spin quantum number $I$.
    :param j_l: The electronic total angular momentum quantum number $J_l$ of the lower state.
    :param f_l: The total angular momentum quantum number $F_l$ of the lower state.
    :param j_u: The electronic total angular momentum quantum number $J_u$ of the upper state.
    :param f1_u: The first total angular momentum quantum number $F_u$ of the upper state.
    :param f2_u: The second total angular momentum quantum number $F_u$ of the upper state.
    :param theta_l: The angle $\theta_\mathrm{L}$ between the electric field of the linearly polarized incoming photon
     and the direction of detection.
    :param as_sympy: Return the result as a symbol (`True`) or as a `float` (`False`).
    :returns: (g) The value of the $g$ function.
    """
    i, j_l, f_l, j_u, f1_u, f2_u, theta_l, c_sum = cast_sympy(i, j_l, f_l, j_u, f1_u, f2_u, theta_l, 0,
                                                              as_sympy=as_sympy)
    if abs(f1_u - f_l) > 1 or abs(f2_u - f_l) > 1:
        return c_sum
    m_i_list = [cast_sympy(m - f_l, as_sympy=as_sympy) for m in range(int(2 * f_l + 1))]
    f_f_list = [cast_sympy(f + abs(i - j_l), as_sympy=as_sympy) for f in range(int(i + j_l - abs(i - j_l) + 1))]
    for m_i in m_i_list:
        for f_f in f_f_list:
            if abs(f1_u - f_f) > 1 or abs(f2_u - f_f) > 1:
                continue
            m_f_list = [cast_sympy(m - f_f, as_sympy=as_sympy) for m in range(int(2 * f_f + 1))]
            for m_f in m_f_list:
                c_sum_x = c_dipole(i, j_l, f_l, m_i, j_u, f1_u, j_l, f_f, m_f, theta_l, 'x', as_sympy)
                c_sum_y = c_dipole(i, j_l, f_l, m_i, j_u, f1_u, j_l, f_f, m_f, theta_l, 'y', as_sympy)
                c_sum_x *= c_dipole(i, j_l, f_l, m_i, j_u, f2_u, j_l, f_f, m_f, theta_l, 'x', as_sympy).conjugate()
                c_sum_y *= c_dipole(i, j_l, f_l, m_i, j_u, f2_u, j_l, f_f, m_f, theta_l, 'y', as_sympy).conjugate()
                c_sum += c_sum_x + c_sum_y
    if not as_sympy:
        c_sum = c_sum.real
    return 3 * c_sum / (2 * sum([2 * f + 1 for f in f_f_list]))


def c_dipole(i: sympy_quant, j_i: sympy_quant, f_i: sympy_quant, m_i: sympy_quant, j_u: sympy_quant, f_u: sympy_quant,
             j_f: sympy_quant, f_f: sympy_quant, m_f: sympy_quant,
             theta_l: sympy_like, scatter_pol: str, as_sympy: bool = False) -> Union[sympy_core, float]:
    r"""
    The transition dipole element $C_{i\rightarrow f}^{F^\prime}$, as described in
    [<a href="https://doi.org/10.1103/PhysRevA.87.032504">R. C. Brown et al., Phys. Rev. A 87, 032504 (2013)</a>].

    :param i: The nuclear spin quantum number $I$.
    :param j_i: The electronic total angular momentum quantum number $J_i$ of the initial lower state.
    :param f_i: The total angular momentum quantum number $F_i$ of the initial lower state.
    :param m_i: The magnetic quantum number $m_i$ of the total angular moment of the initial lower state.
    :param j_u: The electronic total angular momentum quantum number $J_u$ of the intermediate upper state.
    :param f_u: The total angular momentum quantum number $F_u$ of the intermediate upper state.
    :param j_f: The electronic total angular momentum quantum number $J_f$ of the final lower state.
    :param f_f: The total angular momentum quantum number $F_f$ of the final lower state.
    :param m_f: The magnetic quantum number $m_f$ of the total angular moment of the final lower state.
    :param theta_l: The angle $\theta_\mathrm{L}$ between the electric field of the linearly polarized incoming photon
     and the direction of detection.
    :param scatter_pol: The label for the two orthogonal polarizations of the scattered light.
     Can be either `'x'` or anything else.
    :param as_sympy: Return the result as a symbol (`True`) or as a `float` (`False`).
    :returns: (c_dipole) The transition dipole element $C_{i\rightarrow f}^{F^\prime}$.
    """
    i, j_i, f_i, m_i, j_u, f_u, j_f, f_f, m_f, theta_l = \
        cast_sympy(i, j_i, f_i, m_i, j_u, f_u, j_f, f_f, m_f, theta_l, as_sympy=as_sympy)
    c_element = cast_sympy(0.j, as_sympy=as_sympy, dtype=complex)
    sqrt_2 = 1 / sqrt(2) if as_sympy else 1 / np.sqrt(2)
    c1 = nsimplify(1j * sqrt_2) if as_sympy else 1j * sqrt_2
    sin_t = sin(theta_l) if as_sympy else np.sin(theta_l)
    cos_t = cos(theta_l) if as_sympy else np.cos(theta_l)
    m_u_list = [cast_sympy(m - f_u, as_sympy=as_sympy) for m in range(int(2 * f_u + 1))]
    for m_u in m_u_list:
        if abs(m_u - m_i) > 1 or abs(m_u - m_f) > 1:
            continue
        a_l_element = -c1 * sin_t * (a_dipole(i, j_i, f_i, m_i, j_u, f_u, m_u, 1, as_sympy)
                                     + a_dipole(i, j_i, f_i, m_i, j_u, f_u, m_u, -1, as_sympy)) \
            + cos_t * a_dipole(i, j_i, f_i, m_i, j_u, f_u, m_u, 0, as_sympy)
        if scatter_pol == 'x':
            c_element += -sqrt_2 * (a_dipole(i, j_f, f_f, m_f, j_u, f_u, m_u, 1, as_sympy)
                                    - a_dipole(i, j_f, f_f, m_f, j_u, f_u, m_u, -1, as_sympy)) * a_l_element
        else:
            c_element += c1 * (a_dipole(i, j_f, f_f, m_f, j_u, f_u, m_u, 1, as_sympy)
                               + a_dipole(i, j_f, f_f, m_f, j_u, f_u, m_u, -1, as_sympy)) * a_l_element
    return c_element


def a_dipole(i: sympy_quant, j_l: sympy_quant, f_l: sympy_quant, m_l: sympy_quant,
             j_u: sympy_quant, f_u: sympy_quant, m_u: sympy_quant, q: sympy_quant, as_sympy: bool = False) \
        -> Union[sympy_core, float]:
    r"""
    The spherical transition dipole element $(A_{Fm}^{F^\prime m^\prime})_q$, as described in
    [<a href="https://doi.org/10.1103/PhysRevA.87.032504">R. C. Brown et al., Phys. Rev. A 87, 032504 (2013)</a>].
    
    :param i: The nuclear spin quantum number $I$.
    :param j_l: The electronic total angular momentum quantum number $J_l$ of the lower state.
    :param f_l: The total angular momentum quantum number $F_l$ of the lower state.
    :param m_l: The magnetic quantum number $m_l$ of the total angular moment of the lower state.
    :param j_u: The electronic total angular momentum quantum number $J_u$ of the upper state.
    :param f_u: The total angular momentum quantum number $F_u$ of the upper state.
    :param m_u: The magnetic quantum number $m_u$ of the total angular moment of the upper state.
    :param q: The polarization component $q$. Must be in `{-1, 0, 1}` for $\sigma^-$, $\pi$ and $\sigma^+$ light.
    :param as_sympy: Return the result as a symbol (`True`) or as a `float` (`False`).
    :returns: (a_dipole) The spherical transition dipole element $(A_{Fm}^{F^\prime m^\prime})_q$.
    """
    i, j_l, f_l, m_l, j_u, f_u, m_u, q = cast_sympy(i, j_l, f_l, m_l, j_u, f_u, m_u, q, as_sympy=as_sympy)
    if m_u - m_l != q:
        return cast_sympy(0, as_sympy=as_sympy)
    sqrt_f = sqrt(2 * f_l + 1) if as_sympy else np.sqrt(2 * f_l + 1)
    sqrt_j = sqrt(2 * j_u + 1) if as_sympy else np.sqrt(2 * j_u + 1)
    exp = cast_sympy(f_l + i + 1 + j_u, as_sympy=as_sympy, dtype=int)
    return (-1) ** exp * sqrt_f * sqrt_j * wigner_6j(j_u, j_l, 1, f_l, f_u, i, as_sympy) \
        * clebsch_gordan(f_l, 1, f_u, m_l, q, m_u, as_sympy)


def a_dipole_cart(i: sympy_quant, j_l: sympy_quant, f_l: sympy_quant, m_l: sympy_quant,
                  j_u: sympy_quant, f_u: sympy_quant, m_u: sympy_quant, as_sympy: bool = False) \
        -> Union[Vector, ndarray]:
    r"""
    The cartesian transition dipole element vector

    $$\vec{A}_{Fm}^{F^\prime m^\prime} = \begin{pmatrix}
    \frac{1}{\sqrt{2}}\left[(A_{Fm}^{F^\prime m^\prime})_{-1} - (A_{Fm}^{F^\prime m^\prime})_1\right]\\
    \frac{i}{\sqrt{2}}\left[(A_{Fm}^{F^\prime m^\prime})_{-1} + (A_{Fm}^{F^\prime m^\prime})_1\right]\\
    (A_{Fm}^{F^\prime m^\prime})_0
    \end{pmatrix},$$
    with $(A_{Fm}^{F^\prime m^\prime})_q$ as in <a href="{{ '/doc/functions/algebra/a_dipole.html' | relative_url }}">
    `a_dipole`</a>.
    See [<a href="https://doi.org/10.1103/PhysRevA.87.032504">R. C. Brown et al., Phys. Rev. A 87, 032504 (2013)</a>].

    :param i: The nuclear spin quantum number $I$.
    :param j_l: The electronic total angular momentum quantum number $J_l$ of the lower state.
    :param f_l: The total angular momentum quantum number $F_l$ of the lower state.
    :param m_l: The magnetic quantum number $m_l$ of the total angular moment of the lower state.
    :param j_u: The electronic total angular momentum quantum number $J_u$ of the upper state.
    :param f_u: The total angular momentum quantum number $F_u$ of the upper state.
    :param m_u: The magnetic quantum number $m_u$ of the total angular moment of the upper state.
    :param as_sympy: Return the result as a symbol (`True`) or as a `float` (`False`).
    :returns: (a_dipole_cart) The cartesian transition dipole element vector $\vec{A}_{Fm}^{F^\prime m^\prime}$.
    """
    i, j_l, f_l, m_l, j_u, f_u, m_u = cast_sympy(i, j_l, f_l, m_l, j_u, f_u, m_u, as_sympy=as_sympy)
    sqrt_2 = nsimplify(1 / sqrt(2)) if as_sympy else 1 / np.sqrt(2)
    x = sqrt_2 * (a_dipole(i, j_l, f_l, m_l, j_u, f_u, m_u, -1, as_sympy)
                  - a_dipole(i, j_l, f_l, m_l, j_u, f_u, m_u, 1, as_sympy))
    y = 1j * sqrt_2 * (a_dipole(i, j_l, f_l, m_l, j_u, f_u, m_u, -1, as_sympy)
                       + a_dipole(i, j_l, f_l, m_l, j_u, f_u, m_u, 1, as_sympy))
    z = a_dipole(i, j_l, f_l, m_l, j_u, f_u, m_u, 0, as_sympy)
    if as_sympy:
        a3 = CoordSys3D('a3')
        return x * a3.i + y * a3.j + z * a3.k
    return np.array([x, y, z], dtype=complex)


def reduced_f_root(i: sympy_quant, j_l: sympy_quant, f_l: sympy_quant,
                   j_u: sympy_quant, f_u: sympy_quant, as_sympy: bool = False) -> Union[sympy_core, float]:
    r"""
    The geometric coefficient

    $$\sqrt{f_F^{F^\prime}} = (-1)^{F + I + 1 + J_u}\sqrt{2F_l + 1}\sqrt{2F_u + 1}\begin{Bmatrix}
    J_u & J_l & 1 \\
    F_l & F_u & I
    \end{Bmatrix},$$

    as described in
    [<a href="https://doi.org/10.1103/PhysRevA.87.032504">R. C. Brown et al., Phys. Rev. A 87, 032504 (2013)</a>].

    :param i: The nuclear spin quantum number $I$.
    :param j_l: The electronic total angular momentum quantum number $J_l$ of the lower state.
    :param f_l: The total angular momentum quantum number $F_l$ of the lower state.
    :param j_u: The electronic total angular momentum quantum number $J_u$ of the upper state.
    :param f_u: The total angular momentum quantum number $F_u$ of the upper state.
    :param as_sympy: Return the result as a symbol (`True`) or as a `float` (`False`).
    :returns: The geometric coefficient $\sqrt{f_F^{F^\prime}}$.
    """
    i, j_l, f_l, j_u, f_u = cast_sympy(i, j_l, f_l, j_u, f_u, as_sympy=as_sympy)
    sqrt_fl = sqrt(2 * f_l + 1) if as_sympy else np.sqrt(2 * f_l + 1)
    sqrt_fu = sqrt(2 * f_u + 1) if as_sympy else np.sqrt(2 * f_u + 1)
    exp = cast_sympy(f_l + i + 1 + j_u, as_sympy=as_sympy, dtype=int)
    # == (...) * wigner_6j(f_u, 1, f_l, j_l, i, j_u, as_sympy)
    return (-1) ** exp * sqrt_fl * sqrt_fu * wigner_6j(j_u, j_l, 1, f_l, f_u, i, as_sympy)


def reduced_f(i: sympy_quant, j_l: sympy_quant, f_l: sympy_quant,
              j_u: sympy_quant, f_u: sympy_quant, as_sympy: bool = False) -> Union[sympy_core, float]:
    r"""
    The geometric coefficient

    $$f_F^{F^\prime} = (2F_l + 1)(2F_u + 1)\begin{Bmatrix}
    J_u & J_l & 1 \\
    F_l & F_u & I
    \end{Bmatrix}^2,$$

    as described in
    [<a href="https://doi.org/10.1103/PhysRevA.87.032504">R. C. Brown et al., Phys. Rev. A 87, 032504 (2013)</a>].

    :param i: The nuclear spin quantum number $I$.
    :param j_l: The electronic total angular momentum quantum number $J_l$ of the lower state.
    :param f_l: The total angular momentum quantum number $F_l$ of the lower state.
    :param j_u: The electronic total angular momentum quantum number $J_u$ of the upper state.
    :param f_u: The total angular momentum quantum number $F_u$ of the upper state.
    :param as_sympy: Return the result as a symbol (`True`) or as a `float` (`False`).
    :returns: The geometric coefficient $f_F^{F^\prime}$.
    """
    i, j_l, f_l, j_u, f_u = cast_sympy(i, j_l, f_l, j_u, f_u, as_sympy=as_sympy)
    return (2 * f_l + 1) * (2 * f_u + 1) * wigner_6j(j_u, j_l, 1, f_l, f_u, i, as_sympy) ** 2


""" Relative hyperfine structure transition strengths used in TILDA (https://github.com/lasersphere/Tilda). """

def a_tilda(i: sympy_quant, j_l: sympy_quant, f_l: sympy_quant,
            j_u: sympy_quant, f_u: sympy_quant, as_sympy: bool = False) -> Union[sympy_core, float]:
    r"""
    The relative peak strength in a hyperfine structure spectrum

    $$f_F^{F^\prime} = (2F_l + 1)(2F_u + 1)
    \begin{Bmatrix}
    J_u & J_l & 1 \\
    F_l & F_u & I
    \end{Bmatrix}^2,$$

    as used in <a href="https://github.com/lasersphere/Tilda">
    `TILDA`</a>.

    :param i: The nuclear spin quantum number $I$.
    :param j_l: The electronic total angular momentum quantum number $J_l$ of the lower state.
    :param f_l: The total angular momentum quantum number $F_l$ of the lower state.
    :param j_u: The electronic total angular momentum quantum number $J_u$ of the upper state.
    :param f_u: The total angular momentum quantum number $F_u$ of the upper state.
    :param as_sympy: Return the result as a symbol (`True`) or as a `float` (`False`).
    :returns: The relative peak strength $f_F^{F^\prime}$.
    """
    return reduced_f(i, j_l, f_l, j_u, f_u, as_sympy=as_sympy)


def a_m_tilda(i: sympy_quant, j_l: sympy_quant, f_l: sympy_quant, m_l: sympy_quant,
              j_u: sympy_quant, f_u: sympy_quant, m_u: sympy_quant, as_sympy: bool = False) -> Union[sympy_core, float]:
    r"""
    The relative peak strength in a hyperfine structure Zeeman spectrum

    $$f_{Fm}^{F^\prime m^\prime} = (2F_l + 1)(2F_u + 1)\langle F_l, m_l, 1, m_u - m_l\, |\, F_u, m_u\rangle^2
    \begin{Bmatrix}
    J_u & J_l & 1 \\
    F_l & F_u & I
    \end{Bmatrix}^2,$$

    as used in <a href="https://github.com/lasersphere/Tilda">
    `TILDA`</a>.

    :param i: The nuclear spin quantum number $I$.
    :param j_l: The electronic total angular momentum quantum number $J_l$ of the lower state.
    :param f_l: The total angular momentum quantum number $F_l$ of the lower state.
    :param m_l: The magnetic quantum number $m_l$ of the total angular moment of the lower state.
    :param j_u: The electronic total angular momentum quantum number $J_u$ of the upper state.
    :param f_u: The total angular momentum quantum number $F_u$ of the upper state.
    :param m_u: The magnetic quantum number $m_u$ of the total angular moment of the upper state.
    :param as_sympy: Return the result as a symbol (`True`) or as a `float` (`False`).
    :returns: The relative peak strength $f_{Fm}^{F^\prime m^\prime}$.
    """
    return a_tilda(i, j_l, f_l, j_u, f_u, as_sympy) * clebsch_gordan(f_l, 1, f_u, m_l, m_u - m_l, m_u, as_sympy) ** 2


""" Off-diagonal magnetic dipole transition matrix elements for Delta S = Delta L = 0. """

def mu_j_m1(s: sympy_quant, l: sympy_quant, j_l: sympy_quant, j_u: sympy_quant, as_sympy: bool = False) \
        -> Union[sympy_core, float]:
    r"""
    The reduced magnetic dipole transition matrix element

    $$\mu_{J_l}^{J_u}\coloneqq\langle LS,J_l||\vec{\mu}||LS,J_u\rangle$$

    in units of the Bohr magneton $\mu_\mathrm{B}$.

    :param s: The electronic spin quantum number $S$ of the lower and upper state.
    :param l: The electronic orbital angular momentum quantum number $L$ of the lower and upper state.
    :param j_l: The electronic total angular momentum quantum number $J_l$ of the lower state.
    :param j_u: The electronic total angular momentum quantum number $J_u$ of the upper state.
    :param as_sympy: Return the result as a symbol (`True`) or as a `float` (`False`).
    :returns: The magnetic dipole transition matrix element $\mu_{J_l}^{J_u}$ ($\mu_\mathrm{B}$).
    """
    s, l, j_l, j_u = cast_sympy(s, l, j_l, j_u, as_sympy=as_sympy)
    g_s = cast_sympy(physical_constants['electron g factor'][0], as_sympy=as_sympy)

    sqrt_s = sqrt(s * (s + 1) * (2 * s + 1)) if as_sympy else np.sqrt(s * (s + 1) * (2 * s + 1))
    sqrt_l = sqrt(l * (l + 1) * (2 * l + 1)) if as_sympy else np.sqrt(l * (l + 1) * (2 * l + 1))
    sqrt_j = sqrt((2 * j_l + 1) * (2 * j_u + 1)) if as_sympy else np.sqrt((2 * j_l + 1) * (2 * j_u + 1))

    exp_sl = cast_sympy(s + l + 1, as_sympy=as_sympy, dtype=float)
    exp_j = cast_sympy(j_u + 1, as_sympy=as_sympy, dtype=float)

    return ((-1) ** exp_sl * sqrt_j * (
            (-1) ** exp_j * sqrt_l * wigner_6j(l, j_l, s, j_u, l, 1, as_sympy)
            + g_s * (-1) ** j_l * sqrt_s * wigner_6j(s, j_l, l, j_u, s, 1, as_sympy)))


def mu_jj_m1(sc: sympy_quant, lc: sympy_quant, jc_l: sympy_quant, jo_l: sympy_quant, j_l: sympy_quant, so: sympy_quant,
             lo: sympy_quant, jc_u: sympy_quant, jo_u: sympy_quant, j_u: sympy_quant, as_sympy: bool = False) \
        -> Union[sympy_core, float]:
    r"""
    The reduced magnetic dipole transition matrix element

    $$\mu_{J_l}^{J_u}\coloneqq\langle j_{l,c}j_{l,o},J_l||
    \vec{\mu}||j_{u,c}j_{u,o},J_u\rangle$$

    in units of the Bohr magneton $\mu_\mathrm{B}$.

    :param sc: The spin quantum number $s_c$ of the core electron(s) of the lower and upper state.
    :param lc: The orbital angular momentum quantum number $l_c$ of the core electron(s) of the lower and upper state.
    :param jc_l: The total angular momentum quantum number $j_{l,c}$ of the core electron(s) of the lower state.
    :param jo_l: The total angular momentum quantum number $j_{l,o}$ of the outer electron(s) of the lower state.
    :param j_l: The total angular momentum quantum number $J_l$ of the lower state.
    :param so: The spin quantum number $s_o$ of the outer electron(s) of the lower and upper state.
    :param lo: The orbital angular momentum quantum number $l_o$ of the outer electron(s) of the lower and upper state.
    :param jc_u: The total angular momentum quantum number $j_{u,c}$ of the core electron(s) of the upper state.
    :param jo_u: The total angular momentum quantum number $j_{u,o}$ of the outer electron(s) of the upper state.
    :param j_u: The total angular momentum quantum number $J_u$ of the upper state.
    :param as_sympy: Return the result as a symbol (`True`) or as a `float` (`False`).
    :returns: The magnetic dipole transition matrix element $\mu_{J_l}^{J_u}$ ($\mu_\mathrm{B}$).
    """
    sc, lc, so, lo, jc_l, jo_l, j_l, jc_u, jo_u, j_u = (
        cast_sympy(sc, lc, so, lo, jc_l, jo_l, j_l, jc_u, jo_u, j_u, as_sympy=as_sympy))

    ret = 0.
    sqrt_j = sqrt((2 * j_l + 1) * (2 * j_u + 1)) if as_sympy else np.sqrt((2 * j_l + 1) * (2 * j_u + 1))

    if jo_l == jo_u:
        exp = cast_sympy(jc_l + jo_l + j_u + 1, as_sympy=as_sympy, dtype=float)
        ret += ((-1) ** exp * sqrt_j
                * wigner_6j(jc_l, j_l, jo_l, j_u, jc_u, 1, as_sympy)
                * mu_j_m1(sc, lc, jc_l, jc_u, as_sympy))

    if jc_l == jc_u:
        exp = cast_sympy(jc_l + jo_u + j_l + 1, as_sympy=as_sympy, dtype=float)
        ret += ((-1) ** exp * sqrt_j
                * wigner_6j(jo_l, j_l, jc_l, j_u, jo_u, 1, as_sympy)
                * mu_j_m1(so, lo, jo_l, jo_u, as_sympy))

    return ret.real
