# -*- coding: utf-8 -*-
"""
qspec.algebra
=============

Created on 30.04.2020

@author: Patrick Mueller

Module including methods for calculating dipole coefficients.
"""

import numpy as np
from sympy import nsimplify, sqrt, sin, cos, pi
from sympy.vector import CoordSys3D
import sympy.physics.wigner as spw

from qspec._types import *

__all__ = ['cast_sympy', 'clebsch_gordan', 'wigner_3j', 'wigner_6j', 'a', 'b', 'ab', 'c', 'abc', 'f_0', 'g_0',
           'c_dipole', 'a_dipole', 'a_dipole_cart', 'reduced_f_root', 'reduced_f', 'a_tilda', 'a_m_tilda']


def cast_sympy(as_sympy: bool, *args: sympy_like, dtype: type = float):
    """
    :param as_sympy: Whether to return the result as a sympy type.
    :param args: sympy_like arguments.
    :param dtype: The type to use if as_sympy is False.
    :returns: The specified arguments as sympy types if 'as_sympy' is True and else as floats.
    """
    ret = tuple(nsimplify(arg) if as_sympy else dtype(arg) for arg in args)
    if len(ret) == 1:
        ret = ret[0]
    return ret


def clebsch_gordan(j_1: sympy_qn, j_2: sympy_qn, j_3: sympy_qn, m_1: sympy_qn, m_2: sympy_qn, m_3: sympy_qn,
                   as_sympy: bool = False):
    """
    .. math::

        \\langle J_1, m_1, J_2, m_2\\, |\\, J_3, m_3\\rangle

    :param j_1: J1.
    :param j_2: J2.
    :param j_3: The coupled J3 <- J1 + J2.
    :param m_1: m1.
    :param m_2: m2.
    :param m_3: The coupled m3 <- m1 + m2.
    :param as_sympy: Whether to return the result as a sympy type.
    :returns: The Clebsch-Gordan coefficient.
    """
    ret = spw.clebsch_gordan(j_1, j_2, j_3, m_1, m_2, m_3)
    return cast_sympy(as_sympy, ret)


def wigner_3j(j_1: sympy_qn, j_2: sympy_qn, j_3: sympy_qn, m_1: sympy_qn, m_2: sympy_qn, m_3: sympy_qn,
              as_sympy: bool = False):
    """
    .. math::

        \\begin{pmatrix}
        J_1 & J_2 & J_3 \\\\
        m_1 & m_2 & m_3
        \\end{pmatrix}

    :param j_1: J1.
    :param j_2: J2.
    :param j_3: The coupled J3 <- J1 + J2.
    :param m_1: m1.
    :param m_2: m2.
    :param m_3: The coupled m3 <- m1 + m2.
    :param as_sympy: Whether to return the result as a sympy type.
    :returns: The wigner-3j symbol.

    """
    ret = spw.wigner_3j(j_1, j_2, j_3, m_1, m_2, m_3)
    return cast_sympy(as_sympy, ret)


def wigner_6j(j_1: sympy_qn, j_2: sympy_qn, j_3: sympy_qn, j_4: sympy_qn, j_5: sympy_qn, j_6: sympy_qn,
              as_sympy: bool = False):
    """
    .. math::

        \\begin{Bmatrix}
        J_1 & J_2 & J_3 \\\\
        J_4 & J_5 & J_6
        \\end{Bmatrix}

    :param j_1: :math:`J_1`.
    :param j_2: :math:`J_2`.
    :param j_3: :math:`J_3`.
    :param j_4: :math:`J_4`.
    :param j_5: :math:`J_5`.
    :param j_6: :math:`J_6`.
    :param as_sympy: Whether to return the result as a sympy type.
    :returns: The wigner-6j symbol.
    """
    ret = spw.wigner_6j(j_1, j_2, j_3, j_4, j_5, j_6)
    return cast_sympy(as_sympy, ret)


""" Linear polarization absorption dipole coefficients a, b and c.
    Refer to [Brown et al., Phys. Rev. A 87, 032504 (2013)]. """


def a(i: sympy_qn, j_l: sympy_qn, f_l: sympy_qn, j_u: sympy_qn, f_u: sympy_qn, as_sympy: bool = False) \
        -> Union[sympy_core, float]:
    """
    :param i: The nuclear spin quantum number I.
    :param j_l: The electronic total angular momentum quantum number J of the lower state.
    :param f_l: The total angular momentum quantum number F of the lower state.
    :param j_u: The electronic total angular momentum quantum number J of the upper state.
    :param f_u: The total angular momentum quantum number F of the upper state.
    :param as_sympy: Whether to return the result as a sympy type.
    :returns: The A coefficient as an S-object or float.
    """
    ret = spw.wigner_6j(j_u, j_l, 1, f_l, f_u, i) ** 2
    i, j_l, f_l, j_u, f_u, ret = cast_sympy(as_sympy, i, j_l, f_l, j_u, f_u, ret)
    return ret * (2 * f_l + 1) * (2 * f_u + 1) * (2 * j_u + 1) / (3 * (2 * j_l + 1) * (2 * i + 1))
    # i, j_l, f_l, j_u, f_u = cast_sympy(as_sympy, i, j_l, f_l, j_u, f_u)
    # return f_0(i, j_l, f_l, j_u, f_u, 0, as_sympy=as_sympy) - b(i, j_l, f_l, j_u, f_u, as_sympy=as_sympy)  # (slower)


def b(i: sympy_qn, j_l: sympy_qn, f_l: sympy_qn, j_u: sympy_qn, f_u: sympy_qn, as_sympy: bool = False) \
        -> Union[sympy_core, float]:
    """
    :param i: The nuclear spin quantum number I.
    :param j_l: The electronic total angular momentum quantum number J of the lower state.
    :param j_u: The electronic total angular momentum quantum number J of the upper state.
    :param f_l: The total angular momentum quantum number F of the lower state.
    :param f_u: The total angular momentum quantum number F of the upper state.
    :param as_sympy: Whether to return the result as a sympy type.
    :returns: The B coefficient as an S-object or float.
    """
    i, j_l, f_l, j_u, f_u, two, p = cast_sympy(as_sympy, i, j_l, f_l, j_u, f_u, 2, pi)
    return (two / 3) * (f_0(i, j_l, f_l, j_u, f_u, 0, as_sympy) - f_0(i, j_l, f_l, j_u, f_u, p / 2, as_sympy))


def ab(i: sympy_qn, j_l: sympy_qn, f_l: sympy_qn, j_u: sympy_qn, f_u: sympy_qn, as_sympy: bool = False) \
        -> (Union[sympy_core, float], Union[sympy_core, float]):
    """
    :param i: The nuclear spin quantum number I.
    :param j_l: The electronic total angular momentum quantum number J of the lower state.
    :param j_u: The electronic total angular momentum quantum number J of the upper state.
    :param f_l: The total angular momentum quantum number F of the lower state.
    :param f_u: The total angular momentum quantum number F of the upper state.
    :param as_sympy: Whether to return the result as a sympy type.
    :returns: The A and B coefficients as S-objects or floats.
    """
    return a(i, j_l, f_l, j_u, f_u, as_sympy), b(i, j_l, f_l, j_u, f_u, as_sympy)


def c(i: sympy_qn, j_l: sympy_qn, f_l: sympy_qn, j_u: sympy_qn, f1_u: sympy_qn, f2_u: sympy_qn,
      as_sympy: bool = False) -> Union[sympy_core, float]:
    """
    :param i: The nuclear spin quantum number I.
    :param j_l: The electronic total angular momentum quantum number J of the lower state.
    :param j_u: The electronic total angular momentum quantum number J of the upper state.
    :param f_l: The total angular momentum quantum number F of the lower state.
    :param f1_u: The first total angular momentum quantum number F of the upper state.
    :param f2_u: The second total angular momentum quantum number F of the upper state.
    :param as_sympy: Whether to return the result as a sympy type.
    :returns: The C coefficient as an S-object.
    """
    i, j_l, f_l, j_u, f1_u, f2_u = cast_sympy(as_sympy, i, j_l, f_l, j_u, f1_u, f2_u)
    return g_0(i, j_l, f_l, j_u, f1_u, f2_u, 0, as_sympy)


def abc(i: sympy_qn, j_l: sympy_qn, f_l: sympy_qn, j_u: sympy_qn, f1_u: sympy_qn, f2_u: sympy_qn,
        as_sympy: bool = False) -> (Union[sympy_core, float], Union[sympy_core, float], Union[sympy_core, float]):
    """
    :param i: The nuclear spin quantum number I.
    :param j_l: The electronic total angular momentum quantum number J of the lower state.
    :param j_u: The electronic total angular momentum quantum number J of the upper state.
    :param f_l: The total angular momentum quantum number F of the lower state.
    :param f1_u: The first total angular momentum quantum number F of the upper state. This is used for A and B.
    :param f2_u: The second total angular momentum quantum number F of the upper state.
    :param as_sympy: Whether to return the result as a sympy type.
    :returns: The A, B and C coefficient as S-objects.
    """
    return a(i, j_l, f_l, j_u, f1_u, as_sympy), b(i, j_l, f_l, j_u, f1_u, as_sympy), \
        c(i, j_l, f_l, j_u, f1_u, f2_u, as_sympy)


def f_0(i: sympy_qn, j_l: sympy_qn, f_l: sympy_qn, j_u: sympy_qn, f_u: sympy_qn, theta_l: sympy_like,
        as_sympy: bool = False) -> Union[sympy_core, float]:
    """
    :param i: The nuclear spin quantum number I.
    :param j_l: The electronic total angular momentum quantum number J of the lower state.
    :param j_u: The electronic total angular momentum quantum number J of the upper state.
    :param f_l: The total angular momentum quantum number F of the lower state.
    :param f_u: The total angular momentum quantum number F of the upper state.
    :param theta_l: The angle between the electric field of the linear laser polarisation
     and the direction of detection.
    :param as_sympy: Whether to return the result as a sympy type.
    :returns: The value of the f_0 function at the specified position.
     See [Brown et al., Phys. Rev. A 87, 032504 (2013)].
    """
    i, j_l, f_l, j_u, f_u, theta_l, c_sum = cast_sympy(as_sympy, i, j_l, f_l, j_u, f_u, theta_l, 0)
    if abs(f_u - f_l) > 1:
        return c_sum
    m_i_list = [cast_sympy(as_sympy, m - f_l) for m in range(int(2 * f_l + 1))]
    f_f_list = [cast_sympy(as_sympy, f + abs(i - j_l)) for f in range(int(i + j_l - abs(i - j_l) + 1))]
    for m_i in m_i_list:
        for f_f in f_f_list:
            if abs(f_u - f_f) > 1:
                continue
            m_f_list = [cast_sympy(as_sympy, m - f_f) for m in range(int(2 * f_f + 1))]
            for m_f in m_f_list:
                c_sum_x = c_dipole(i, j_l, f_l, m_i, j_u, f_u, j_l, f_f, m_f, theta_l, 'x', as_sympy)
                c_sum_y = c_dipole(i, j_l, f_l, m_i, j_u, f_u, j_l, f_f, m_f, theta_l, 'y', as_sympy)
                c_sum_x *= c_sum_x.conjugate()
                c_sum_y *= c_sum_y.conjugate()
                c_sum += c_sum_x + c_sum_y
    if not as_sympy:
        c_sum = c_sum.real
    return 3 * c_sum / (2 * sum([2 * f + 1 for f in f_f_list]))


def g_0(i: sympy_qn, j_l: sympy_qn, f_l: sympy_qn, j_u: sympy_qn, f1_u: sympy_qn, f2_u: sympy_qn, theta_l: sympy_like,
        as_sympy: bool = False) -> Union[sympy_core, float]:
    """
    :param i: The nuclear spin quantum number I.
    :param j_l: The electronic total angular momentum quantum number J of the lower state.
    :param j_u: The electronic total angular momentum quantum number J of the upper state.
    :param f_l: The total angular momentum quantum number F of the lower state.
    :param f1_u: The first total angular momentum quantum number F of the upper state.
    :param f2_u: The second total angular momentum quantum number F of the upper state.
    :param theta_l: The angle between the electric field of the linear laser polarisation
     and the direction of detection.
    :param as_sympy: Whether to return the result as a sympy type.
    :returns: The value of the g_0 function at the specified position.
     See [Brown et al., Phys. Rev. A 87, 032504 (2013)].
    """
    i, j_l, f_l, j_u, f1_u, f2_u, theta_l, c_sum = cast_sympy(as_sympy, i, j_l, f_l, j_u, f1_u, f2_u, theta_l, 0)
    if abs(f1_u - f_l) > 1 or abs(f2_u - f_l) > 1:
        return c_sum
    m_i_list = [cast_sympy(as_sympy, m - f_l) for m in range(int(2 * f_l + 1))]
    f_f_list = [cast_sympy(as_sympy, f + abs(i - j_l)) for f in range(int(i + j_l - abs(i - j_l) + 1))]
    for m_i in m_i_list:
        for f_f in f_f_list:
            if abs(f1_u - f_f) > 1 or abs(f2_u - f_f) > 1:
                continue
            m_f_list = [cast_sympy(as_sympy, m - f_f) for m in range(int(2 * f_f + 1))]
            for m_f in m_f_list:
                c_sum_x = c_dipole(i, j_l, f_l, m_i, j_u, f1_u, j_l, f_f, m_f, theta_l, 'x', as_sympy)
                c_sum_y = c_dipole(i, j_l, f_l, m_i, j_u, f1_u, j_l, f_f, m_f, theta_l, 'y', as_sympy)
                c_sum_x *= c_dipole(i, j_l, f_l, m_i, j_u, f2_u, j_l, f_f, m_f, theta_l, 'x', as_sympy).conjugate()
                c_sum_y *= c_dipole(i, j_l, f_l, m_i, j_u, f2_u, j_l, f_f, m_f, theta_l, 'y', as_sympy).conjugate()
                c_sum += c_sum_x + c_sum_y
    if not as_sympy:
        c_sum = c_sum.real
    return 3 * c_sum / (2 * sum([2 * f + 1 for f in f_f_list]))


def c_dipole(i: sympy_qn, j_i: sympy_qn, f_i: sympy_qn, m_i: sympy_qn, j_u: sympy_qn, f_u: sympy_qn,
             j_f: sympy_qn, f_f: sympy_qn, m_f: sympy_qn,
             theta_l: sympy_like, scatter_pol: str, as_sympy: bool = False) -> Union[sympy_core, float]:
    """
    :param i: The nuclear spin quantum number I.
    :param j_i: The electronic total angular momentum quantum number J of the initial lower state.
    :param f_i: The total angular momentum quantum number F of the initial lower state.
    :param m_i: The z-projection quantum number m of the total angular moment of the initial lower state
    :param j_u: The electronic total angular momentum quantum number J of the intermediate upper state.
    :param f_u: The total angular momentum quantum number F of the intermediate upper state.
    :param j_f: The electronic total angular momentum quantum number J of the final lower state.
    :param f_f: The total angular momentum quantum number F of the final lower state.
    :param m_f: The z-projection quantum number m of the total angular moment of the final lower state.
    :param theta_l: The angle between the electric field of the linear laser polarisation
     and the direction of detection.
    :param scatter_pol: The label for the two orthogonal polarizations of the scattered light.
     Can be either 'x' or anything else.
    :param as_sympy: Whether to return the result as a sympy type.
    :returns: The transition dipole element C_{i->f}^{F'} as defined in [Brown et al., Phys. Rev. A 87, 032504 (2013)].
    """
    i, j_i, f_i, m_i, j_u, f_u, j_f, f_f, m_f, theta_l = \
        cast_sympy(as_sympy, i, j_i, f_i, m_i, j_u, f_u, j_f, f_f, m_f, theta_l)
    c_element = cast_sympy(as_sympy, 0.j, dtype=complex)
    sqrt_2 = 1 / sqrt(2) if as_sympy else 1 / np.sqrt(2)
    c1 = nsimplify(1j * sqrt_2) if as_sympy else 1j * sqrt_2
    sin_t = sin(theta_l) if as_sympy else np.sin(theta_l)
    cos_t = cos(theta_l) if as_sympy else np.cos(theta_l)
    m_u_list = [cast_sympy(as_sympy, m - f_u) for m in range(int(2 * f_u + 1))]
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


def a_dipole(i: sympy_qn, j_l: sympy_qn, f_l: sympy_qn, m_l: sympy_qn,
             j_u: sympy_qn, f_u: sympy_qn, m_u: sympy_qn, q: sympy_qn, as_sympy: bool = False) \
        -> Union[sympy_core, float]:
    """
    :param i: The nuclear spin quantum number I.
    :param j_l: The electronic total angular momentum quantum number J of the lower state.
    :param f_l: The total angular momentum quantum number F of the lower state.
    :param m_l: The z-projection quantum number m of the total angular moment of the lower state
    :param j_u: The electronic total angular momentum quantum number J of the upper state.
    :param f_u: The total angular momentum quantum number F of the upper state.
    :param m_u: The z-projection quantum number m of the total angular moment of the upper state.
    :param q: The polarisation component. Can be either -1, 0 or 1 for sigma-, pi or sigma+ light.
    :param as_sympy: Whether to return the result as a sympy type.
    :returns: The q component of the spherical dipole matrix elements A_q(F_l, m_l, F_u, m_u)
     between two hyperfine Zeeman states of an electric transition with quantum numbers j_l and j_u.
     See [Brown et al., Phys. Rev. A 87, 032504 (2013)].
    """
    i, j_l, f_l, m_l, j_u, f_u, m_u, q = cast_sympy(as_sympy, i, j_l, f_l, m_l, j_u, f_u, m_u, q)
    if m_u - m_l != q:
        return cast_sympy(as_sympy, 0)
    sqrt_f = sqrt(2 * f_l + 1) if as_sympy else np.sqrt(2 * f_l + 1)
    sqrt_j = sqrt(2 * j_u + 1) if as_sympy else np.sqrt(2 * j_u + 1)
    exp = cast_sympy(as_sympy, f_l + i + 1 + j_u, dtype=int)
    return (-1) ** exp * sqrt_f * sqrt_j * wigner_6j(j_u, j_l, 1, f_l, f_u, i, as_sympy) \
        * clebsch_gordan(f_l, 1, f_u, m_l, q, m_u, as_sympy)


def a_dipole_cart(i: sympy_qn, j_l: sympy_qn, f_l: sympy_qn, m_l: sympy_qn,
                  j_u: sympy_qn, f_u: sympy_qn, m_u: sympy_qn, as_sympy: bool = False) -> ndarray:
    """
    :param i: The nuclear spin quantum number I.
    :param j_l: The electronic total angular momentum quantum number J of the lower state.
    :param f_l: The total angular momentum quantum number F of the lower state.
    :param m_l: The z-projection quantum number m of the total angular moment of the lower state
    :param j_u: The electronic total angular momentum quantum number J of the upper state.
    :param f_u: The total angular momentum quantum number F of the upper state.
    :param m_u: The z-projection quantum number m of the total angular moment of the upper state.
    :param as_sympy: Whether to return the result as a sympy type.
    :returns: The cartesian reduced dipole vector.
    """
    i, j_l, f_l, m_l, j_u, f_u, m_u = cast_sympy(as_sympy, i, j_l, f_l, m_l, j_u, f_u, m_u)
    sqrt_2 = nsimplify(1 / sqrt(2)) if as_sympy else 1 / np.sqrt(2)
    x = sqrt_2 * (a_dipole(i, j_l, f_l, m_l, j_u, f_u, m_u, -1, as_sympy)
                  - a_dipole(i, j_l, f_l, m_l, j_u, f_u, m_u, 1, as_sympy))
    y = -1j * sqrt_2 * (a_dipole(i, j_l, f_l, m_l, j_u, f_u, m_u, -1, as_sympy)
                        + a_dipole(i, j_l, f_l, m_l, j_u, f_u, m_u, 1, as_sympy))
    z = a_dipole(i, j_l, f_l, m_l, j_u, f_u, m_u, 0, as_sympy)
    if as_sympy:
        a3 = CoordSys3D('a3')
        return x * a3.i + y * a3.j + z * a3.k
    return np.array([x, y, z], dtype=complex)


def a_dipole_cart_(i: sympy_qn, j_l: sympy_qn, f_l: sympy_qn, m_l: sympy_qn,
                 j_u: sympy_qn, f_u: sympy_qn, m_u: sympy_qn, as_sympy: bool = False) -> ndarray:
    """
    :param i: The nuclear spin quantum number I.
    :param j_l: The electronic total angular momentum quantum number J of the lower state.
    :param f_l: The total angular momentum quantum number F of the lower state.
    :param m_l: The z-projection quantum number m of the total angular moment of the lower state
    :param j_u: The electronic total angular momentum quantum number J of the upper state.
    :param f_u: The total angular momentum quantum number F of the upper state.
    :param m_u: The z-projection quantum number m of the total angular moment of the upper state.
    :param as_sympy: Whether to return the result as a sympy type.
    :returns: The cartesian reduced dipole vector.
    """
    i, j_l, f_l, m_l, j_u, f_u, m_u = cast_sympy(as_sympy, i, j_l, f_l, m_l, j_u, f_u, m_u)
    sqrt_2 = nsimplify(1 / sqrt(2)) if as_sympy else 1 / np.sqrt(2)
    x = sqrt_2 * (a_dipole(i, j_l, f_l, m_l, j_u, f_u, m_u, -1, as_sympy)
                  + a_dipole(i, j_l, f_l, m_l, j_u, f_u, m_u, 1, as_sympy))
    y = 1j * sqrt_2 * (a_dipole(i, j_l, f_l, m_l, j_u, f_u, m_u, 1, as_sympy)
                       - a_dipole(i, j_l, f_l, m_l, j_u, f_u, m_u, -1, as_sympy))
    z = a_dipole(i, j_l, f_l, m_l, j_u, f_u, m_u, 0, as_sympy)
    if as_sympy:
        a3 = CoordSys3D('a3')
        return x * a3.i + y * a3.j + z * a3.k
    return np.array([x, y, z], dtype=complex)


def reduced_f_root(i: sympy_qn, j_l: sympy_qn, f_l: sympy_qn, j_u: sympy_qn, f_u: sympy_qn, as_sympy: bool = False):
    """
    :param i: The nuclear spin quantum number I.
    :param j_l: The electronic total angular momentum quantum number J of the lower state.
    :param f_l: The total angular momentum quantum number F of the lower state.
    :param j_u: The electronic total angular momentum quantum number J of the upper state.
    :param f_u: The total angular momentum quantum number F of the upper state.
    :param as_sympy: Whether to return the result as a sympy type.
    :returns: The square root of the relative hyperfine transition strengths.
    """
    i, j_l, f_l, j_u, f_u = cast_sympy(as_sympy, i, j_l, f_l, j_u, f_u)
    sqrt_fl = sqrt(2 * f_l + 1) if as_sympy else np.sqrt(2 * f_l + 1)
    sqrt_fu = sqrt(2 * f_u + 1) if as_sympy else np.sqrt(2 * f_u + 1)
    exp = cast_sympy(as_sympy, f_l + i + 1 + j_u, dtype=int)
    # == (...) * wigner_6j(f_u, 1, f_l, j_l, i, j_u, as_sympy)
    return (-1) ** exp * sqrt_fl * sqrt_fu * wigner_6j(j_u, j_l, 1, f_l, f_u, i, as_sympy)


def reduced_f(i: sympy_qn, j_l: sympy_qn, f_l: sympy_qn, j_u: sympy_qn, f_u: sympy_qn, as_sympy: bool = False):
    """
    :param i: The nuclear spin quantum number I.
    :param j_l: The electronic total angular momentum quantum number J of the lower state.
    :param f_l: The total angular momentum quantum number F of the lower state.
    :param j_u: The electronic total angular momentum quantum number J of the upper state.
    :param f_u: The total angular momentum quantum number F of the upper state.
    :param as_sympy: Whether to return the result as a sympy type.
    :returns: The relative hyperfine transition strengths.
    """
    i, j_l, f_l, j_u, f_u = cast_sympy(as_sympy, i, j_l, f_l, j_u, f_u)
    return (2 * f_l + 1) * (2 * f_u + 1) * wigner_6j(j_u, j_l, 1, f_l, f_u, i, as_sympy) ** 2


def a_tilda(i: sympy_qn, j_l: sympy_qn, f_l: sympy_qn, j_u: sympy_qn, f_u: sympy_qn, as_sympy: bool = False):
    """
    :param i: The nuclear spin quantum number I.
    :param j_l: The electronic total angular momentum quantum number J of the lower state.
    :param j_u: The electronic total angular momentum quantum number J of the upper state.
    :param f_l: The total angular momentum quantum number F of the lower state.
    :param f_u: The total angular momentum quantum number F of the upper state.
    :param as_sympy: Whether to return the result as a sympy type.
    :returns: The relative transition strengths normed as in tilda.
    """
    i, j_l, f_l, j_u, f_u = cast_sympy(as_sympy, i, j_l, f_l, j_u, f_u)
    return (2 * f_l + 1) * (2 * f_u + 1) * wigner_6j(j_u, j_l, 1, f_l, f_u, i, as_sympy) ** 2


def a_m_tilda(i: sympy_qn, j_l: sympy_qn, f_l: sympy_qn, m_l: sympy_qn,
              j_u: sympy_qn, f_u: sympy_qn, m_u: sympy_qn, as_sympy: bool = False):
    """
    :param i: The nuclear spin quantum number I.
    :param j_l: The electronic total angular momentum quantum number J of the lower state.
    :param f_l: The total angular momentum quantum number F of the lower state.
    :param m_l: The z-projection quantum number m of the total angular moment of the lower state
    :param j_u: The electronic total angular momentum quantum number J of the upper state.
    :param f_u: The total angular momentum quantum number F of the upper state.
    :param m_u: The z-projection quantum number m of the total angular moment of the upper state
    :param as_sympy: Whether to return the result as a sympy type.
    :returns: The relative transition strengths between Zeeman substates normed as in tilda.
    """
    return a_tilda(i, j_l, f_l, j_u, f_u, as_sympy) * clebsch_gordan(f_l, 1, f_u, m_l, m_u - m_l, m_u, as_sympy) ** 2
