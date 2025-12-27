"""
qspec.analyze
=============

Module for analyzing/evaluating/fitting data.

Linear regression algorithms (2d):
    - york_fit(); [York et al., Am. J. Phys. 72, 367 (2004)]
    - linear_fit(); 2-dimensional maximum likelihood fit.
    - linear_monte_carlo(); based on [Gebert et al., Phys. Rev. Lett. 115, 053003 (2015), Suppl.]

Linear regression algorithms (nd):
    - linear_nd_fit(); n-dimensional maximum likelihood fit.
    - linear_monte_carlo_nd(); based on [Gebert et al., Phys. Rev. Lett. 115, 053003 (2015), Suppl.]

Curve fitting methods:
    - curve_fit(); Reimplements the scipy.optimize.curve_fit method to allow fixing parameters
      and having parameter-dependent y-uncertainties.
    - odr_fit(); Encapsulates the scipy.odr.odr method to accept inputs similarly to curve_fit().

Classes:
    - King; Creates a King plot with isotope shifts or nuclear charge radii.

LICENSE NOTES:
    The method curve_fit is a modified version of scipy.optimize.curve_fit.
    Therefore, it is licensed under the 'BSD 3-Clause New or Revised License' provided with scipy.
"""

from qspec.analyze import _analyze, _analyze_cpp
from qspec.analyze._analyze import (
    King,
    const,
    covariance_matrix,
    curve_fit,
    draw_sigma2d,
    draw_straight_unc_area,
    ellipse2d,
    generate_collinear_points_py,
    linear_alpha_fit,
    linear_fit,
    linear_monte_carlo,
    linear_nd_fit,
    linear_nd_monte_carlo,
    odr_fit,
    poly,
    straight,
    straight_direction,
    straight_std,
    straight_x_std,
    weight,
    york_fit,
)
from qspec.analyze._analyze_cpp import generate_collinear_points_cpp

__all__ = []
__all__.extend(_analyze_cpp.__all__)
__all__.extend(_analyze.__all__)
