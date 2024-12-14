# -*- coding: utf-8 -*-
"""
examples.tutorial_5
===================

Tutorial 5 from the website: King plot analysis of Ca+ isotopes.
"""

import qspec as qs

# The mass numbers of the Ca isotopes.
a = [40, 42, 43, 44, 46, 48, 50, 52]

# The masses of the isotopes (AME 2020).
m = [(39.962590850, 22e-9), (41.958617780, 159e-9),  # 40Ca, 42Ca
     (42.958766381, 244e-9), (43.955481489, 348e-9),  # 43Ca, 44Ca
     (45.953687726, 2398e-9), (47.952522654, 18e-9),  # 46Ca, 48Ca
     (49.957499215, 1.7e-6), (51.963213646, 720e-9)]  # 50Ca, 52Ca

# Use absolute values given in the shape (#isotopes, #observables, 2).
# Frequencies for (D1, D2) line.
x_abs = [[(755222765.66, 0.10), (761905012.53, 0.11)],  # 40Ca
         [(755223191.15, 0.10), (761905438.57, 0.10)],  # 42Ca
         [(755223443.57, 0.30), (761905691.89, 0.17)],  # 43Ca
         [(755223614.66, 0.10), (761905862.62, 0.09)],  # 44Ca
         [(755224063.27, 0.33), (761906311.60, 0.57)],  # 46Ca
         [(755224471.12, 0.10), (761906720.11, 0.11)],  # 48Ca
         [(        0.  , 0.  ), (        0.  , 0.  )],  # 50Ca
         [(        0.  , 0.  ), (        0.  , 0.  )]]  # 52Ca

# Construct a King object. Optionally specify 'x_abs' here
# to omit isotope shifts when fitting. 20 electron masses are subtracted
# to perform the King plot analysis with the nuclear masses.
king = qs.King(a=a, m=m, x_abs=x_abs, subtract_electrons=20)

a_fit = [42, 43, 44, 46, 48]  # Choose the isotopes to fit.
a_ref = [40, 48, 42, 40, 44]  # Choose individual reference isotopes.

# Do a simple 2d King plot. The 'mode' keyword is only used for the axis labels.
popt, pcov = king.fit(a_fit, a_ref, mode='shifts')
# >>> f(x) = (177.3 u MHz) + 1.00068 * x

a_unknown = [50, 52]  # Specify the unknown isotopes
a_unknown_ref = [40, 40]  # and their references.

# Specify the isotope shifts of the D2 line.
y = [(1969.2, 5.6), (2219.2, 7.0)]

# Calculate the isotope shifts of the D1 line and their covariances.
x, cov, cov_stat = king.get_unmodified(
    a_unknown, a_unknown_ref, y, axis=1, show=True, mode='shifts')

for iso, c in zip(a_unknown, cov):
    qs.printh(f'\n{iso}Ca+:')  # Print colored headline.
    qs.print_cov(c)  # Print color-coded covariance matrix.
