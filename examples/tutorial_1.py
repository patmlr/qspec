# -*- coding: utf-8 -*-
"""
examples.tutorial_1
===================

Tutorial 1 from the website: Getting started.
"""

import numpy as np
import qspec as qs

# This imports the analyze, algebra,
# physics, stats, and tools modules

q = 1  # (e), Ion charge state
m = 87.905612253 - q * qs.me_u  # (u) [1]
# Mass of 87Sr+

U = 20000  # (V), Acceleration voltage

# Resonance frequency from NIST database
f0 = qs.inv_cm_to_freq(24516.65)  # (MHz) [2]
# >>> 734990676.5 MHz
print(f'f0: {f0} MHz')

# Relativistic velocity of 88Sr+
v = qs.v_el(U, q, m)  # (m/s)
# >>> 209533.6 m/s
print(f'v: {v} MHz')

# The anti-collinear lab. frequency
f_laser = qs.doppler(f0, v, np.pi)  # (MHz)
# >>> 735504562.3 MHz
print(f'f_laser: {f_laser} MHz')
