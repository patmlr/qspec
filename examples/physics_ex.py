# -*- coding: utf-8 -*-
"""
examples.physics_ex

Created on 12.05.2021

@author: Patrick Mueller

Example script / Guide for the PyCLS.physics module.
"""

import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import pycol.physics as ph
import pycol.tools as tools


"""
Example 1: Calculation of energies, velocities and Doppler shifts.

The physics module provides functions to calculate basic observables.
In example 1 basic properties of charged carbon isotopes are calculated.
"""

m = 12  # The mass of the most abundant stable carbon isotope (u).
q = 4  # The charge of the ions (e).
f0 = (2455169.9 - 2411271.2) * sc.c * 1e-4  # The 3S1 -> 3P0 transition of C4+.

# a) Typical COALA calculation.
U = 20e3  # The acceleration voltage (V).
e_el = ph.e_el(U, q)  # The potential energy difference corresponding to the voltage (eV)

v = ph.v_el(U, q, m, v0=0, relativistic=True)  # The velocity of the accelerated carbon ions (m/s).
gamma = ph.gamma(v)  # The Lorentzian time-dilation factor.

f_col = ph.doppler(f0, v, alpha=0, return_frame='lab')  # The resonant laser frequency in collinear direction (MHz).
f_acol = ph.doppler(f0, v, alpha=np.pi, return_frame='lab')  # " in anticollinear direction (MHz).
f_div_col = ph.doppler_el_d1(f_col, 0, U, q, m, 0, return_frame='atom')
# The differential Doppler factor in the atoms rest frame (MHz / V).
# == The change of the laser frequency in the atoms rest frame with the voltage.
f_div_acol = ph.doppler_el_d1(f_acol, np.pi, U, q, m, 0, return_frame='atom')

# b) Typical CRYRING calculation.
f_col = 340e6 * 4  # The resonant laser frequency in collinear direction (MHz).
v = ph.inverse_doppler(f0, f_col, alpha=0)  # The velocity of the carbon ions (m/s).
beta = ph.beta(v)  # The relative velocity of the carbon ions.
e_kin = ph.e_kin(v, m, relativistic=True)  # The kinetic energy of the carbon ions (eV).


"""
Example 2: Calculation of Doppler shifts for an ensemble of ion-velocity vectors.

Most of the functions in the physics module are array compatible. In the following ions from example 1a) are simulated.
"""

n = 100000  # The number of samples.

f_col = 1321038455  # Resonant collinear laser frequency at 20kV.

v_vec = ph.thermal_v_rvs(m, 2500, (n, 3))  # Generate 100,000 thermally distributed velocity values for each direction.
v_vec[:, 0] = ph.v_el(U, q, m, v_vec[:, 0], relativistic=True)  # Offset the kinetic energy by 20 kV =1a)= 80 keV
# in the x direction.
gamma3d = np.mean(ph.gamma_3d(v_vec, axis=-1), axis=0)  # The mean Lorentzian time-dilation factor for the 3d vectors.

f_vec = np.array([[f_col, f_col, 0, 0], ])  # The collinear frequency 4-vector.
# Note that it was expanded along axis 0 to match the shape of v_vec.
f_vec_atom = ph.boost(f_vec, v_vec, axis=-1)  # The Lorentz boosts of the frequency 4-vector.
# Note that the vector components are aligned with the last axis.
f_atom = tools.absolute(f_vec_atom[:, 1:], axis=-1)  # The length of the cartesian components of the boosted 4-vectors.

num_test = f_atom - f_vec_atom[:, 0]  # That length must be equal to the zeroth component of the 4-vectors.
# plt.plot(num_test / f_vec_atom[:, 0])  # Plot relative numeric variation.
# plt.show()

plt.xlabel('Laser freq. in the ion system relative to the resonance (MHz)')
plt.ylabel('Abundance')
plt.hist((f_atom - f0), bins=200)
plt.show()

plt.xlabel('Laser freq. in the ion system relative to its x-component (MHz)')
plt.ylabel('Abundance')
plt.hist(f_atom - f_vec_atom[:, 1], bins=200)
plt.show()
