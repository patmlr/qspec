# -*- coding: utf-8 -*-
"""
examples.tutorial_5
===================

Tutorial 5 from the website: Time evolution of quantum interference in fluorescence spectra.
"""

import numpy as np
import qspec.simulate as sim

f_sp = 446810183.163  # Transition frequency (MHz)
a_sp = 36.891  # Einstein coefficient (rad MHz)

s_hyper = [401.75825]  # HFS constants (MHz)
p_hyper = [-3.055038, -0.29670]

s = sim.gen_electronic_ls_state(
    0., s=0.5, l=0, j=0.5, i=1.5, hyper_const=s_hyper, label='s')
p = sim.gen_electronic_ls_state(
    f_sp, s=0.5, l=1, j=1.5, i=1.5, hyper_const=p_hyper, label='p')

decay = sim.DecayMap(labels=[('s', 'p')], a=[a_sp])
li7 = sim.Atom(s + p, decay)

intensity = 1.  # uW /mm**2
polarization = sim.Polarization([0, 1, 0])  # Linear polarization
laser = sim.Laser(f_sp, intensity, polarization)

inter = sim.Interaction(li7, [laser])
inter.controlled = True  # Error controlled integrator

t = 0.2  # Integration time (us)
delta = np.linspace(-325, -275, 201)  # Frequency detunings (MHz)
theta, phi = 0., 0.  # Angles from z-axis in x- and y-direction (rad)

n = inter.rates(t, delta)  # Rate equations, 0.2 us
y_rates = li7.scattering_rate(n, as_density_matrix=False,
                              theta=theta, phi=phi)
y_rates = y_rates[0, :, -1]  # Remove angle and time axes

rho = inter.master(t, delta)  # Master equation, 0.2 us
y_master = li7.scattering_rate(rho, theta=theta, phi=phi)
y_master = y_master[0, :, -1]

rho = inter.master(0.4, delta)  # Master equation, 0.4 us
y4_master = li7.scattering_rate(rho, theta=theta, phi=phi)
y4_master = y4_master[0, :, -1]

sr = sim.ScatteringRate(li7, laser=laser)
y_brown = sr.generate_y(delta, theta, phi)  # Brown et al.
y_brown = y_brown[:, 0, 0]  # Remove (theta, phi) axes

import matplotlib.pyplot as plt

scale = 1e3
x_lim = delta[0], delta[-1]

fig, (m, r) = plt.subplots(
    2, 1, sharex='all', height_ratios=[3, 1], figsize=(6, 5))

m.plot(delta, y_brown * scale, '-k',
       label=r'Brown $et\,al.$', zorder=20)
m.plot(delta, y_rates * scale, '-C0',
       label=r'rates, $t = 0.2\,\mu$s', zorder=0)
m.plot(delta, y_master * scale, '-C1',
       label=r'master, $t = 0.2\,\mu$s', zorder=60)
m.plot(delta, y4_master * scale, '--C3',
       label=r'master, $t = 0.4\,\mu$s', linewidth=1.5, zorder=30)

r.plot(delta, (y_brown - y_rates) * scale,
       '-k', zorder=20)
r.plot(delta, (y_master - y_rates) * scale,
       '-C1', zorder=60)
r.plot(delta, (y4_master - y_rates) * scale,
       '--C3', linewidth=1.5, zorder=10)
r.hlines(0, *x_lim, 'C0', '-', zorder=0)

m.legend()
m.set_ylabel(r'$\mathrm{d}\Gamma / \mathrm{d}\Omega$ (kHz)')
m.set_xlim(*x_lim)
r.set_xlabel('Relative frequency (MHz)')
r.set_ylabel('Residuals (kHz)')

y_lim = m.get_ylim()
r.set_ylim(-(y_lim[1] - y_lim[0]) / 6, (y_lim[1] - y_lim[0]) / 6)

plt.subplots_adjust(left=0.09, bottom=0.1, right=0.99, top=0.99, hspace=0.05)
plt.show()
