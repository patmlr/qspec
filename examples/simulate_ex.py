# -*- coding: utf-8 -*-
"""
examples.simulate_ex

Created on 24.06.2021

@author: Patrick Mueller

Example script / Guide for the PyCLS.simulate module.
"""

import numpy as np
import scipy.constants as sc

import pycol.physics as ph
import pycol.algebra as al
import pycol.simulate as sim


"""
Example 1: Interaction between a singly-charged calcium ion and two lasers.

The simulate module provides functions to simulate the interaction between lasers and atoms.
In example 1, the master equation is solved for 40Ca+.
"""

f_sp = 755222766  # The 4s -> 4p 2P1/2 transition frequency.
f_dp = 346000235  # The 3d 2D3/2 -> 4p 2P1/2 transition frequency.

a_sp = 140  # The Einstein coefficients of the two transitions
a_dp = 10.7

s = sim.construct_electronic_state(freq_0=0, s=0.5, l=0, j=0.5, label='s')  # A list of all 4s sub-states.
p = sim.construct_electronic_state(f_sp, 0.5, 1, 0.5, label='p')  # A list of all 4p 2P1/2 sub-states.
d = sim.construct_electronic_state(f_sp - f_dp, 0.5, 2, 1.5, label='d')  # A list of all 3d 2D3/2 sub-states.

decay = sim.DecayMap(labels=[('s', 'p'), ('p', 'd')], a=[a_sp, a_dp])
# The states are linked by Einstein-A coefficients via the specified labels.

ca40 = sim.Atom(states=s + p + d, decay_map=decay)  # The Atom with all states and the decay information.

pol = sim.Polarization([1, 1, 1], q_axis=2)
laser_sp = sim.Laser(freq=f_sp, polarization=pol, intensity=500)  # Linear polarized laser for the two
laser_dp = sim.Laser(freq=f_dp, polarization=pol, intensity=500)  # transitions, with 500 uW / mm**2.

inter = sim.Interaction(atom=ca40, lasers=[laser_sp, laser_dp])  # The interaction.
# inter.resonance_info()  # Print the detuning of the lasers from the considered transitions.
inter.environment = sim.Environment(B=[0, 0, 0.01])

inter.master(t=0.5)  # Solve the master equation for 0.5 us, assuming equal population in all s-states.
# (Here the s-states are the first in the list and they all have the same label.)
# inter.master(t=0.5, dissipation=False)  # Without spontaneous emission.
quit()
"""
Example 2: Interaction between a singly-charged calcium ion and two lasers.

In example 2, the master equation is solved for 43Ca+.
"""

# Hyperfine-structure constants [A, B]
s_hyper = [-806.4, ]
p1_hyper = [-145.6, ]
p3_hyper = [-31., -6.9]
d3_hyper = [-47.3, -3.7]

s = sim.construct_electronic_state(freq_0=0, s=0.5, l=0, j=0.5, i=3.5, hyper_const=s_hyper, label='s')
p = sim.construct_electronic_state(f_sp, 0.5, 1, 0.5, i=3.5, hyper_const=p1_hyper, label='p')
d = sim.construct_electronic_state(f_sp - f_dp, 0.5, 2, 1.5, i=3.5, hyper_const=d3_hyper, label='d')

states = s + p + d
ca43 = sim.Atom(states=states, decay_map=decay)  # The Atom with all states and the decay information from above.

pol_sp = sim.Polarization([1, 1, 1], q_axis=2)
pol_dp = sim.Polarization([0, 1, 0], q_axis=2)
laser_sp = sim.Laser(freq=f_sp - 1487, polarization=pol_sp, intensity=500)
laser_dp = sim.Laser(freq=f_dp - 172, polarization=pol_dp, intensity=500)

inter = sim.Interaction(atom=ca43, lasers=[laser_sp, laser_dp], delta_max=400.)  # Laser excitations with transitions
# inter.resonance_info()  # Print the detuning of the lasers from the considered transitions.

# off by less than 400 MHz are considered.
inter.controlled = True  # Use an error controlled solver to deal with fast dynamics.
# inter.dt = 1e-4  # Alternatively, decrease the step size.

inter.master(t=0.5)  # Solve the master equation for 0.5 us, assuming equal population in all s-states.
# (Here the s-states are the first in the list and they all have the same label.)


"""
Example 3: Interaction between a singly-charged lithium ion and two lasers.

In example 3 Fig. 5 from [Noertershaeuser et al. Phys. Rev. Accel. Beams 24, 024701 (2021),
https://doi.org/10.1103/PhysRevAccelBeams.24.024701] is calculated.
"""

f = (494263.44 - 476034.98) * sc.c * 1e-4  # sc.c / 548.5 * 1e-3  # 3S1 -> 3P2 (MHz)
a = 22.727
df_s = 19.8e3  # frequency splitting between the two s-states.
df_p = 11.8e3  # frequency splitting between two p-states.

states = sim.construct_hyperfine_state(freq_0=0, s=1, l=0, j=1, i=1.5, f=1.5, hyper_const=[df_s / 2.5, ], label='s3')
states += sim.construct_hyperfine_state(0, 1, 0, 1, 1.5, 2.5, [df_s / 2.5, ], label='s5')
states += sim.construct_hyperfine_state(f, 1, 1, 2, 1.5, 2.5, [df_p / 3.5, ], label='p')

decay = sim.DecayMap(labels=[('s3', 'p'), ('s5', 'p')], a=[a, a])

li7 = sim.Atom(states=states, decay_map=decay)  # The Atom with all states and the decay information.

i_b = 200  # Intensity of the blue laser (uW / mm ** 2)
i_r = 2000  # Intensity of the red laser (uW / mm ** 2)

pol_b = sim.Polarization([0, 1, 0], q_axis=2)
pol_r = sim.Polarization([0, 1, 0], q_axis=2)
laser_b = sim.Laser(freq=f + 6234.29, polarization=pol_b, intensity=i_b)  # blue laser
laser_r = sim.Laser(freq=f - 13566., polarization=pol_r, intensity=i_r)  # red laser (with detunings)

print('Saturation s(blue): {}'.format(ph.saturation(i_b, f, a, al.a(1.5, 1, 1.5, 2, 2.5))))
print('Saturation s(red): {}'.format(ph.saturation(i_r, f, a, al.a(1.5, 1, 2.5, 2, 2.5))))
# The saturation intensity can be compared easily to the specified values in the paper.

inter = sim.Interaction(atom=li7, lasers=[laser_b, laser_r])
# inter.resonance_info()

delta = np.linspace(-2., 2., 11)
y0 = li7.get_y0(['s3', 's5'])

# Solve for the first 0.1 ms (These are 100 000 time steps times 11 detunings)
# inter.spectrum(100, delta, m=1, y0=y0, solver='master', x_scale='log')  # with the master equation ...
# inter.spectrum(100, delta, m=1, y0=y0, solver='rates', x_scale='log')  # ... and the rate equations.
