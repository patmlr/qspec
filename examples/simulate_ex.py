# -*- coding: utf-8 -*-
"""
examples.simulate_ex

Created on 24.06.2021

@author: Patrick Mueller

Example script / Guide for the pycol.simulate module.
"""

import numpy as np
import scipy.constants as sc

from pycol.types import *
import pycol.physics as ph
import pycol.algebra as al
import pycol.simulate as sim


def example(n: Union[set, int] = None):
    """
    Run one or several of the available examples. Scroll to the end for the function call.

    :param n: The number of the example or a list of numbers.

    Example 0: Interaction between 40Ca+ and a laser.

    Example 1: Interaction between 40Ca+ and two lasers off-resonance / Rabi pumping.

    Example 2: Interaction between 43Ca+ with hyperfine structure and a laser.

    Example 3: Interaction between a singly-charged lithium ion and two lasers.

    :returns: None.
    """
    if n is None:
        n = {0, 1, 2}
    if isinstance(n, int):
        n = {n, }

    if 0 in n:
        """
        Example 0: Interaction between 40Ca+ and a laser.
        
        The simulate module provides functions to simulate the interaction between lasers and atoms.
        In example 0, the master equation is solved for 40Ca+.
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

        pol = sim.Polarization([0, 1, 0], q_axis=2, vec_as_q=True)
        laser_sp = sim.Laser(freq=f_sp, polarization=pol, intensity=500)  # Linear polarized laser for the ground-state
        # transition with 500 uW / mm**2.

        inter = sim.Interaction(atom=ca40, lasers=[laser_sp, ])  # The interaction.
        inter.resonance_info()  # Print the detuning of the lasers from the considered transitions.

        t = 0.5  # Integration time in us.

        inter.rates(t=t)  # Solve the rate equations for t, assuming equal population in all s-states.
        inter.master(t=t)  # Solve the master equation for t, assuming equal population in all s-states.
        # inter.master(t=t, dissipation=False)  # Without spontaneous emission.

        # Calculate the integrated population for each state after t.
        inter.spectrum(t, np.linspace(-100, 100, 401), solver='rates')
        # inter.spectrum(t, np.linspace(-100, 100, 401), solver='master')  # Use the master equation solver.

    if 1 in n:
        """
        Example 1: Interaction between 40Ca+ and two lasers off-resonance,
        leading to Rabi oscillations between s and d state without going through p.
        
        In example 1, the master equation is solved for 40Ca+ with two lasers off-resonance.
        Same atomic system as in example 1.
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

        pol = sim.Polarization([0, 1, 0], q_axis=2)
        laser_sp = sim.Laser(freq=f_sp + 5000, polarization=pol, intensity=10000)  # Linear polarized laser for
        # the ground-state transition.
        laser_dp = sim.Laser(freq=f_dp + 5000, polarization=pol, intensity=10000)  # Linear polarized laser for
        # the metastable-state transition.

        inter = sim.Interaction(atom=ca40, lasers=[laser_sp, laser_dp], delta_max=10000)
        inter.controlled = True  # Use the controlled solver.
        # inter.dt = 4e-5  # or small step sizes.

        t = 4.  # Integration time in us.
        delta = np.linspace(-1.5, 1.5, 101)

        # inter.rates(t=t)  # Solve the rate equations for t, assuming equal population in all s-states.
        inter.master(t=t)  # Solve the master equation for t, assuming equal population in all s-states.

        spec = inter.spectrum(4., delta=delta, solver='master', show=False)  # Suppress the plot here ...
        spec.plot([2.5, 4])  # ... to plot only the integrated population of the desired interval.

    if 2 in n:
        """
        Example 2: Interaction between 43Ca+ with hyperfine structure and a laser.
        
        In example 2, the master equation is solved for 43Ca+.
        """
        # Frequencies (40Ca+) taken from [P. Mueller et al., Phys. Rev. Research 2, 043351 (2020),
        # https://doi.org/10.1103/PhysRevResearch.2.043351].
        # f_sp1 = 755222766  # The 4s -> 4p 2P1/2 transition frequency.
        f_sp3 = 761905013  # The 4s -> 4p 2P3/2 transition frequency.
        # f_d3p1 = 346000235  # The 3d 2D3/2 -> 4p 2P1/2 transition frequency.
        f_d3p3 = 352682482  # The 3d 2D3/2 -> 4p 2P3/2 transition frequency.
        f_d5p3 = 350862883  # The 3d 2D5/2 -> 4p 2P3/2 transition frequency.

        # Einstein coefficients taken from [NIST Atomic Spectra Database, https://doi.org/10.18434/T4W30F].
        # a_sp1 = 140  # The Einstein coefficients of the two transitions
        a_sp3 = 147  # The Einstein coefficients of the two transitions
        # a_d3p1 = 10.7
        a_d3p3 = 1.11
        a_d5p3 = 9.9

        # Hyperfine-structure constants [A, B], taken from [Noertershaeuser et al., Eur. Phys. J. D 2, 33–39 (1998),
        # https://doi.org/10.1007/s100530050107]
        s_hyper = [-806.4, ]
        # p1_hyper = [-145.6, ]
        p3_hyper = [-31., -6.9]
        d3_hyper = [-47.3, -3.7]
        d5_hyper = [-3.8, -3.9]

        # Create only the states for the D2 transition.
        s = sim.construct_electronic_state(freq_0=0, s=0.5, l=0, j=0.5, i=3.5, hyper_const=s_hyper, label='s')
        p3 = sim.construct_electronic_state(f_sp3, 0.5, 1, 1.5, i=3.5, hyper_const=p3_hyper, label='p3')
        d3 = sim.construct_electronic_state(f_sp3 - f_d3p3, 0.5, 2, 1.5, i=3.5, hyper_const=d3_hyper, label='d3')
        d5 = sim.construct_electronic_state(f_sp3 - f_d5p3, 0.5, 2, 2.5, i=3.5, hyper_const=d5_hyper, label='d5')

        decay = sim.DecayMap(labels=[('s', 'p3'), ('p3', 'd3'), ('p3', 'd5')], a=[a_sp3, a_d3p3, a_d5p3])
        # The states are linked by Einstein-A coefficients via the specified labels.

        states = s + p3 + d3 + d5
        ca43 = sim.Atom(states=states, decay_map=decay)
        ca43.plot()  # Plot the involved states.
        # The Atom with all states and the decay information from above.

        pol_sp = sim.Polarization([0, 1, 0], q_axis=2)
        laser_sp = sim.Laser(freq=f_sp3 - 1697, polarization=pol_sp, intensity=500)

        inter = sim.Interaction(atom=ca43, lasers=[laser_sp, ], delta_max=400.)
        # Set delta_max (MHz) to do a RWA in the off-resonance transitions.

        inter.resonance_info()  # Print the detunings of the lasers from the considered transitions.

        inter.controlled = True  # Use an error controlled solver to deal with fast dynamics.
        # inter.dt = 1e-4  # Alternatively, decrease the step size.

        t = 0.2  # Integration time in us.
        delta = np.linspace(-180, 150, 331)

        inter.rates(t=t)  # Solve the rate equations for t, assuming equal population in all s-states.
        # inter.master(t=t)  # Solve the master equation for t, assuming equal population in all s-states.

        # inter.spectrum(t, delta=delta, solver='rates')
        inter.spectrum(t, delta=delta, solver='master')

    if 3 in n:
        """
        Example 3: Interaction between a singly-charged lithium ion and two lasers.
        
        In example 3 Fig. 5 from [Noertershaeuser et al. Phys. Rev. Accel. Beams 24, 024701 (2021),
        https://doi.org/10.1103/PhysRevAccelBeams.24.024701] is calculated.
        """

        f = (494263.44 - 476034.98) * sc.c * 1e-4  # sc.c / 548.5 * 1e-3  # 3S1 -> 3P2 (MHz)
        a = 22.727
        df_s = 19.8e3  # frequency splitting between the two s-states.
        df_p = 11.8e3  # frequency splitting between two p-states.

        states = sim.construct_hyperfine_state(freq_0=0, s=1, l=0, j=1, i=1.5, f=1.5,
                                               hyper_const=[df_s / 2.5, ], label='s3')
        states += sim.construct_hyperfine_state(0, 1, 0, 1, 1.5, 2.5, [df_s / 2.5, ], label='s5')
        states += sim.construct_hyperfine_state(f, 1, 1, 2, 1.5, 2.5, [df_p / 3.5, ], label='p')

        decay = sim.DecayMap(labels=[('s3', 'p'), ('s5', 'p')], a=[a, a])

        li7 = sim.Atom(states=states, decay_map=decay)  # The Atom with all states and the decay information.
        li7.plot()  # Plot the involved states.

        i_b = 200  # Intensity of the blue laser (uW / mm ** 2)
        i_r = 2000  # Intensity of the red laser (uW / mm ** 2)

        pol_b = sim.Polarization([0, 1, 0], q_axis=2)  # sigma+ polarization
        pol_r = sim.Polarization([0, 1, 0], q_axis=2)  # sigma+ polarization
        laser_b = sim.Laser(freq=f + 6234.29, polarization=pol_b, intensity=i_b)  # blue laser
        laser_r = sim.Laser(freq=f - 13566., polarization=pol_r, intensity=i_r)  # red laser

        print('Saturation s(blue): {}'.format(ph.saturation(i_b, f, a, al.a(1.5, 1, 1.5, 2, 2.5))))
        print('Saturation s(red): {}'.format(ph.saturation(i_r, f, a, al.a(1.5, 1, 2.5, 2, 2.5))))
        # The saturation intensity can be compared easily to the specified values in the paper.

        inter = sim.Interaction(atom=li7, lasers=[laser_b, laser_r])
        inter.resonance_info()  # Print the resonance info.

        t = 10.  # Integration time in us.
        delta = np.linspace(-50., 50., 201)
        y0 = li7.get_y0(['s3', 's5'])

        inter.rates(t=t, y0=y0, x_scale='log')  # Solve the master equation for t and plot with logarithmic scaling.
        # inter.master(t=t, y0=y0, x_scale='log')  # Solve the master equation for t and plot with logarithmic scaling.

        # Solve for the first 10 us (these are 10 000 time steps times 201 detunings)
        inter.spectrum(t, delta, m=0, y0=y0, solver='rates')  # with the rate equations ...
        # inter.spectrum(t, delta, m=0, y0=y0, solver='master')  # ... and the master equation.


if __name__ == '__main__':
    example({0})
