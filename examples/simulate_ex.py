# -*- coding: utf-8 -*-
"""
examples.simulate_ex

Created on 24.06.2021

@author: Patrick Mueller

Example script / Guide for the qspec.simulate module.


- Einstein coefficients taken from [NIST Atomic Spectra Database, https://doi.org/10.18434/T4W30F].
- Frequencies of 40Ca+ taken from [P. Mueller et pc., Phys. Rev. Research 2, 043351 (2020),
                                  https://doi.org/10.1103/PhysRevResearch.2.043351].
- Hyperfine-structure constants [A, B] of 40Ca+ taken from [Noertershaeuser et pc., Eur. Phys. J. D 2, 33–39 (1998),
                                                            https://doi.org/10.1007/s100530050107]
"""

import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt

import qspec as qs
import qspec.simulate as sim
import qspec.models as mod


def example(n=None):
    """
    Run one or several of the available examples. Scroll to the end for the function call.

    :param n: The number of the example or a list of numbers.

    Example 0: Interaction between 40Ca+ and a laser.

    Example 1: Interaction between 40Ca+ and two lasers off-resonance / Rabi pumping.

    Example 2: Interaction between 43Ca+ with hyperfine structure and a laser.

    Example 3: Interaction between a singly-charged lithium-ion and two lasers.

    :returns: None.
    """
    if n is None:
        n = {0, 1, 2, 3, 4, 5, 6}
    if isinstance(n, int):
        n = {n, }

    if 0 in n:
        """
        Example 0: Interaction between 40Ca+ and a laser.
        
        The simulate module provides functions to simulate the interaction between lasers and atoms.
        In example 0, the rate (master) equation is solved for different laser detunings in 40Ca+.
        """
        f_sp = 755222766  # The 4s -> 4p 2P1/2 transition frequency.
        f_dp = 346000235  # The 3d 2D3/2 -> 4p 2P1/2 transition frequency.

        a_sp = 140  # The Einstein coefficients of the two transitions
        a_dp = 10.7

        s = sim.construct_electronic_state(freq_0=0, s=0.5, l=0, j=0.5, label='s')  # A list of all 4s substates.
        p = sim.construct_electronic_state(f_sp, 0.5, 1, 0.5, label='p')  # A list of all 4p 2P1/2 substates.
        d = sim.construct_electronic_state(f_sp - f_dp, 0.5, 2, 1.5, label='d')  # A list of all 3d 2D3/2 substates.

        decay = sim.DecayMap(labels=[('s', 'p'), ('p', 'd')], a=[a_sp, a_dp])
        # The states are linked by Einstein-A coefficients via the specified labels.

        ca40 = sim.Atom(states=s + p + d, decay_map=decay)  # The Atom with all states and the decay information.

        pol = sim.Polarization([0, 1, 0], q_axis=2, vec_as_q=True)
        laser_sp = sim.Laser(freq=f_sp, polarization=pol, intensity=500)  # Linear polarized laser for the ground-state
        # transition with 500 uW / mm**2.

        inter = sim.Interaction(atom=ca40, lasers=[laser_sp, ])  # The interaction.
        inter.resonance_info()  # Print the detuning of the lasers from the considered transitions.

        times = np.linspace(0., 0.5, 1001)  # Integration times.
        delta = np.linspace(-50, 50, 101)  # Laser detunings from the given laser frequency.

        results = inter.rates(times, delta=delta)  # rate equation

        # results = inter.master(times, delta=delta)  # master equation
        # results = np.transpose(np.diagonal(results, axis1=1, axis2=2).real, axes=[0, 2, 1])

        for res in results:
            i = ca40.get_state_indexes('p')
            plt.plot(times, np.sum(res[i], axis=0))
        plt.xlabel('time (us)')
        plt.ylabel('p-state populations')
        plt.show()

        i_t = -1
        for s in ['s', 'p', 'd']:  # Plot all fine-structure states.
            i = ca40.get_state_indexes(s)
            plt.plot(delta, np.sum(results[:, i, i_t], axis=1), label=s)
        plt.ylim(0, 1)
        plt.xlabel('f - {} (MHz)'.format(laser_sp.freq))
        plt.ylabel('state population after {} us'.format(times[i_t]))
        plt.legend()
        plt.show()

    if 1 in n:
        """
        Example 1: Interaction between 40Ca+ and two lasers off-resonance,
        leading to Rabi oscillations between s and d state without going through p.
        The simulation also shows the ac-Stark shift
        
        In example 1, the master equation is solved for 40Ca+ with two lasers off-resonance.
        Same atomic system as in example 1.
        """
        f_sp = 755222766  # The 4s -> 4p 2P1/2 transition frequency.
        f_dp = 346000235  # The 3d 2D3/2 -> 4p 2P1/2 transition frequency.

        a_sp = 140  # The Einstein coefficients of the two transitions
        a_dp = 10.7

        s = sim.construct_electronic_state(freq_0=0, s=0.5, l=0, j=0.5, label='s')  # A list of all 4s substates.
        p = sim.construct_electronic_state(f_sp, 0.5, 1, 0.5, label='p')  # A list of all 4p 2P1/2 substates.
        d = sim.construct_electronic_state(f_sp - f_dp, 0.5, 2, 1.5, label='d')  # A list of all 3d 2D3/2 substates.

        decay = sim.DecayMap(labels=[('s', 'p'), ('p', 'd')], a=[a_sp, a_dp])
        # The states are linked by Einstein-A coefficients via the specified labels.

        ca40 = sim.Atom(states=s + p + d, decay_map=decay)  # The Atom with all states and the decay information.

        pol = sim.Polarization([0, 1, 0], q_axis=2)
        laser_sp = sim.Laser(freq=f_sp + 1000, polarization=pol, intensity=10000)  # Linear polarized laser for
        # the ground-state transition.
        laser_dp = sim.Laser(freq=f_dp + 1000, polarization=pol, intensity=100)  # Linear polarized laser for
        # the metastable-state transition.

        inter = sim.Interaction(atom=ca40, lasers=[laser_sp, laser_dp], delta_max=10000)
        inter.controlled = True  # Use the controlled solver.
        # inter.dt = 4e-5  # or small step sizes.

        times = [0, 4]  # Integration time in us.
        delta = np.linspace(-1.5, 1.5, 101)

        results = inter.master(times, delta, m=1)  # m=0 for delta in first laser.
        # Solve the master equation for t, assuming equal population in all s-states.

        print('Shape of the resulting density-matrices object: ', results.shape)
        y = np.diagonal(results[:, :, :, -1], axis1=1, axis2=2).real
        print('Reshaped to (#delta, #states): ', y.shape)

        for s in ['s', 'p', 'd']:  # Plot all fine-structure states.
            i = ca40.get_state_indexes(s)
            plt.plot(delta, np.sum(y[:, i], axis=1), label=s)
        plt.xlabel('f - {} (MHz)'.format(laser_sp.freq))
        plt.ylabel('state population after {} us'.format(times[-1]))
        plt.legend()
        plt.show()

    if 2 in n:
        """
        Example 2: Interaction between 43Ca+ with hyperfine structure and a laser.
        
        In example 2, the rate (master) equation is solved for 43Ca+.
        """
        # f_sp1 = 755222766  # The 4s -> 4p 2P1/2 transition frequency.
        f_sp3 = 761905013  # The 4s -> 4p 2P3/2 transition frequency.
        # f_d3p1 = 346000235  # The 3d 2D3/2 -> 4p 2P1/2 transition frequency.
        f_d3p3 = 352682482  # The 3d 2D3/2 -> 4p 2P3/2 transition frequency.
        f_d5p3 = 350862883  # The 3d 2D5/2 -> 4p 2P3/2 transition frequency.

        # The Einstein coefficients of the transitions.
        # a_sp1 = 140
        a_sp3 = 147
        # a_d3p1 = 10.7
        a_d3p3 = 1.11
        a_d5p3 = 9.9

        # The hyperfine-structure constants of the states.
        s_hyper = [-806.4, ]
        # p1_hyper = [-145.6, ]
        p3_hyper = [-31., -6.9]
        d3_hyper = [-47.3, -3.7]
        d5_hyper = [-3.8, -3.9]

        i = 3.5

        # Create only the states for the D2 transition.
        s = sim.construct_electronic_state(freq_0=0, s=0.5, l=0, j=0.5, i=i, hyper_const=s_hyper, label='s')
        p3 = sim.construct_electronic_state(f_sp3, 0.5, 1, 1.5, i=i, hyper_const=p3_hyper, label='p3')
        d3 = sim.construct_electronic_state(f_sp3 - f_d3p3, 0.5, 2, 1.5, i=i, hyper_const=d3_hyper, label='d3')
        d5 = sim.construct_electronic_state(f_sp3 - f_d5p3, 0.5, 2, 2.5, i=i, hyper_const=d5_hyper, label='d5')

        decay = sim.DecayMap(labels=[('s', 'p3'), ('p3', 'd3'), ('p3', 'd5')], a=[a_sp3, a_d3p3, a_d5p3])
        # The states are linked by Einstein-A coefficients via the specified labels.

        states = s + p3 + d3 + d5
        ca43 = sim.Atom(states=states, decay_map=decay)

        ca43.plot()  # Plot the atom with all states.

        pol_sp = sim.Polarization([0, 1, 0], q_axis=2)
        laser_sp = sim.Laser(freq=f_sp3 - 1697, polarization=pol_sp, intensity=500)
        # Put the laser on a specific hyperfine transition.

        inter = sim.Interaction(atom=ca43, lasers=[laser_sp, ], delta_max=400.)
        # Set delta_max (MHz) to do a RWA in the off-resonance transitions.

        # inter.resonance_info()  # Print the detunings of the lasers from the considered transitions.
        # inter.controlled = True  # Use an error controlled solver to deal with fast dynamics.
        # inter.dt = 1e-4  # Alternatively, decrease the step size.

        times = [0., 0.2]  # Integration time in us.
        delta = np.linspace(-180, 150, 331)

        results = inter.rates(times, delta)
        # Solve the rate equation for all times, assuming equal population in all s-states.

        # Plot the population of all d5-states.
        for f in qs.get_f(i, 2.5):
            plt.plot(delta, np.sum(results[:, ca43.get_state_indexes('d5', f), -1], axis=1), label='F={}'.format(f))
        plt.xlabel('f - {} (MHz)'.format(laser_sp.freq))
        plt.ylabel('d5-state population after {} us'.format(times[-1]))
        plt.legend()
        plt.show()

        ''' Master equation here already takes ~ 5 - 10 min '''
        # # Plot the population of all d5-states.
        # results = inter.master(times, delta)
        # # Solve the master equation for all times, assuming equal population in all s-states
        # y = np.diagonal(results[:, :, :, -1], axis1=1, axis2=2).real
        # for f in pc.get_f(I, 2.5):
        #     plt.plot(delta, np.sum(y[:, ca43.get_state_indexes('d5', f)], axis=1), label='F={}'.format(f))
        # plt.xlabel('f - {} (MHz)'.format(laser_sp.freq))
        # plt.ylabel('d5-state population after {} us'.format(times[-1]))
        # plt.show()

    if 3 in n:
        """
        Example 3: Interaction between a singly-charged lithium ion and two lasers.
        
        In example 3, Fig. 5 from [Noertershaeuser et al. Phys. Rev. Accel. Beams 24, 024701 (2021),
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

        pol_b = sim.Polarization([0, 0, 1], q_axis=2)  # sigma+ polarization
        pol_r = sim.Polarization([0, 0, 1], q_axis=2)  # sigma+ polarization
        laser_b = sim.Laser(freq=f + 6234.29, polarization=pol_b, intensity=i_b)  # blue laser
        laser_r = sim.Laser(freq=f - 13566., polarization=pol_r, intensity=i_r)  # red laser

        print('Saturation s(blue): {}'.format(qs.saturation(i_b, f, a, qs.a(1.5, 1, 1.5, 2, 2.5))))
        print('Saturation s(red): {}'.format(qs.saturation(i_r, f, a, qs.a(1.5, 1, 2.5, 2, 2.5))))
        # The saturation intensity can be compared easily to the specified values in the paper.

        inter = sim.Interaction(atom=li7, lasers=[laser_b, laser_r])
        inter.controlled = True
        inter.resonance_info()  # Print the resonance info.

        t = np.concatenate([np.zeros(1), np.logspace(-4, 1, 5000)], axis=0)  # Integration time in us.
        y0 = li7.get_y0(['s3', 's5'])

        y = inter.rates(t, y0=y0)  # Solve the rate equation for t and plot with logarithmic scaling.
        plt.xscale('log')
        for s in ['s3', 's5', 'p']:
            plt.plot(t, np.sum(y[0, li7.get_state_indexes(s)], axis=0), label=s)
        plt.xlabel('time (us)')
        plt.ylabel('state population')
        plt.legend()
        plt.show()

        ''' The result is quiet different with fully coherent dynamics. '''
        y = inter.master(t, y0=y0)  # Solve the master equation for t and plot with logarithmic scaling.
        y = np.diagonal(y[0], axis1=0, axis2=1).real
        plt.xscale('log')
        for s in ['s3', 's5', 'p']:
            plt.plot(t, np.sum(y[:, li7.get_state_indexes(s)], axis=1), label=s)
        plt.xlabel('time (us)')
        plt.ylabel('state population')
        plt.legend()
        plt.show()

    if 4 in n:
        """
        Example 4: Scattering rate of a para-he-like system.
        
        In example 4, the scattering rate, including quantum interference effects, is derived from the rate equations,
        the master equation and perturbatively for the lowest lying transition in the singlet system of a he-like atom.
        """
        f_p = 7e8
        a_p = 100.

        i = 1.5
        s_hyper = [0.]
        p_hyper = [10.]

        s = sim.construct_electronic_state(freq_0=0, s=0, l=0, j=0, i=i, hyper_const=s_hyper, label='s')
        p = sim.construct_electronic_state(freq_0=f_p, s=0, l=1, j=1, i=i, hyper_const=p_hyper, label='p')

        decay = sim.DecayMap(labels=[('s', 'p')], a=[a_p])

        states = s + p
        he = sim.Atom(states=states, decay_map=decay)
        # he.plot()

        intensity = 0.1
        pol_sp = sim.Polarization([0, 1, 0], vec_as_q=True, q_axis=2)
        print(pol_sp.x)
        print(pol_sp.q)
        laser_sp = sim.Laser(freq=f_p, polarization=pol_sp, intensity=intensity)

        inter = sim.Interaction(atom=he, lasers=[laser_sp, ], delta_max=500.)
        # inter.resonance_info()

        times = [0., 0.2]
        delta = np.linspace(-100, 100, 201)
        theta, phi = np.pi / 2, 0.
        # theta, phi = 0., 0.

        results = inter.rates(times, delta)
        y = he.scattering_rate(results, as_density_matrix=False)[:, -1] / (4 * np.pi)
        plt.plot(delta, y, '-C3', label=r'$4\pi$ rates')
        y = he.scattering_rate(results, theta, phi, as_density_matrix=False)[:, -1]
        plt.plot(delta, y, '-C2', label='angular non-QI')

        rho = inter.master(times, delta)
        y = he.scattering_rate(rho.real)[:, -1] / (4 * np.pi)
        plt.plot(delta, y, '--C0', label=r'$4\pi$ master')
        y = he.scattering_rate(rho, theta, phi)[:, -1]
        plt.plot(delta, y, '-C1', label='full QI')

        sr = sim.ScatteringRate(he, polarization=pol_sp)
        y = sr.generate_y(delta, theta, phi)[:, 0, 0]
        s = qs.saturation(intensity, f_p, a_p, 1)
        plt.plot(delta, s * y, '--C7', label='QI pert.')

        plt.xlabel('f - {} (MHz)'.format(laser_sp.freq))
        plt.ylabel('scattering rate after {} us (MHz)'.format(times[-1]))
        plt.legend()
        plt.show()

    if 5 in n:
        """
        Example 5: Coherence of two laser beams / Time-dependent Rabi frequencies.
        
        In example 5, time-dependent Rabi frequencies are tested using two laser beams that drive the same transition.
        """
        f_p = 7e8
        a_p = 10.

        i = 0.
        s_hyper = [0.]
        p_hyper = [10.]

        s = sim.construct_electronic_state(freq_0=0, s=0, l=0, j=0, i=i, hyper_const=s_hyper, label='s')
        p = sim.construct_electronic_state(freq_0=f_p, s=0, l=1, j=1, i=i, hyper_const=p_hyper, label='p')

        decay = sim.DecayMap(labels=[('s', 'p')], a=[a_p])

        states = s + p
        he = sim.Atom(states=states, decay_map=decay)
        # he.plot()

        intensity = 1000.
        pol_0 = sim.Polarization([0, 0, 1], vec_as_q=False, q_axis=2)
        pol_1 = sim.Polarization([0, 0, 1], vec_as_q=False, q_axis=2)
        print('x:', pol_0.x)
        print('q:', pol_0.q)
        laser_0 = sim.Laser(freq=f_p - 0.1, polarization=pol_0, intensity=intensity)
        laser_1 = sim.Laser(freq=f_p + 0.1, polarization=pol_1, intensity=intensity)

        inter = sim.Interaction(atom=he, lasers=[laser_0, laser_1], delta_max=500.)
        inter.time_dependent = True
        # inter.resonance_info()

        times = np.linspace(0, 10., 10001)

        y = inter.rates(times)
        plt.plot(times, np.sum(y[0, he.get_state_indexes('s')], axis=0), '--C0')
        plt.plot(times, np.sum(y[0, he.get_state_indexes('p')], axis=0), '--C1')

        rho = inter.master(times)
        y = np.diagonal(rho, axis1=1, axis2=2).real
        y = np.transpose(y, axes=[0, 2, 1])
        ys = np.sum(y[0, he.get_state_indexes('s')], axis=0)
        yp = np.sum(y[0, he.get_state_indexes('p')], axis=0)

        plt.plot(times, ys, '-C0', label='s')
        plt.plot(times, yp, '-C1', label='p')
        plt.legend()
        plt.xlabel('time (us)')
        plt.ylabel('state population')
        plt.show()

    if 6 in n:
        p = sim.State(0., 1, 1, 2, i=1, f=2, m=0, hyper_const=[1.], label='p')
        d = sim.State(7e8, 1, 2, 2, i=1, f=2, m=1, hyper_const=[1.], label='d')
        decay = sim.DecayMap([('p', 'd')], [50.])
        atom = sim.Atom([p, d], decay)

        pol = sim.Polarization([1, 1, 1], q_axis=2, vec_as_q=True)
        laser = sim.Laser(7e8, intensity=100, polarization=pol)

        inter = sim.Interaction(atom, [laser])
        print(inter.get_rabi(0))


if __name__ == '__main__':
    example({1})
