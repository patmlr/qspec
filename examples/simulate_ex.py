"""
examples.simulate_ex
====================

Example script / Guide for the qspec.simulate module.


- Einstein coefficients taken from [NIST Atomic Spectra Database, https://doi.org/10.18434/T4W30F].
- Frequencies of 40Ca+ taken from [P. Mueller et pc., Phys. Rev. Research 2, 043351 (2020),
                                  https://doi.org/10.1103/PhysRevResearch.2.043351].
- Hyperfine-structure constants [A, B] of 40Ca+ taken from [Noertershaeuser et pc., Eur. Phys. J. D 2, 33-39 (1998),
                                                            https://doi.org/10.1007/s100530050107]
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc

import qspec as qs
import qspec.simulate as sim

# from qspec import a_einstein_m1, a_einstein_m1_fm


def example(n=None):
    """
    Run one or several of the available examples.

    Example 0: Interaction between 40Ca+ and a laser.
    Example 1: Interaction between 40Ca+ and two lasers off-resonance / Rabi pumping.
    Example 2: Interaction between 43Ca+ with hyperfine structure and a laser.
    Example 3: Interaction between a singly-charged lithium-ion and two lasers.
    Example 4: Time-evolved scattering rate of a para-he-like system.
    Example 5: Coherent excitation with two laser beams resulting in time-dependent Rabi frequencies.
    Example 6: Monte-Carlo simulation of 40Ca+ interacting with two lasers, including photon recoils.
    Example 7: M1 decay of the 3S1 ground state of ortho-heliumlike C4+ into the para-heliumlike 1S0 ground state.
    Example 8: Rabi oscillations in the S1/2 <-> D3/2,5/2 transitions.
    Example 9: 3D multipole emission pattern.

    :param n: The number of the example or a list/set of numbers.
    """
    if n is None:
        n = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
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

        s = sim.gen_electronic_ls_state(freq_j=0, s=0.5, l=0, j=0.5, label='s')  # A list of all 4s substates.
        p = sim.gen_electronic_ls_state(f_sp, 0.5, 1, 0.5, label='p')  # A list of all 4p 2P1/2 substates.
        d = sim.gen_electronic_ls_state(f_sp - f_dp, 0.5, 2, 1.5, label='d')  # A list of all 3d 2D3/2 substates.

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

        s = sim.gen_electronic_ls_state(freq_j=0, s=0.5, l=0, j=0.5, label='s')  # A list of all 4s substates.
        p = sim.gen_electronic_ls_state(f_sp, 0.5, 1, 0.5, label='p')  # A list of all 4p 2P1/2 substates.
        d = sim.gen_electronic_ls_state(f_sp - f_dp, 0.5, 2, 1.5, label='d')  # A list of all 3d 2D3/2 substates.

        decay = sim.DecayMap(labels=[('s', 'p'), ('p', 'd')], a=[a_sp, a_dp])
        # The states are linked by Einstein-A coefficients via the specified labels.

        ca40 = sim.Atom(states=s + p + d, decay_map=decay)  # The Atom with all states and the decay information.

        pol = sim.Polarization([0, 1, 0], q_axis=2)
        laser_sp = sim.Laser(freq=f_sp + 400, polarization=pol, intensity=1000)  # Linear polarized laser for
        # the ground-state transition.
        laser_dp = sim.Laser(freq=f_dp + 400, polarization=pol, intensity=1000)  # Linear polarized laser for
        # the metastable-state transition.

        inter = sim.Interaction(atom=ca40, lasers=[laser_sp, laser_dp], delta_max=1000)
        inter.controlled = True  # Use the controlled solver.
        # inter.dt = 4e-5  # or small step sizes.

        times = [0, 2]  # Integration time in us.
        delta = np.linspace(-6.5, 6.5, 501)
        r0 = inter.rabi(0)
        r1 = inter.rabi(1)

        d = inter.delta()

        h = inter.hamiltonian(0., [5.], 1, [0.])

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
        s = sim.gen_electronic_ls_state(freq_j=0, s=0.5, l=0, j=0.5, i=i, hyper_const=s_hyper, label='s')
        p3 = sim.gen_electronic_ls_state(f_sp3, 0.5, 1, 1.5, i=i, hyper_const=p3_hyper, label='p3')
        d3 = sim.gen_electronic_ls_state(f_sp3 - f_d3p3, 0.5, 2, 1.5, i=i, hyper_const=d3_hyper, label='d3')
        d5 = sim.gen_electronic_ls_state(f_sp3 - f_d5p3, 0.5, 2, 2.5, i=i, hyper_const=d5_hyper, label='d5')

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

        states = sim.gen_hyperfine_ls_state(freq_j=0, s=1, l=0, j=1, i=1.5, f=1.5,
                                            hyper_const=[df_s / 2.5, ], label='s3')
        states += sim.gen_hyperfine_ls_state(0, 1, 0, 1, 1.5, 2.5, [df_s / 2.5, ], label='s5')
        states += sim.gen_hyperfine_ls_state(f, 1, 1, 2, 1.5, 2.5, [df_p / 3.5, ], label='p')

        decay = sim.DecayMap(labels=[('s3', 'p'), ('s5', 'p')], a=[a, a])

        li7 = sim.Atom(states=states, decay_map=decay)  # The Atom with all states and the decay information.
        li7.plot()  # Plot the involved states.

        i_b = 200  # Intensity of the blue laser (uW / mm ** 2)
        i_r = 2000  # Intensity of the red laser (uW / mm ** 2)

        pol_b = sim.Polarization([0, 0, 1], q_axis=2)  # sigma+ polarization
        pol_r = sim.Polarization([0, 0, 1], q_axis=2)  # sigma+ polarization
        laser_b = sim.Laser(freq=f + 6234.29, polarization=pol_b, intensity=i_b)  # blue laser
        laser_r = sim.Laser(freq=f - 13566., polarization=pol_r, intensity=i_r)  # red laser

        print('Saturation s(blue): {}'.format(qs.saturation(i_b, f, a)))
        print('Saturation s(red): {}'.format(qs.saturation(i_r, f, a)))
        # The saturation intensity can be compared easily to the specified values in the paper.

        inter = sim.Interaction(atom=li7, lasers=[laser_b, laser_r])
        inter.controlled = True
        inter.resonance_info()  # Print the resonance info.

        t = np.concatenate([np.zeros(1), np.logspace(-4, 2.5, 5000)], axis=0)  # Integration time in us.
        y0 = li7.get_y0(['s3', 's5'])

        y = inter.rates(t, y0=y0)  # Solve the rate equation for t and plot with logarithmic scaling.
        plt.xscale('log')
        for s in ['s3', 's5', 'p']:
            plt.plot(t, np.sum(y[0, li7.get_state_indexes(s)], axis=0), label=s)
        plt.xlabel('time (us)')
        plt.ylabel('state population')
        plt.legend()
        plt.show()

        ''' The result is quite different with fully coherent dynamics. '''
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

        i = 0.5
        s_hyper = [0.]
        p_hyper = [10.]

        s = sim.gen_electronic_ls_state(freq_j=0, s=0, l=0, j=0, i=i, hyper_const=s_hyper, label='s')
        p = sim.gen_electronic_ls_state(freq_j=f_p, s=0, l=1, j=1, i=i, hyper_const=p_hyper, label='p')

        decay = sim.DecayMap(labels=[('s', 'p')], a=[a_p])

        states = s + p
        he = sim.Atom(states=states, decay_map=decay)
        # he.plot()

        intensity = 0.1
        pol_sp = sim.Polarization([0, 0, 1], vec_as_q=False, q_axis=[0, 0, 1])
        qs.printh('Polarization before processing.')
        print('x:', pol_sp.x)
        print('q:', pol_sp.q)
        print('q_axis:', pol_sp.q_axis)

        laser_sp = sim.Laser(freq=f_p, polarization=pol_sp, intensity=intensity, k=[0, 1, 0])

        env = sim.Environment(B=[0., 0., 1e-6])
        inter = sim.Interaction(atom=he, lasers=[laser_sp, ], environment=env, delta_max=1000.)
        # inter.dt_max = 1e-4
        inter.controlled = True
        # inter.resonance_info()
        qs.printh('\nPolarization after processing.')
        print('x:', pol_sp.x)
        print('q:', pol_sp.q)
        print('q_axis:', pol_sp.q_axis)
        print('kpol:', laser_sp.get_kpol(1, True, env.B))

        times = [0., 0.2]
        delta = np.linspace(-100, 100, 201)
        theta, phi = np.pi / 2, 0.
        # theta, phi = 0., 0.

        r = inter.rabi()

        results = inter.rates(times, delta)
        y = he.scattering_rate(results, as_density_matrix=False, theta=theta, phi=phi)[0, :, -1]
        plt.plot(delta, y, '-C2', label='angular non-QI')
        y = he.scattering_rate(results, as_density_matrix=False)[:, -1] / (4 * np.pi)
        plt.plot(delta, y, '-C3', label=r'$4\pi$ rates')

        rho = inter.master(times, delta)
        y = he.scattering_rate(rho.real)[:, -1] / (4 * np.pi)
        plt.plot(delta, y, '--C0', label=r'$4\pi$ master')
        y = he.scattering_rate(rho, theta=theta, phi=phi)[0, :, -1]
        plt.plot(delta, y, '-C1', label='QI master')

        sr = sim.ScatteringRate(he, laser=laser_sp, b=env.B)
        y = sr.generate_y(delta, theta=theta, phi=phi)[:, 0, 0]
        plt.plot(delta, y, '--C7', label='QI pert.')

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

        s = sim.gen_electronic_ls_state(freq_j=0, s=0, l=0, j=0, i=i, hyper_const=s_hyper, label='s')
        p = sim.gen_electronic_ls_state(freq_j=f_p, s=0, l=1, j=1, i=i, hyper_const=p_hyper, label='p')

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

        times = np.linspace(0., 10., 10001)

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
        """
        Example 6: Interaction between 40Ca+ and two lasers.
        
        In example 6, a Monte-Carlo simulation of 40Ca+ interacting with two lasers,
        including photon recoils, is implemented.
        """
        f_sp = 755222766  # The 4s -> 4p 2P1/2 transition frequency.
        f_dp = 346000235  # The 3d 2D3/2 -> 4p 2P1/2 transition frequency.

        a_sp = 140  # The Einstein coefficients of the two transitions
        a_dp = 10.7

        s = sim.gen_electronic_ls_state(freq_j=0, s=0.5, l=0, j=0.5, label='s')  # A list of all 4s substates.
        p = sim.gen_electronic_ls_state(f_sp, 0.5, 1, 0.5, label='p')  # A list of all 4p 2P1/2 substates.
        d = sim.gen_electronic_ls_state(f_sp - f_dp, 0.5, 2, 1.5, label='d')  # A list of all 3d 2D3/2 substates.

        decay = sim.DecayMap(labels=[('s', 'p'), ('p', 'd')], a=[a_sp, a_dp])
        # The states are linked by Einstein-A coefficients via the specified labels.

        ca40 = sim.Atom(states=s + p + d, decay_map=decay, mass=39.9)
        # The Atom with all states and the decay information.

        pol_sp = sim.Polarization([1, 1, 1], q_axis=2, vec_as_q=True)
        laser_sp = sim.Laser(freq=f_sp, polarization=pol_sp, intensity=500)
        # Linear polarized laser for the ground-state transition with 500 uW / mm**2.

        pol_dp = sim.Polarization([1, 1, 1], q_axis=2, vec_as_q=True)
        laser_dp = sim.Laser(freq=f_dp, polarization=pol_dp, intensity=500, k=[1, 0, 0])
        # Linear polarized laser for the ground-state transition with 500 uW / mm**2.

        inter = sim.Interaction(atom=ca40, lasers=[laser_sp, laser_dp])  # The interaction.
        inter.controlled = False
        inter.dense = False
        inter.dt = 1e-3
        inter.resonance_info()  # Print the detuning of the lasers from the considered transitions.

        times = np.linspace(0., 3., 1001)  # Integration times.
        # delta = np.linspace(-50, 50, 101)  # Laser detunings from the given laser frequency.

        y0 = ca40.get_y0_mc(1000)
        rho, v = inter.mc_master(times, y0=y0, dynamics=True, as_density_matrix=True)
        y = np.diagonal(rho, axis1=1, axis2=2).real
        y = np.transpose(y, axes=[0, 2, 1])
        y = np.mean(y, axis=0)

        ys = np.sum(y[ca40.get_state_indexes('s')], axis=0)
        yp = np.sum(y[ca40.get_state_indexes('p')], axis=0)
        yd = np.sum(y[ca40.get_state_indexes('d')], axis=0)

        plt.plot(times, ys, label='s')
        plt.plot(times, yp, label='p')
        plt.plot(times, yd, label='d')
        plt.legend()
        plt.show()

        plt.hist(v[:, 0], bins=40)
        plt.hist(v[:, 1], bins=40)
        plt.show()

    if 7 in n:
        """
        Example 7: m1 decay in C4+.
        
        In example 7, The 3S1 ground state of ortho-heliumlike C4+ is decaying
        into the global para-heliumlike 1S0 ground state through an m1 transition.
        """

        f31 = qs.inv_cm_to_freq(2411292.798)
        a31 = 4.857e-5
        print(f'tau: {1e-3 / a31} ms')

        s1 = sim.gen_electronic_state(0., parity='even', j=0, i=0, ls=[0, 0], label='s1')
        s3 = sim.gen_electronic_state(f31, parity='even', j=1, i=0, ls=[0, 1], label='s3')

        decay_map = sim.DecayMap(labels=[('s1', 's3')], a=[a31])

        atom = sim.Atom(s1 + s3, decay_map)

        # laser = sim.Laser(f31, intensity=100)

        inter = sim.Interaction(atom, [])
        inter.controlled = True
        inter.dt = 1.
        inter.dt_max = 1.

        y0 = np.zeros(atom.size, dtype=float)
        y0[atom.get_state_indexes('s3')] = 1.

        t = np.linspace(0., 100000., 101)
        y = inter.rates(t, y0=y0)[0]

        y1 = y[atom.get_state_indexes('s1')][0]
        y3 = np.sum(y[atom.get_state_indexes('s3')], axis=0)

        plt.plot(t * 1e-3, y1, label=r'$^1$S$_0$')
        plt.plot(t * 1e-3, y3, label=r'$^3$S$_1$')
        plt.plot([1e-3 / a31], [1 / np.e], 'oC3', label=r'$\tau_{31}$')
        plt.vlines(1e-3 / a31, 0, 1 / np.e, colors='C3', ls='--')

        plt.legend()
        plt.xlabel('Time (ms)')
        plt.ylabel('Population')
        plt.show()

    if 8 in n:
        """
        Example 8: Rabi oscillations in the S1/2 <-> D3/2,5/2 transitions.
        
        In example 8, coherent population transfer between the S1/2 ground state and the D3/2 or D5/2 excited state
        in Ca+ is simulated.
        """
        print(2.356 / np.pi)
        f_sp = 755222766.
        f_d3p3 = 352682482.
        f_d5p3 = 350862883.
        a = 1.3e-6
        print(f'tau: {1e-6 / a} s')

        s = sim.gen_electronic_state(freq_j=0, parity='e', ls=[0, 0.5], j=0.5, label='s')
        d3 = sim.gen_electronic_state(f_sp - f_d3p3, parity='e', j=1.5, ls=[2, 0.5], label='d3')
        d5 = sim.gen_electronic_state(f_sp - f_d5p3, parity='e', j=2.5, ls=[2, 0.5], label='d5')

        states = s + d3
        # states = s + [d3[-1]]
        decay_map = sim.DecayMap(labels=[('s', 'd3'), ('s', 'd5')],
                                 a=[{'m1': 0.01 * a, 'e2': a}, {'e2': a}], k_max=2)

        ca40 = sim.Atom(states, decay_map)

        phase_angle = 3 * np.pi / 2
        phase = np.exp(1j * phase_angle)
        pol_sd3_0 = sim.Polarization([0, phase, 0], q_axis=2, vec_as_q=False)
        laser_sd3_0 = sim.Laser(freq=f_sp - f_d3p3, polarization=pol_sd3_0, intensity=10, k=[-1, 0, 0])

        pol_sd3_1 = sim.Polarization([-1, 1, 0], q_axis=2, vec_as_q=False)
        laser_sd3_1 = sim.Laser(freq=f_sp - f_d3p3, polarization=pol_sd3_1, intensity=10, k=[-1, -1, 0])

        pol_sd5 = sim.Polarization([0, 1, 0], q_axis=2)
        laser_sd5 = sim.Laser(freq=f_sp - f_d5p3, polarization=pol_sd5, intensity=10)

        print(ca40.get_multipole_types('s', 'd3'))
        inter = sim.Interaction(ca40, lasers=[laser_sd3_0, laser_sd3_1])
        inter.resonance_info()

        times = np.linspace(0., 2000., 101)
        y0 = np.zeros(ca40.size, dtype=float)
        y0[0] = 1.
        # y = inter.rates(times, analytic=False, y0=y0)[0]
        y = inter.master(times, y0=y0)[0]
        y = np.transpose(np.diagonal(y, axis1=0, axis2=1).real, axes=[1, 0])

        r = inter.rabi(0)

        # labels = ['d3', 'd5']
        # for label in labels:
        #     i = ca40.get_state_indexes(label)
        #     plt.plot(times, np.sum(y[i], axis=0), label=label)

        plt.plot(times, y[0], '--k', label='s($' + str(qs.quant(-0.5)) + '$)', zorder=50)
        plt.plot(times, y[1], ':k', label='s($' + str(qs.quant(0.5)) + '$)', zorder=100)

        i_d3 = ca40.get_state_indexes('d3')
        cmap = plt.get_cmap('viridis', i_d3.size)
        for ic, im in enumerate(i_d3):
            m = ca40.states[im].m
            plt.plot(times, y[im], c=cmap(ic), label='d($' + str(qs.quant(m)) + '$)')

        plt.legend()
        plt.xlabel(r'Time ($\mu$s)')
        plt.ylabel('Population')
        plt.show()

    if 9 in n:
        """
        Example 9: 3D multipole emission pattern.
        
        In example 9, the differential scattering rate for higher multipole orders is calculated and plotted in 3D.
        """
        f_eg = 7e8
        a_eg = 1.

        g_hyper = [0.]
        e_hyper = [5.]

        i = 0.
        jg = 2.
        je = 4.
        dm = 0

        g = sim.State(0., parity='e', j=jg, i=i, f=jg + i, m=jg + i, hyper_const=g_hyper, label='g')
        e = sim.State(f_eg, parity='e', j=je, i=i, f=je + i, m=jg + i + dm, hyper_const=e_hyper, label='e')

        decay = sim.DecayMap(labels=[('g', 'e')], a=[a_eg], k_max=int(je - jg))

        states = [g, e]
        atom = sim.Atom(states=states, decay_map=decay)
        # atom.plot()

        intensity = 100.
        pol_eg = sim.Polarization([1, 0, 0], vec_as_q=False, q_axis=[0, 0, 1])
        laser_eg = sim.Laser(freq=f_eg, polarization=pol_eg, intensity=intensity, k=[0., 1., 0.])

        env = sim.Environment(B=[0., 0., 1e-6])
        inter = sim.Interaction(atom=atom, lasers=[laser_eg, ], environment=env, delta_max=1000.)
        # inter.dt_max = 1e-4
        inter.controlled = True

        r = inter.rabi()

        t = np.linspace(0., 1., 301)

        y0 = np.zeros(atom.size, dtype=complex)
        y0[1] = 1.

        rho = inter.master(t, y0=y0)
        y = sim.density_matrix_diagonal(rho, axis=1)[0]

        plt.plot(t, y[0], label=g.label)
        plt.plot(t, y[1], label=e.label)
        plt.legend()
        plt.xlabel(r'Time ($\mu$s)')
        plt.ylabel('Population')
        plt.show()

        # n_points = 101
        # indexes = np.arange(0, n_points, dtype=float) + 0.5
        # phi = np.arccos(1 - 2 * indexes / n_points)
        # theta = np.pi * (1 + np.sqrt(5)) * indexes

        n_theta, n_phi = 32, 64
        theta = np.linspace(0., np.pi, n_theta)
        phi = np.linspace(0., 2 * np.pi, n_phi)

        theta, phi = np.meshgrid(theta, phi, indexing='ij')

        r0 = atom.scattering_rate(rho[0, :, :, -1], theta=np.pi / 2, phi=0., x_vec='-', axis=0)
        print(f'r0: {r0[0]} MHz')

        r = atom.scattering_rate(rho[0, :, :, -1], theta=theta, phi=phi, x_vec='+', axis=0)
        r /= np.max(r)
        r = r.reshape((n_theta, n_phi))

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        cm = plt.get_cmap('plasma')

        ax.plot_surface(x, y, z, facecolors=cm(r), rcount=64, ccount=128, linewidth=0, antialiased=False)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_zlim(-1.1, 1.1)
        ax.set_box_aspect((1., 1., 1.))
        plt.show()



if __name__ == "__main__":
    example({0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
    # example({0})
