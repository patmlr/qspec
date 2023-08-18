# -*- coding: utf-8 -*-
"""
PyCLS.tests.test_physics

Created on 14.04.2021

@author: Patrick Mueller

Module including unittests for the physics module.
"""

import unittest as ut

import numpy as np
import matplotlib.pyplot as plt
import qspec.simulate as sim


class TestPhysics(ut.TestCase):

    def test_simple_atom(self):
        states = [sim.State(0, 0.5, 0, 0.5, 0, 0.5, -0.5, label='s-'),
                  sim.State(7e8, 0.5, 1, 0.5, 0, 0.5, -0.5, label='p-'),
                  sim.State(7e8, 0.5, 1, 0.5, 0, 0.5, 0.5, label='p+'),
                  sim.State(4e8, 0.5, 2, 1.5, 0, 1.5, 0.5, label='d+')]
        decay_map = sim.DecayMap([('s-', 'p-'), ('s-', 'p+'), ('d+', 'p-'), ('d+', 'p+')], [1e2, 1e2, 10., 10.])
        atom = sim.Atom(states, decay_map=decay_map)
        pol_0 = sim.Polarization([0, 0, 1], q_axis=2, vec_as_q=False)
        pol_1 = sim.Polarization([0, 1, 0], q_axis=2, vec_as_q=True)
        lasers = [sim.Laser(7e8, 1e3, polarization=pol_0),
                  sim.Laser(3e8, 0, polarization=pol_1)]
        inter = sim.Interaction(atom, lasers, controlled=True)
        inter.resonance_info()

        t = np.linspace(0, 4, 1001)

        # y = inter.rates(t)[0]

        y = inter.master(t)
        y = np.diagonal(y[0], axis1=0, axis2=1).real.T

        for s in states:
            plt.plot(t, np.sum(y[atom.get_state_indexes(s.label)], axis=0), label=s.label)
        plt.xlabel('time (us)')
        plt.ylabel('state populations')
        plt.legend()
        plt.show()

        for s in ['s-', 'd+']:
            sr = atom.scattering_rate(y, j=atom.get_state_indexes(s), axis=0)
            plt.plot(t, sr, label=f'into {s}')
        plt.xlabel('time (us)')
        plt.ylabel('scattering rate (MHz)')
        plt.legend()
        plt.show()
