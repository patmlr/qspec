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
import qspec as qs
import qspec.simulate as sim


class TestPhysics(ut.TestCase):

    def test_simple_atom(self):
        states = [sim.State(0, 0.5, 0, 0.5, -0.5, 'even', label='s-'),
                  sim.State(7e8, 0.5, 0, 0.5, -0.5, 'odd', label='p-'),
                  sim.State(7e8, 0.5, 0, 0.5, 0.5, 'odd', label='p+'),
                  sim.State(4e8, 1.5, 0, 1.5, 0.5, 'even', label='d+')]
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

    def test_magnetic_field(self):
        i = 1.5
        j = 1
        mu = -1.09316
        gi = mu / i
        gj = 1.35
        a_hyper = 2.1743
        b_hyper = 49.11
        b = np.linspace(0., 4e-3, 4000)
        b_env = b[2000]
        e_eig, m_list, fm_list, mi_mj_list = qs.hyper_zeeman_num(i, j, a_hyper, b_hyper, gi, gj, b)

        states = sim.gen_electronic_state(0, j, i, [a_hyper, b_hyper], gj=gj, gi=gi)
        atom = sim.Atom(states)
        env = sim.Environment(B=b_env)
        inter = sim.Interaction(atom, environment=env)

        f_plotted = set()
        m_plotted = set()
        cmap = plt.get_cmap('inferno')
        f_colored = True
        for im, (_e_eig, _f_list, _m, mi_mj) in enumerate(zip(e_eig, fm_list, m_list, mi_mj_list)):
            for k in range(_e_eig.shape[1]):
                if f_colored:
                    c_val = (_f_list[k] - abs(i - j)) / (i + j - abs(i - j) + 1)
                    c = cmap(c_val)
                    plt.plot(b * 1e3, _e_eig[:, k], color=c, ls='-', lw=0.85, alpha=0.8,
                             label=rf'$F = {int(_f_list[k])}$' if c not in f_plotted else None, zorder=10 * c_val)
                    f_plotted.add(c)
                    handles, labels = plt.gca().get_legend_handles_labels()
                else:
                    mj = mi_mj[k][1]
                    c_val = (mj + j) / (2 * j + 1)
                    c = cmap(c_val)
                    plt.plot(b * 1e3, _e_eig[:, k], color=c, ls='-', lw=0.85, alpha=0.8,
                             label=rf'$m_J = {mj}$' if mj not in m_plotted else None, zorder=10 * c_val)
                    handles, labels = plt.gca().get_legend_handles_labels()
                    handles, labels = handles[::-1], labels[::-1]
                    m_plotted.add(mj)

        for s in inter.atom.states:
            c_val = (s.f - abs(i - j)) / (i + j - abs(i - j) + 1)
            c = cmap(c_val)
            plt.plot([b_env * 1e3], [s.freq], '.', color=c)

        # plt.hlines(e_th, 0., 4., colors='grey', ls='--')

        plt.xlabel('Magnetic field (mT)')
        plt.ylabel('Frequency shift (MHz)')
        # plt.legend(handles, labels)
        plt.xlim(0., 4.)
        plt.subplots_adjust(left=0.11, bottom=0.1, right=0.98, top=0.99)
        plt.show()

