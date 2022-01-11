# -*- coding: utf-8 -*-
"""
PyCLS.tests.test_simulate

Created on 17.05.2021

@author: Patrick Mueller

Module including unittests for the simulate module.
"""

import unittest as ut
import numpy as np
import matplotlib.pyplot as plt

import pycol.simulate as sim


s_hyper = [-806.4, ]
p1_hyper = [-145.6]
p3_hyper = [-31., -6.9]
d3_hyper = [-47.3, -3.7]


class TestPhysics(ut.TestCase):
    s = sim.construct_electronic_state(0, 0.5, 0, 0.5, 0, hyper_const=s_hyper, label='s')
    p = sim.construct_electronic_state(7e8, 0.5, 1, 0.5, 0, hyper_const=p1_hyper, label='p')
    d = sim.construct_electronic_state(4e8, 0.5, 2, 1.5, 0, hyper_const=d3_hyper, label='d')

    pol = sim.Polarization([0, 1, 0], q_axis=2)
    decay_map = sim.DecayMap([('s', 'p'), ('d', 'p'), ('d', 's')], [1.4e2, 2e1, 1e0])
    atom = sim.Atom(s + p + d, decay_map)
    laser_1 = sim.Laser(7e8 + 0, intensity=200, polarization=pol)
    laser_2 = sim.Laser(3e8 + 0, intensity=500, polarization=pol)
    laser_3 = sim.Laser(4e8 + 0, intensity=1, polarization=pol)
    inter = sim.Interaction(atom, [laser_1, laser_2, laser_3], delta_max=3000)
    # result = inter.rates(0.4)
    # result = inter.schroedinger(0.4)
    print(inter.deltamap)
    result = inter.master(0.4)
    x = result.x
    y = result.y
    plt.plot(x, y)
    plt.plot(x, np.sum(y, axis=1), 'k--')
    plt.show()
