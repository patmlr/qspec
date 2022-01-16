# -*- coding: utf-8 -*-
"""
PyCLS.tests.test_simulate

Created on 17.05.2021

@author: Patrick Mueller

Module including unittests for the simulate module.
"""

import unittest as ut
from time import time
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt

import pycol.simulate as sim
from pycol import tools


s_hyper = [-806.4, ]
p1_hyper = [-145.6]
p3_hyper = [-31., -6.9]
d3_hyper = [-47.3, -3.7]


class TestPhysics(ut.TestCase):
    s = sim.construct_electronic_state(0, 0.5, 0, 0.5, 0, hyper_const=s_hyper, label='s')
    p = sim.construct_electronic_state(7e8, 0.5, 1, 0.5, 0, hyper_const=p1_hyper, label='p')
    d = sim.construct_electronic_state(4e8, 0.5, 2, 1.5, 0, hyper_const=d3_hyper, label='d')

    pol = sim.Polarization([0, 1, 0], q_axis=2)
    decay_map = sim.DecayMap([('s', 'p'), ('d', 'p'), ('d', 's')], [140, 20, 0])
    # decay_map = sim.DecayMap([('s', 'p'), ], [140, ])
    atom = sim.Atom(s + p + d, decay_map, mass=40)
    laser_1 = sim.Laser(7e8, intensity=1000, polarization=pol)
    laser_2 = sim.Laser(3e8, intensity=500, polarization=pol)
    laser_3 = sim.Laser(4e8, intensity=1000, polarization=pol)
    inter = sim.Interaction(atom, [laser_1, ], delta_max=2000)
    inter.controlled = False
    inter.dt = 1e-3

    y0 = tools.unit_vector(0, atom.size)

    n = 1001
    sigma = 1
    fwhm = np.sqrt(8 * np.log(2)) * sigma
    v = np.random.normal(loc=0, scale=sigma, size=(n, 3))
    v = np.zeros((n, 3))

    t = time()
    # result = inter.rates(0.4)
    # result = inter.schroedinger(0.4)
    # result = inter.master(0.4, y0=None)
    result = inter.master_mc(0.4, y0=None, ntraj=500, v=v, dynamics=False)
    print('Time: {} s'.format(time() - t))
    v_ph = sc.h * p[0].freq / (atom.mass * sc.atomic_mass * sc.c) * 1e6
    x = result.x
    y = result.y
    v = result.v / v_ph
    plt.hist(v[:, 0], bins=100)
    plt.hist(v[:, 1], bins=100)
    plt.hist(v[:, 2], bins=100)
    plt.show()
    plt.plot(x, y)
    plt.plot(x, np.sum(y, axis=1), 'k--')
    plt.show()
