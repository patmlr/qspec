# -*- coding: utf-8 -*-
"""
PyCLS.tests.test_physics

Created on 14.04.2021

@author: Patrick Mueller

Module including unittests for the physics module.
"""

import unittest as ut
import qspec.physics as ph


class TestPhysics(ut.TestCase):

    def test_beta(self):
        self.assertEqual(ph.beta(299792458. / 4.), 0.25)

    def test_gamma(self):
        self.assertAlmostEqual(ph.gamma(299792458. / 4.), 1.0327955589886444, places=15)
