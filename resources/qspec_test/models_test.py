# -*- coding: utf-8 -*-
"""
PyCLS.tests.test_lineshapes

Created on 16.05.2021

@author: Patrick Mueller

Module including unittests for the lineshapes module.
"""

import unittest as ut
import qspec.models as mod


class TestLineshapes(ut.TestCase):

    def test_lineshapes(self):
        shapes = ['lorentz', 'gauss', 'voigt']
        qi = [False, True]
        quantum_numbers = [[0, 0.5, 1.5], [3.5, 0.5, 1.5], [2, 3, 4]]
        for shape in shapes:
            for qn in quantum_numbers:
                model = mod.gen_model(qn, mod.Lorentz)
                print()
                print(model.names)
                print(model.vals)
                print(model.fixes)
            model = mod.gen_model(quantum_numbers, shape)
            print()
            print(model.names)
            print(model.vals)
            print(model.fixes)
