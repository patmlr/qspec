# -*- coding: utf-8 -*-
"""
PyCLS.tests.test_lineshapes

Created on 16.05.2021

@author: Patrick Mueller

Module including unittests for the lineshapes module.
"""

import unittest as ut
import pycol.lineshapes as ls


class TestLineshapes(ut.TestCase):

    def test_lineshapes(self):
        shapes = ['lorentz', 'gauss', 'voigt']
        qi = [False, True]
        quantum_numbers = [[0, 0.5, 1.5], [3.5, 0.5, 1.5], [2, 3, 4]]
        for shape in shapes:
            for qn in quantum_numbers:
                model = ls.gen_model(qn, ls.spectrum.Lorentz)
                print()
                print(model.names)
                print(model.vals)
                print(model.fixes)
            model = ls.gen_model(quantum_numbers, shape)
            print()
            print(model.names)
            print(model.vals)
            print(model.fixes)
