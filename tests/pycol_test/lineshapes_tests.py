# -*- coding: utf-8 -*-
"""
PyCLS.tests.test_lineshapes

Created on 16.05.2021

@author: Patrick Mueller

Module including unittests for the lineshapes module.
"""

import unittest as ut
# import pycol.databases as db
import pycol.lineshapes as ls


class TestLineshapes(ut.TestCase):

    def test_shape_class(self):
        shapes = ['lorentz', 'gauss', 'voigt']
        args = [None, None, None]
        qi = [False, True]
        fixed_ratios = [False, True]
        quantum_numbers = [[0, 0.5, 1.5], [3.5, 0.5, 1.5], [2, 3, 4]]
        # db.generate_dipole_coef([3.5, ], [0.5, ], [1.5, ])
        # db.generate_dipole_coef([2, ], [3, ], [4, ])

        for shape, arg in zip(shapes, args):
            for b_qi in qi:
                for b_fr in fixed_ratios:
                    for q in quantum_numbers:
                        s0 = ls.Shape(shape, arg)
                        s = ls.HyperfineShape(shape, arg, *q)
                        s.define(qi=b_qi, fixed_ratios=b_fr)
                        m00 = ls.Model(s0, n_shape=2)
                        m0 = ls.Model(s, n_shape=2)
                        m1 = ls.Model(s, n_shape=2, x_cuts=[100, 200])
                        m2 = ls.Model(s, n_shape=3, x_cuts=[100, 200], arg_map={'s_gamma': [0, 1, 3], })
                        m3 = ls.Model(s, offset_order=3, n_shape=2, x_cuts=[100, 200],
                                      arg_map={'gamma': [0, 1], 'y2': [0, 1, 1]})
                        sm01 = ls.SumModel(m0, m1)
                        sm0123 = ls.SumModel(m0, m1, m2, m3)
                        print('\n{}({}, {}, {}){}'.format(ls.tools.COLORS.HEADER, b_qi, b_fr, q, ls.tools.COLORS.ENDC))
                        print('m00 args: {}'.format(m00.arg_list))
                        print('m0 args: {}'.format(m0.arg_list))
                        print('m1 args: {}'.format(m1.arg_list))
                        print('m2 args: {}'.format(m2.arg_list))
                        print('m3 args: {}'.format(m3.arg_list))
                        print('sm01 args: {}'.format(sm01.arg_list))
                        print('sm0123 args: {}'.format(sm0123.arg_list))
