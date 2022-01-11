# -*- coding: utf-8 -*-
"""
PyCLS.tests.test_algebra

Created on 30.04.2021

@author: Patrick Mueller

Module including unittests for the algebra module.
"""

import unittest as ut
from sympy import S
from sympy.core.numbers import Integer
import pycol.algebra as al


class TestAlgebra(ut.TestCase):

    def test_cast_sympy(self):
        self.assertIsInstance(al.cast_sympy(False, S(0), dtype=int), int)
        self.assertIsInstance(al.cast_sympy(False, S(0), dtype=float), float)
        self.assertIsInstance(al.cast_sympy(True, S(0)), Integer)

    def test_a(self):
        self.assertEqual(al.a(3.5, 0.5, 3, 1.5, 2, as_sympy=True), S(5) / 48)
        self.assertEqual(al.a(3.5, 0.5, 3, 1.5, 3, as_sympy=True), S(7) / 64)
        self.assertEqual(al.a(3.5, 0.5, 3, 1.5, 4, as_sympy=True), S(5) / 64)
        self.assertEqual(al.a(3.5, 0.5, 4, 1.5, 3, as_sympy=True), S(7) / 192)
        self.assertEqual(al.a(3.5, 0.5, 4, 1.5, 4, as_sympy=True), S(7) / 64)
        self.assertEqual(al.a(3.5, 0.5, 4, 1.5, 5, as_sympy=True), S(11) / 48)

    def test_b(self):
        self.assertEqual(al.b(3.5, 0.5, 3, 1.5, 2, as_sympy=True), -S(1) / 336)
        self.assertEqual(al.b(3.5, 0.5, 3, 1.5, 3, as_sympy=True), -S(7) / 256)
        self.assertEqual(al.b(3.5, 0.5, 3, 1.5, 4, as_sympy=True), S(11) / 1792)
        self.assertEqual(al.b(3.5, 0.5, 4, 1.5, 3, as_sympy=True), S(7) / 2304)
        self.assertEqual(al.b(3.5, 0.5, 4, 1.5, 4, as_sympy=True), -S(77) / 6400)
        self.assertEqual(al.b(3.5, 0.5, 4, 1.5, 5, as_sympy=True), -S(143) / 3600)

    def test_c(self):
        self.assertEqual(al.c(3.5, 0.5, 3, 1.5, 2, 3, as_sympy=True), -S(1) / 64)
        self.assertEqual(al.c(3.5, 0.5, 3, 1.5, 3, 4, as_sympy=True), -S(3) / 256)
        self.assertEqual(al.c(3.5, 0.5, 3, 1.5, 4, 2, as_sympy=True), -S(15) / 448)
        self.assertEqual(al.c(3.5, 0.5, 4, 1.5, 3, 4, as_sympy=True), S(7) / 1280)
        self.assertEqual(al.c(3.5, 0.5, 4, 1.5, 4, 5, as_sympy=True), -S(77) / 1600)
        self.assertEqual(al.c(3.5, 0.5, 4, 1.5, 5, 3, as_sympy=True), -S(77) / 2880)
