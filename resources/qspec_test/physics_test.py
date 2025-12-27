"""
PyCLS.tests.test_physics

Created on 14.04.2021

@author: Patrick Mueller

Module including unittests for the physics module.
"""

import unittest as ut

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc

import qspec.physics as ph


class TestPhysics(ut.TestCase):
    def test_beta(self) -> None:
        self.assertEqual(ph.beta(299792458.0 / 4.0), 0.25)

    def test_gamma(self) -> None:
        self.assertAlmostEqual(float(ph.gamma(299792458.0 / 4.0)), 1.0327955589886444, places=15)

    def test_pdf(self) -> None:
        m = 40.0
        t = 290.0
        scale_e = 0.01
        e0 = 1.0
        de = 0.05
        e = np.linspace(-de, 3 * de, 2001) + e0
        dist = ph.normal_chi2_convolved_ex_pdf(e, t, scale_e, e0)
        norm = np.sum(dist) * (e[1] - e[0])
        print(f"Norm(ex): {norm}")

        scale = scale_e / (sc.k * t / sc.e)
        ex_special = scale**2 * (sc.k * t / sc.e) + e0
        dist_special = ph.normal_chi2_convolved_ex_pdf(ex_special, t, scale_e, e0)
        print(f"Special Ex: {ex_special}")
        print(f"Special value: {dist_special}")

        plt.plot(e, dist)
        plt.plot([ex_special], [dist_special], "oC1")
        plt.xlabel("Relative energy (eV)")
        plt.ylabel(r"Probability density (1/eV)")
        plt.show()

        v0 = ph.v_e(e0, m, 0.0)
        v = ph.v_e(e, m, 0.0)
        v = np.linspace(np.min(v), np.max(v), 1001)
        dist = ph.normal_chi2_convolved_vx_pdf(v, m, t, scale_e, e0, relativistic=False)
        norm = np.sum(dist) * (v[1] - v[0])

        print(f"Norm(vx): {norm}")
        plt.plot(v, dist)
        plt.xlabel("Velocity (m/s)")
        plt.ylabel(r"Probability density (s/m)")
        plt.show()

        alpha = 0.0
        f_lab = 7e8
        f0 = ph.doppler(f_lab, v0, 0.0, return_frame="atom")
        f = ph.doppler(f_lab, v, 0.0, return_frame="atom")
        f = np.linspace(np.min(f), np.max(f), 1001)
        dist = ph.normal_chi2_convolved_f_pdf(f, f_lab, alpha, m, t, scale_e, e0, relativistic=False)
        norm = np.sum(dist) * (f[1] - f[0])

        print(f"Norm(f): {norm}")
        plt.plot(f - f_lab, dist)

        xi = ph.xi_t(t, f0, e0, 1.0, m)
        sigma = scale_e * np.abs(ph.doppler_e_d1(f_lab, alpha, e0, m, 0.0, return_frame="atom"))
        col = True
        dist = ph.normal_chi2_convolved_f_xi_pdf(f, f0, xi, sigma, col)
        norm = np.sum(dist) * (f[1] - f[0])

        print(f"xi: {xi} MHz")
        print(f"Norm(f_xi): {norm}")
        plt.xlabel("Doppler shift (MHz)")
        plt.ylabel(r"Probability density (1/MHz)")
        plt.plot(f - f_lab, dist, "C1--")
        plt.show()

    # def test_inverse_doppler(self):
    #     print(ph.inverse_doppler(1e7, 1e7, 0.3))
