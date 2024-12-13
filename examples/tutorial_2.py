# -*- coding: utf-8 -*-
"""
examples.tutorial_2
===================

Tutorial 2 from the website: Calculation of hyperfine structure and Zeeman shifts.
"""

import numpy as np
import qspec as qs
import matplotlib.pyplot as plt

I = 4.5  # Total nuclear spin quantum number of 87Sr+
J = 2.5  # Total angular momentum quantum number of the 2D5/2 state
g_i = -1.09316 / I  # The nuclear g-factor [1]
g_j = qs.lande_j(s=0.5, l=2, j=2.5)
# The nuclear g-factor, calculated from the nuclear magnetic moment

A, B = 2.1743, 49.11  # (MHz), The hyperfine-structure constants [2]
b_field = np.linspace(0., 4e-3, 4000)  # (T), The magnetic flux density

e_eig, m_list, fm_list, mi_mj_list \
    = qs.hyper_zeeman_num(I, J, A, B, g_i, g_j, b_field)
# The eigenvalues of the HFS + Zeeman-effect Hamiltonian
# and the lists of m_F, F and (m_I, m_J) quantum numbers.


""" Plot """
f_colored = False  # (F, m) or (mI, mJ) color scheme.

f_plotted = set()
m_plotted = set()
cmap = plt.get_cmap('inferno')
handles, labels = [], []
for im, (_e_eig, _f_list, _m, mi_mj) in enumerate(zip(e_eig, fm_list, m_list, mi_mj_list)):
    for k in range(_e_eig.shape[1]):
        if f_colored:
            c_val = (_f_list[k] - abs(I - J)) / (I + J - abs(I - J) + 1)
            c = cmap(c_val)
            plt.plot(b_field * 1e3, _e_eig[:, k], color=c, ls='-', lw=0.85, alpha=0.8,
                     label=rf'$F = {int(_f_list[k])}$' if c not in f_plotted else None, zorder=10 * c_val)
            f_plotted.add(c)
            handles, labels = plt.gca().get_legend_handles_labels()
        else:
            mj = mi_mj[k][1]
            c_val = (mj + J) / (2 * J + 1)
            c = cmap(c_val)
            plt.plot(b_field * 1e3, _e_eig[:, k], color=c, ls='-', lw=0.85, alpha=0.8,
                     label=rf'$m_J = {mj}$' if mj not in m_plotted else None, zorder=10 * c_val)
            handles, labels = plt.gca().get_legend_handles_labels()
            handles, labels = handles[::-1], labels[::-1]
            m_plotted.add(mj)

plt.xlabel('Magnetic field (mT)')
plt.ylabel('Frequency shift (MHz)')
plt.legend(handles, labels)
plt.xlim(0., 4.)
plt.subplots_adjust(left=0.11, bottom=0.1, right=0.98, top=0.99)
plt.show()
