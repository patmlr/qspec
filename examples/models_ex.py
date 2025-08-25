# -*- coding: utf-8 -*-
"""
examples.models_ex
==================

Example script / Guide for the qspec.models module.
"""

import numpy as np
import matplotlib.pyplot as plt

import qspec as qs
import qspec.models as mod


def example(n=None):
    """
    Run one or several of the available examples. Scroll to the end for the function call.

    :param n: The number of the example or a list of numbers.

    Example 0: A simple Gaussian model with an x- and y-axis shift.

    Example 1: A model of two Gaussian peaks with linear dependent widths.

    Example 2: A model of linked Gaussian peaks.
    """
    if n is None:
        n = {0, 1, 2, 3}
    if isinstance(n, int):
        n = {n, }

    if 0 in n:
        model = mod.Offset(mod.NPeak(mod.Gauss()))
        # model.names >>> ['sigma', 'x0', 'p0', 'off0e0']
        print(model.names)

    if 1 in n:
        model = mod.Offset(mod.Summed([mod.Gauss(), mod.Gauss()]))
        # model.names >>> ['sigma__0', 'sigma__1', 'center__0', 'int__0', 'center__1', 'int__1', 'off0e0']
        # model.fixes >>> ['5.0(0.9)', '2 * sigma__0', False, False, False, False, False]
        print(model.names)
        model.set_fix(0, '5(0.9)')
        model.set_fix(1, '2 * sigma__0')
        print(model.fixes)

        p0 = [4, 90, -10, 0.8, 15, 1, 0]
        x = np.linspace(-40, 40, 801)
        plt.plot(x, model(x, *p0))
        plt.show()

    if 2 in n:
        models = [mod.Gauss() for _ in range(3)]
        # model.names >>> ['sigma__0', 'sigma__1', 'sigma__2']
        # model.fixes >>> [False, 'sigma__0', 'sigma__0']
        # model.links >>> [True, True, True]
        for m in models:
            m.set_link(0, True)
        model = mod.Linked(models)
        print(model.names)
        print(model.fixes)
        print(model.links)

        p0 = [3, 0, 1, 5, 20, 1]
        x = [np.linspace(-10 + i * 20, 10 + i * 20, 201) for i in range(2)]
        plt.plot(np.concatenate(x, axis=0), model(x, *p0))
        plt.show()

    if 3 in n:
        i, jl, ju = 0.5, 6., 6.
        mu = -0.2310
        gj = -1.165
        gi, gj_l, gj_u = mu / i, gj, gj
        gamma = 0.1
        al, au = -392.314, -392.314
        b_field = 0.4e-3
        # b_field = 0.
        linear = True

        models = [mod.HyperfineZeeman(
            mod.Lorentz(), i, jl, ju, gi, gj_l, gj_u, f_l=6.5, f_u=5.5, q=_q - 1, linear=linear, label=f'{_q - 1}')
            for _q in range(3)]

        x_min, x_max = [], []
        for model in models:
            model.set_val('Gamma', gamma)
            model.set_val('Al', al)
            model.set_val('Au', au)
            model.set_val('B_field', b_field)
            qs.printh(f'{model.label}:')
            print(model.names)
            print(model.vals)
            print(model.fixes)
            print(model.intervals())
            x_min.append(model.min())
            x_max.append(model.max())
        x_min = np.min(x_min)
        x_max = np.max(x_max)

        x = np.linspace(x_min, x_max, 6001)
        y = [model(x, *model.vals) for model in models]

        c = [0, 1, 4]
        labels = [r'$\sigma^-$', r'$\pi$', r'$\sigma^+$']
        for iq, _y in enumerate(y):
            plt.plot(x, _y, f'-C{c[iq]}', label=labels[iq], zorder=-100 * abs(iq - 1))

        plt.legend()
        plt.xlabel('Relative frequency (MHz)')
        plt.ylabel('Intensity (arb. units)')
        plt.subplots_adjust(left=0.12, bottom=0.11, right=0.99, top=0.99)
        plt.show()


if __name__ == '__main__':
    example({3})
