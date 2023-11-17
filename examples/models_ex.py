# -*- coding: utf-8 -*-
"""
examples.models_ex

Created on 16.11.2023

@author: Patrick Mueller

Example script / Guide for the qspec.models module.
"""

import matplotlib.pyplot as plt
import numpy as np

import qspec.models as mod


def example(n=None):
    """
    Run one or several of the available examples. Scroll to the end for the function call.

    :param n: The number of the example or a list of numbers.

    Example 0: .

    Example 1: .
    """
    if n is None:
        n = {0, 1, 2, 3}
    if isinstance(n, int):
        n = {n, }

    if 0 in n:
        model = mod.Offset(mod.NPeak(mod.Gauss()))
        # model.names >>> ['sigma__0', 'sigma__1', 'center__0', 'int__0', 'center__1', 'int__1', 'off0e0']
        print(model.names)

    if 1 in n:
        model = mod.Offset(mod.Summed([mod.Gauss(), mod.Gauss()]))
        # model.names >>> ['sigma__0', 'sigma__1', 'center__0', 'int__0', 'center__1', 'int__1', 'off0e0']
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
        # model.names >>> ['sigma__0', 'sigma__1', 'center__0', 'int__0', 'center__1', 'int__1', 'off0e0']
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


if __name__ == '__main__':
    example({2})
