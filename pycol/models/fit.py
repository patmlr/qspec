# -*- coding: utf-8 -*-
"""
pycol.models.fit

Created on 06.05.2023

@author: Patrick Mueller

Module to fit the models to data.
"""

import numpy as np

from pycol.types import *
from pycol.tools import asarray_optional, print_colored, print_cov
# noinspection PyUnresolvedReferences
from pycol.analyze import curve_fit, odr_fit
import pycol.models.base as base


ROUTINES = {'curve_fit', 'odr_fit'}


class Xlist:  # Custom list to trick 'curve_fit' for linked fitting of files with different x-axis sizes.
    def __init__(self, x):
        self.x = x

    def __iter__(self):
        for _x in self.x:
            yield _x

    def __getitem__(self, key):
        return self.x[key]

    def __setitem__(self, key, value):
        self.x[key] = value


def residuals(model, x, y):
    return y - model(x, *model.vals)


def reduced_chi2(model, x, y, sigma_y):
    fixed, bounds = model.fit_prepare()
    n_free = sum(int(not f) for f in fixed)
    return np.sum(residuals(model, x, y) ** 2 / sigma_y ** 2) / (x.size - n_free)


def fit(model: base.Model, x: array_iter, y: array_iter, sigma_x: array_iter = None, sigma_y: array_iter = None,
        report: bool = False, routine: Union[Callable, str] = None, guess_offset: bool = False,
        mc_sigma: int = 0, **kwargs):

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    sigma_x, sigma_y = asarray_optional(sigma_x, dtype=float), asarray_optional(sigma_y, dtype=float)

    if routine is None:
        routine = curve_fit if sigma_x is None else odr_fit
    if isinstance(routine, str) and routine in ROUTINES:
        routine = eval(routine)
    if routine.__name__ not in ROUTINES:
        raise ValueError('Specified routine \'{}\' is not supported.'.format(routine.__name__))

    if not isinstance(model, base.Model):
        raise TypeError('\'model\' must have or inherit type pycol.models.Model but has type {}.'.format(type(model)))
    if model.error:
        raise ValueError(model.error)

    if isinstance(model, base.Offset):
        model.update_on_call = False
        model.gen_offset_masks(x)
        if guess_offset:
            model.guess_offset(x, y)

    p0_fixed, bounds = model.fit_prepare()

    kwargs |= {'bounds': bounds, 'sigma': sigma_y, 'sigma_x': sigma_x, 'sigma_y': sigma_y, 'report': False}

    err = False
    warn = False
    e = None
    try:
        if mc_sigma:
            kwargs['absolute_sigma'] = False
            kwargs['sigma_x'] = None
            kwargs['sigma_y'] = None

            if sigma_x is None:
                x_samples = [x, ] * mc_sigma
            else:
                x_samples = np.random.normal(loc=x, scale=sigma_x, size=(mc_sigma, x.size))
            if sigma_y is None:
                y_samples = [y, ] * mc_sigma
            else:
                y_samples = np.random.normal(loc=y, scale=sigma_y, size=(mc_sigma, y.size))

            pt = np.array([routine(model, x_sample, y_sample, p0=model.vals, p0_fixed=p0_fixed, **kwargs)[0]
                           for x_sample, y_sample in zip(x_samples, y_samples)], dtype=float)
            popt = np.mean(pt, axis=0)
            pcov = np.zeros((popt.size, popt.size))
            indices = np.array([i for i, fix in enumerate(p0_fixed) if not fix])
            mask = (indices[:, None], indices)
            pcov[mask] = np.cov(pt[:, indices], rowvar=False)
        else:
            popt, pcov = routine(model, x, y, p0=model.vals, p0_fixed=p0_fixed, **kwargs)
        popt = np.array(model.update_args(popt))
        model.set_vals(popt, force=True)
        chi2 = 0. if sigma_y is None else reduced_chi2(model, x, y, sigma_y)
    except (ValueError, RuntimeError) as e:
        warn = True
        err = True
        chi2 = 0.
        popt = np.array(model.vals, dtype=float)
        pcov = np.zeros((popt.size, popt.size))

    if report:
        digits = int(np.floor(np.log10(np.abs(model.size)))) + 1
        print('Optimized parameters:')
        for j, (name, val, err) in enumerate(zip(model.names, popt, np.sqrt(np.diag(pcov)))):
            print('{}:   {} = {} +/- {}'.format(str(j).zfill(digits), name, val, err))
        print('\nCov. Matrix:')
        print_cov(pcov, normalize=True, decimals=2)
        print('\nRed. chi2 = {}'.format(np.around(chi2, decimals=2)))

        if err:
            print_colored('FAIL', 'Error while fitting: {}\n'.format(e))
        elif np.any(np.isinf(pcov)):
            warn = True
            print_colored('WARNING', 'Failed to estimate uncertainties.\n')
        else:
            print_colored('OKGREEN', 'Fit successful.\n')

    info = dict(warn=warn, err=err, chi2=chi2)
    return popt, pcov, info
