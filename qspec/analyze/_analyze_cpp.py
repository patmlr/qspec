# -*- coding: utf-8 -*-
"""
qspec._analyze_cpp
==================

Created on 24.02.2024

@author: Patrick Mueller

Classes and methods for the 'analyze' module using the Python/C++ interface.
"""

import numpy as np

from qspec._types import *
from qspec._cpp import *

__all__ = ['generate_collinear_points_cpp']


class MultivariateNormal:
    def __init__(self, mean: array_like, cov: array_like, instance=None):
        self.instance = instance
        if self.instance is None:
            mean, cov = np.ascontiguousarray(mean, dtype=float), np.ascontiguousarray(cov, dtype=float)
            self.instance = dll.multivariatenormal_construct(
                mean.ctypes.data_as(c_double_p), cov.ctypes.data_as(c_double_p), c_size_t(mean.size))

    def __del__(self):
        dll.multivariatenormal_destruct(self.instance)

    @property
    def size(self):
        return dll.multivariatenormal_size(self.instance)

    def rvs(self):
        ret = np.zeros(self.size, dtype=float)
        dll.multivariatenormal_rvs(self.instance, ret.ctypes.data_as(c_double_p))
        return ret


def generate_collinear_points_cpp(mean: ndarray, cov: ndarray, n_samples: int = 100000, n_accepted: int = None,
                                  seed: int = None):
    mean, cov = np.ascontiguousarray(mean, dtype=float), np.ascontiguousarray(cov, dtype=float)
    size = mean.shape[0]
    dim = mean.shape[1]
    if n_samples is None:
        n_samples = 0
    else:
        n_samples = int(n_samples)
    if n_accepted is None:
        n_accepted = max(100000, n_samples)
    else:
        n_accepted = int(n_accepted)
    if seed is None:
        user_seed, seed = False, 0
    else:
        user_seed, seed = True, int(seed)
    n_target = n_accepted
    x = np.zeros((n_accepted, size, dim), dtype=float)
    n_accepted = c_size_t(n_accepted)
    n_samples = c_size_t(n_samples)
    dll.gen_collinear(x.ctypes.data_as(c_double_p), mean.ctypes.data_as(c_double_p), cov.ctypes.data_as(c_double_p),
                      c_size_t_p(n_accepted), c_size_t(size), c_size_t(dim),
                      c_size_t_p(n_samples), c_bool(user_seed), c_size_t(seed))
    if n_accepted.value < n_target:
        x = x[:n_accepted.value]
    return x, n_accepted.value, n_samples.value
