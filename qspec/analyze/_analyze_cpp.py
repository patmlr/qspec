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

__all__ = ['generate_collinear_points']


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


def generate_collinear_points(mean: ndarray, cov: ndarray, n: int, max_samples=None, seed=None):
    mean, cov = np.ascontiguousarray(mean, dtype=float), np.ascontiguousarray(cov, dtype=float)
    size = mean.shape[0]
    dim = mean.shape[1]
    if max_samples is None:
        max_samples = 0
    else:
        max_samples = int(max_samples)
    if seed is None:
        user_seed, seed = False, 0
    else:
        user_seed, seed = True, int(seed)
    n_target = int(n)
    x = np.zeros((n, size, dim), dtype=float)
    n = c_size_t(n)
    max_samples = c_size_t(max_samples)
    dll.gen_collinear(x.ctypes.data_as(c_double_p), mean.ctypes.data_as(c_double_p), cov.ctypes.data_as(c_double_p),
                      c_size_t_p(n), c_size_t(size), c_size_t(dim),
                      c_size_t_p(max_samples), c_bool(user_seed), c_size_t(seed))
    if n.value < n_target:
        x = x[:max_samples.value]
    return x, n.value, max_samples.value
